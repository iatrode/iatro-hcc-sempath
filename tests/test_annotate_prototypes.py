from __future__ import annotations

import io
import json
import random
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from PIL import Image

from hcc_sempath.cli.annotate_prototypes import (
    AnnotationData,
    AnnotationArchive,
    AnnotationState,
    HTML,
    L1_PROTOTYPES,
    L2_PROTOTYPES,
    ROI_L2_PROTOTYPES,
    RoiCandidateQueue,
    SharedPriorityQueue,
    StrictReviewQueue,
    _annotation_parser,
    _auth_ok,
    _annotation_key,
    _load_or_create_auth_token,
    _validate_brush_geometry,
    discover_iac_packages,
    make_handler,
)
from hcc_sempath.cli.build_priority_list import build_priority_manifest
from hcc_sempath.cli.build_roi_queue import build_roi_candidate_queue
from iatro.iac.adapters.features import build_teacher_feature_package
from iatro.iac.adapters.manifests import TileRecord
from iatro.iac.adapters.tiles import build_tile_package_from_records, encode_jxl_array


def _records() -> list[TileRecord]:
    return [
        TileRecord(
            tile_id=f"s1_{idx:07d}",
            patient_id="p1",
            slide_id="s1",
            tile_path=Path(f"tiles/s1_{idx:07d}.jxl"),
            x=x * 16,
            y=y * 16,
            split="train",
        )
        for idx, (x, y) in enumerate((x, y) for y in range(2) for x in range(2))
    ]


def _write_iac(path: Path) -> None:
    records = _records()
    payloads = []
    for idx, _record in enumerate(records):
        arr = np.full((16, 16, 3), 40 + idx * 20, dtype=np.uint8)
        payloads.append(encode_jxl_array(arr, lossless=True, distance=None, effort=1))
    build_tile_package_from_records(
        records,
        payloads,
        path,
        tile_width=16,
        tile_height=16,
        stride_x=16,
        stride_y=16,
        lossless=True,
        effort=1,
        overwrite=True,
    )


def _write_iac_with_stride(path: Path, stride: int) -> None:
    records = [
        TileRecord(
            tile_id=f"s1_{idx:07d}",
            patient_id="p1",
            slide_id="s1",
            tile_path=Path(f"tiles/s1_{idx:07d}.jxl"),
            x=x * stride,
            y=y * stride,
            split="train",
        )
        for idx, (x, y) in enumerate((x, y) for y in range(2) for x in range(2))
    ]
    payloads = [
        encode_jxl_array(np.full((16, 16, 3), 60 + idx * 30, dtype=np.uint8), lossless=True, distance=None, effort=1)
        for idx, _record in enumerate(records)
    ]
    build_tile_package_from_records(
        records,
        payloads,
        path,
        tile_width=16,
        tile_height=16,
        stride_x=stride,
        stride_y=stride,
        lossless=True,
        effort=1,
        overwrite=True,
    )


def test_discover_iac_packages_supports_direct_file_and_dataset_dirs(tmp_path: Path) -> None:
    direct = tmp_path / "direct.iac"
    dataset_path = tmp_path / "cohort_a" / "nested.iac"
    dataset_path.parent.mkdir()
    _write_iac(direct)
    _write_iac(dataset_path)

    single = discover_iac_packages(direct)
    assert single[0].rel_path == "direct.iac"
    assert single[0].dataset == ""

    packages = discover_iac_packages(tmp_path)
    by_name = {package.rel_path: package for package in packages}
    assert by_name["direct.iac"].dataset == ""
    assert by_name["cohort_a/nested.iac"].dataset == "cohort_a"
    assert by_name["cohort_a/nested.iac"].total == 4

    included = discover_iac_packages(
        tmp_path,
        ("cohort_a/nested.iac",),
    )
    assert [package.rel_path for package in included] == [
        "cohort_a/nested.iac"
    ]
    included_by_glob = discover_iac_packages(
        tmp_path,
        ("cohort_a/*.iac",),
    )
    assert [package.rel_path for package in included_by_glob] == [
        "cohort_a/nested.iac"
    ]


def test_discover_iac_packages_follows_symlinked_dataset_dirs(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "cohort_a"
    input_dir = tmp_path / "input"
    source_path = source_dir / "nested.iac"
    source_dir.mkdir(parents=True)
    input_dir.mkdir()
    _write_iac(source_path)
    (input_dir / "cohort_link").symlink_to(source_dir, target_is_directory=True)

    packages = discover_iac_packages(input_dir)

    assert [(package.rel_path, package.dataset, package.total) for package in packages] == [
        ("cohort_link/nested.iac", "cohort_link", 4)
    ]


def test_auth_token_requires_exact_nonempty_match() -> None:
    assert _auth_ok("secret", "secret")
    assert not _auth_ok("", "secret")
    assert not _auth_ok("secret ", "secret")
    assert not _auth_ok("other", "secret")


def test_annotation_cli_always_requires_l1_and_l2_roi_workspaces() -> None:
    parser = _annotation_parser()
    args = parser.parse_args([
        "--input", "tiles",
        "--l1-state", "l1.json",
        "--l2-state", "l2.json",
        "--priority-manifest", "priority.json",
        "--l1-review-manifest", "l1-review.json",
        "--l2-review-manifest", "l2-review.json",
        "--include-packages", "301/a.iac,301/b.iac",
        "--roi-candidate-manifest", "roi-candidates.json",
        "--roi-priority-attributes", "vascular-structure-present,ductular-portal-present",
        "--review-existing",
    ])
    assert (args.l1_state, args.l2_state, args.priority_manifest) == (
        "l1.json", "l2.json", "priority.json",
    )
    assert not any("--state" in action.option_strings for action in parser._actions)
    assert args.roi_candidate_manifest == "roi-candidates.json"
    assert args.l1_review_manifest == "l1-review.json"
    assert args.l2_review_manifest == "l2-review.json"
    assert args.include_packages == "301/a.iac,301/b.iac"
    assert args.roi_priority_attributes == "vascular-structure-present,ductular-portal-present"
    assert args.review_existing is True
    assert not any("--roi-plan-config" in action.option_strings for action in parser._actions)
    assert not any("--roi-plan-checkpoint" in action.option_strings for action in parser._actions)
    assert not any("--roi-plan-device" in action.option_strings for action in parser._actions)
    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "tiles", "--l1-state", "l1.json"])


def test_roi_ui_complete_review_is_dynamic_and_only_required_in_roi_mode() -> None:
    assert "Unmarked classes stay unknown/ignored unless explicitly toggled negative." in HTML
    assert "ROI conflict: marked negative but has ROI annotation" in HTML
    assert "Positive or uncertain by default" in HTML
    assert "%ROI_MODE_JSON%" in HTML
    assert 'id="roiClassBar"' in HTML
    assert "function applyModeLayout()" in HTML
    assert "document.getElementById('roiCanvas').style.display=l1Mode?'none':'block'" in HTML
    assert "document.getElementById('roiTools').style.display=l1Mode?'none':'flex'" in HTML
    assert "document.getElementById('prototypeLabels').style.display=l1Mode?'block':'none'" in HTML
    assert "document.getElementById('l1Section').style.display=l1Mode?'block':'none'" in HTML
    assert "document.getElementById('l2Section').style.display='none'" in HTML
    assert "MODE==='l1'?'L1 classification':'L2 ROI annotation'" in HTML
    assert "index<9?String(index+1):''" in HTML
    assert "class=chipShortcut" in HTML
    assert "await showRecord(rec.record);closeOverview();" in HTML
    assert "horizontalWheelPan:true" in HTML
    assert "Math.abs(ev.deltaX)>=2" in HTML
    assert "contextTrackpadWheel=true" in HTML
    assert "if(!contextTrackpadWheel)return" in HTML
    assert "horizontalWheelUntil" not in HTML
    assert "ev.key.toLowerCase()==='c'" in HTML
    assert "dialogOpen=document.querySelector('dialog[open]')" in HTML
    assert "MODE==='l1'&&!mod&&!editing&&!dialogOpen" in HTML
    assert "/^[1-9]$/.test(ev.key)" in HTML
    assert "l1=L1[index];renderLabels()" in HTML
    assert "ev.code==='Space'" in HTML
    assert "if(!ev.repeat)save()" in HTML
    assert "All ROI classes visible. Select one L2 class before drawing." in HTML
    assert "currentCandidate" not in HTML
    assert "function hasPositiveRoi(attribute)" in HTML
    assert "function roiClassIcon(attribute,state)" in HTML
    assert "hasPositiveRoi(attribute)?'●':'○'" in HTML
    assert "Positive ROI present" in HTML
    assert "overflow-x:auto" in HTML
    assert 'id="thumb" class="thumb panzoom" draggable="false"' in HTML
    assert "el.addEventListener('dragstart',ev=>ev.preventDefault())" in HTML
    assert "if(ev.button!==0)return;ev.preventDefault();measure()" in HTML
    assert "This review list is complete." in HTML
    assert "Review list ${review.reviewed}/${review.total}" in HTML
    assert 'class="imageControlRow tileControlRow"' in HTML
    assert 'class="primaryWorkspace"' in HTML
    assert ".tileViewport,.tileStage,.tile,.roiCanvas{-webkit-user-select:none;user-select:none" in HTML
    assert "-webkit-user-drag:none" in HTML
    assert '<img id="tile" class="tile" draggable="false">' in HTML
    assert "grid-template-columns:minmax(420px,680px) minmax(300px,1fr)" in HTML
    assert ".annotationControls{grid-column:2;grid-row:1/span 2" in HTML
    assert "@media(max-width:1200px)" in HTML
    assert ".primaryWorkspace{display:block}" in HTML
    assert ".roiClassBar{display:flex;flex-wrap:wrap" in HTML
    assert ".roiClassBar{flex-wrap:nowrap;max-width:min(100%,760px);overflow-x:auto" in HTML
    assert 'id="tileZoom"' in HTML
    assert 'id="overviewBtn" type="button">Overview' in HTML
    assert 'id="overviewOverlay" class="contextOverlay hidden"' in HTML
    assert 'class="imageControlRow overviewControlRow"' not in HTML
    assert '<b>Location overview</b>' in HTML
    assert 'id="overviewZoom"' in HTML
    assert ".thumbWrap{position:relative;width:100%;height:100%;min-height:0" in HTML
    assert ".overviewStage{min-height:0;flex:1;overflow:auto" in HTML
    assert 'id="tileZoomIn"' not in HTML
    assert 'id="roiRedo"' in HTML
    assert "undoRoi()" in HTML
    assert "redoRoi()" in HTML
    assert "roiPreview" in HTML
    assert "ev.key.toLowerCase()==='z'" in HTML
    assert "ev.key.toLowerCase()==='y'" in HTML
    assert 'id="brushWidth"' in HTML
    assert 'type="range"' in HTML
    assert 'id="brushDot"' not in HTML
    assert 'max="0.500"' in HTML
    assert "updateBrushWidth()" in HTML
    assert "brushWidth()" in HTML
    assert "getElementById('roiCanvas').addEventListener('wheel'" in HTML
    assert "if(!ROI_MODE||roiTool!=='brush')return;ev.preventDefault();setBrushWidth(brushWidth()" in HTML
    assert "getElementById('tileViewport').addEventListener('wheel'" not in HTML
    assert "function selectRoiTool(tool,remember=true)" in HTML
    assert 'data-roi-tool="eraser">Eraser' in HTML
    assert "['point','brush','eraser','circle']" in HTML
    assert "roiToolByClass[roiAttribute]=roiTool" in HTML
    assert "const AREA_ONLY_CLASSES=new Set(['necrosis-present','fibrous-stroma-present'])" in HTML
    assert "function roiToolAllowed(tool,attribute=roiAttribute)" in HTML
    assert "AREA_ONLY_CLASSES.has(name)?'brush':'point'" in HTML
    assert 'id="roiNavigationNotice"' in HTML
    assert "candidates reviewed; the information curve needs more coverage" in HTML
    assert "Brush / eraser width" in HTML
    assert "function densifyPath(points,maxStep)" in HTML
    assert "function eraseBrushGeometry(item,center,eraserRadius)" in HTML
    assert "function eraseCurrentClassAt(center,eraserRadius)" in HTML
    assert "function eraseCurrentClassAlong(start,end,eraserRadius)" in HTML
    assert "function setupEraser()" in HTML
    assert "function finishRoiStroke(ev=null)" in HTML
    assert "const finalPoint=[p.x,p.y],lastPoint=drawing.points" in HTML
    assert "pointDistanceSq(lastPoint,finalPoint)>1e-12" in HTML
    assert "window.addEventListener('pointerup',ev=>finishRoiStroke(ev),true)" in HTML
    assert "window.addEventListener('pointercancel',ev=>finishRoiStroke(ev),true)" in HTML
    assert "addEventListener('lostpointercapture'" in HTML
    assert "drawing.last||drawing.start" in HTML
    assert "p=ev?roiPoint(ev):(drawing.last||drawing.start)" in HTML
    assert "roiPoint(ev,true)" not in HTML
    assert "roiDrawing.last=p" in HTML
    assert "window.addEventListener('mouseup',()=>finishRoiStroke(),true)" in HTML
    assert "function setupTileUndo()" in HTML
    assert "getElementById('tileStage').addEventListener('contextmenu'" in HTML
    assert "ev.preventDefault();ev.stopPropagation();undoRoi()" in HTML
    assert "if(ev.button!==0)return" in HTML
    assert "if(ev.button!==0||roiTool!=='eraser')return" in HTML
    assert "setupEraser();setupRoi();setupTileUndo()" in HTML
    assert "item.attribute!==roiAttribute" in HTML
    assert "pushUndo();roiPlan=null;roiDrawing={tool:'eraser',pointerId:ev.pointerId" in HTML
    assert "syncPositiveLabels();renderRoi()" in HTML
    assert "wheelZoom:false,horizontalWheelPan:true,doubleClickReset:false,onScale:syncOverviewZoom" in HTML
    assert "if(!wheelZoom)return;ev.preventDefault();setScale(" in HTML
    assert "overviewZoom.addEventListener('input'" in HTML
    assert "function openOverview()" in HTML
    assert "function closeOverview()" in HTML
    assert "if(!overviewIsOpen())return" in HTML
    assert 'id="contextViewport"' in HTML
    assert "const CONTEXT_RADIUS=5,CONTEXT_RECENTER_DELTA=2" in HTML
    assert "requestAnimationFrame(paintContextPan)" in HTML
    assert "translate3d(${contextTx}px,${contextTy}px,0)" in HTML
    assert "Math.min(availableW/(visible*contextData.cell_width),availableH/(visible*contextData.cell_height))" in HTML
    assert ".contextLoading{position:absolute" in HTML
    assert "pointer-events:none" in HTML
    assert "repeating-linear-gradient(135deg" in HTML
    assert "--context-grid-x" in HTML
    assert "if(contextDrag){" in HTML
    assert "contextPendingWindow={data,image,centerRow,focus}" in HTML
    assert "/api/context-center?" in HTML
    assert "contextTx=contextDrag.startTx+dx;contextTy=contextDrag.startTy+dy;\n        scheduleContextPan()" in HTML
    assert "viewport.clientWidth-baseW*scale" in HTML
    assert "translate3d(${tx}px,${ty}px,0) scale(${scale})" in HTML
    assert "/api/context-window?" in HTML
    assert "async function openContextTile(record)" in HTML
    assert "await showRecord(record)" in HTML
    assert "replenishContextBuffer()" in HTML
    assert "roiCursor" in HTML
    assert "updateRoiCursor" in HTML
    assert "exclude_row" in HTML
    assert "Priority list ${priority.reviewed}/${priority.total}" in HTML
    assert 'id="roiPlanGenerate"' in HTML
    assert 'id="roiPlanAccept"' in HTML
    assert 'id="roiPlanRestart"' in HTML
    assert 'id="roiSimilarity"' in HTML
    assert 'min="0.20" max="0.99" step="0.01" value="0.70"' in HTML
    assert "const NUCLEUS_MATCH_CLASSES=new Set" in HTML
    assert "Image-only nucleus matching is not validated for this ROI class." in HTML
    assert "setRoiSimilarityThreshold(result.summary?.recommended_similarity" in HTML
    assert 'id="roiExclusion"' in HTML
    assert 'min="3" max="40" step="1" value="7"' in HTML
    assert "async function generateRoiPlan()" in HTML
    assert "function acceptRoiPlan()" in HTML
    assert "function restartRoiFromScratch()" in HTML
    assert "function seedCentersForSelectedClass()" in HTML
    assert "function occupiedRoiCenters()" in HTML
    assert "occupied:occupiedRoiCenters()" in HTML
    assert "function visiblePlannedSuggestions()" in HTML
    assert "target=roiPlan?.summary?.preview_target??suggestions.length" in HTML
    assert "if(selected.length>=target)break" in HTML
    assert "function roiExclusionDistance()" in HTML
    assert "function plannedCenter(item)" in HTML
    assert "roiExclusionOverrides[roiAttribute]" in HTML
    assert "occupied.some(point=>" in HTML
    assert "selectedCenters.some(point=>" in HTML
    assert "roiExclusion').addEventListener('input'" in HTML
    assert "Preview marks are not saved yet." in HTML
    assert "Matching each selected seed as an independent local image template." in HTML
    assert "Nucleus-image preview" in HTML
    assert "minimum_spacing_pixels" in HTML
    assert "'/api/roi-similar'" in HTML
    assert "function drawPlannedPoint(x,g,color)" in HTML
    assert "function drawPointMark(x,point,color,alpha=1)" in HTML
    assert "[[3.4,'#000'],[2.3,'#fff'],[1.2,color]]" in HTML
    assert "x.globalAlpha=.5" in HTML
    assert "x.moveTo(px-3.5,py);x.lineTo(px+3.5,py)" in HTML
    assert "x.moveTo(px,py-3.5);x.lineTo(px,py+3.5)" in HTML
    assert "[[2.5,'rgba(15,23,42,.9)'],[1.5,'#fff'],[.7,color]]" in HTML
    assert "function drawPlannedCircle(x,g,color)" in HTML
    assert "[[7,'rgba(15,23,42,.9)'],[5,'#fff'],[2.5,color]]" in HTML
    assert "Choose Accept visible matches or Start from scratch before saving." in HTML


def _write_roi_queue(path: Path, tile_ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "l2_prototypes": ROI_L2_PROTOTYPES,
                "candidates": [
                    {
                        "tile_id": tile_id,
                        "rank": rank,
                        "source_l2": [ROI_L2_PROTOTYPES[rank % len(ROI_L2_PROTOTYPES)]],
                    }
                    for rank, tile_id in enumerate(tile_ids)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_shared_priority_list_drives_roi_then_expands_to_fallback_tile(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "roi.json"
    priority_path = tmp_path / "priority.json"
    _write_iac(iac_path)
    priority_path.write_text(json.dumps({"version": 1, "candidates": [{"tile_id": "s1_0000000", "iac": "tiles.iac", "row": 0, "slide": "s1"}]}), encoding="utf-8")
    priority = SharedPriorityQueue(priority_path)
    data = AnnotationData(iac_path, state_path, roi_mode=True, priority_queue=priority)
    try:
        assert data.state.l2_prototypes == ROI_L2_PROTOTYPES
        assert len(data.annotation_records(0)) == 4
        package = data.package(0)
        record = data.random_record(0)["record"]
        assert record["tile_id"] == "s1_0000000"
        iac_record = data.viewer(0)._by_row[record["row"]]
        geometry = {
            "attribute": ROI_L2_PROTOTYPES[0],
            "state": "positive",
            "geometry": {"type": "point", "coordinate_space": "normalized", "point": [0.5, 0.5]},
        }
        data.state.save_annotation(
            package,
            iac_record,
            L1_PROTOTYPES[0],
            [ROI_L2_PROTOTYPES[0]],
            [geometry],
        )
        fallback = data.random_record(0)["record"]
        assert fallback is not None
        assert fallback["tile_id"] != "s1_0000000"
        assert priority.contains(fallback["tile_id"])
        assert data.progress(0)["roi_counts"][ROI_L2_PROTOTYPES[0]] == 1
        assert data.progress(0)["priority"] == {"reviewed": 1, "skipped": 0, "total": 2, "remaining": 1}
        saved = data.annotation_json(0, iac_record.row)["annotation"]
        assert saved["roi_reviewed"] is True
        assert saved["roi_complete_all"] is False

        with pytest.raises(ValueError, match="both positive and negative"):
            data.state.save_annotation(
                package,
                iac_record,
                L1_PROTOTYPES[0],
                [ROI_L2_PROTOTYPES[0]],
                [geometry, {"attribute": ROI_L2_PROTOTYPES[0], "state": "negative", "review_complete": True}],
            )
    finally:
        data.close()


def test_strict_review_list_revisits_existing_tiles_and_stops_without_fallback(
    tmp_path: Path,
) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "l1.json"
    priority_path = tmp_path / "priority.json"
    review_path = tmp_path / "review.json"
    _write_iac(iac_path)
    priority_payload = {
        "version": 1,
        "candidates": [
            {
                "tile_id": f"s1_{row:07d}",
                "iac": "tiles.iac",
                "row": row,
                "slide": "s1",
            }
            for row in range(4)
        ],
    }
    priority_path.write_text(json.dumps(priority_payload), encoding="utf-8")
    review_path.write_text(
        json.dumps(
            {
                "version": 1,
                "review_id": "tumor-differentiation-v1",
                "candidates": priority_payload["candidates"][:2],
            }
        ),
        encoding="utf-8",
    )
    priority_before = priority_path.read_text(encoding="utf-8")
    priority = SharedPriorityQueue(priority_path)
    review = StrictReviewQueue(review_path)
    seed = AnnotationData(iac_path, state_path)
    package = seed.package(0)
    first_record = seed.viewer(0)._by_row[0]
    seed.state.save_annotation(
        package,
        first_record,
        L1_PROTOTYPES[0],
        [],
    )
    seed.close()

    data = AnnotationData(
        iac_path,
        state_path,
        priority_queue=priority,
        review_queue=review,
    )
    try:
        first = data.random_record("all")
        assert first["record"]["tile_id"] == "s1_0000000"
        assert first["selection"] == "strict_review_manifest"
        data.state.save_skip(
            data.package(0),
            data.viewer(0)._by_row[0],
            review_id=review.review_id,
        )
        assert data.state.annotation_for(
            data.package(0),
            data.viewer(0)._by_row[0],
        )["l1"] == L1_PROTOTYPES[0]

        second = data.random_record("all")
        assert second["record"]["tile_id"] == "s1_0000001"
        second_record = data.viewer(0)._by_row[1]
        data.state.save_annotation(
            data.package(0),
            second_record,
            L1_PROTOTYPES[1],
            [],
            review_id=review.review_id,
        )
        assert data.random_record("all") == {
            "record": None,
            "done": "review_complete",
        }
        assert data.progress(0)["review"] == {
            "review_id": "tumor-differentiation-v1",
            "reviewed": 1,
            "skipped": 1,
            "total": 2,
            "remaining": 0,
        }
        assert priority_path.read_text(encoding="utf-8") == priority_before
    finally:
        data.close()

    reloaded = AnnotationData(
        iac_path,
        state_path,
        review_queue=StrictReviewQueue(review_path),
    )
    try:
        assert reloaded.random_record("all")["done"] == "review_complete"
    finally:
        reloaded.close()


def test_brush_validation_keeps_edge_overlap_and_rejects_invalid_geometry() -> None:
    _validate_brush_geometry(
        {
            "type": "brush",
            "coordinate_space": "normalized",
            "points": [[-0.04, 0.4], [-0.02, 0.6]],
            "width": 0.10,
        }
    )

    with pytest.raises(ValueError, match="does not intersect"):
        _validate_brush_geometry(
            {
                "type": "brush",
                "coordinate_space": "normalized",
                "points": [[-0.20, 0.4], [-0.15, 0.6]],
                "width": 0.05,
            }
        )
    with pytest.raises(ValueError, match="finite positive width"):
        _validate_brush_geometry(
            {
                "type": "brush",
                "coordinate_space": "normalized",
                "points": [[0.2, 0.4]],
                "width": float("nan"),
            }
        )
    with pytest.raises(ValueError, match="finite x/y"):
        _validate_brush_geometry(
            {
                "type": "brush",
                "coordinate_space": "normalized",
                "points": [[float("inf"), 0.4]],
                "width": 0.05,
            }
        )


def test_old_l2_candidates_bias_random_navigation_without_prefilling_labels(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "roi.json"
    queue_path = tmp_path / "roi_candidates.json"
    _write_iac(iac_path)
    queue_path.write_text(
        json.dumps(
            {
                "version": 1,
                "l2_prototypes": ROI_L2_PROTOTYPES,
                "candidates": [
                    {
                        "tile_id": "s1_0000000",
                        "source_l2": ["fibrous-stroma-present"],
                    },
                    {
                        "tile_id": "s1_0000001",
                        "source_l2": ["vascular-structure-present"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    data = AnnotationData(
        iac_path,
        state_path,
        roi_candidate_manifest=queue_path,
        roi_priority_attributes=["vascular-structure-present"],
        min_tissue_fraction=0,
    )
    try:
        result = data.random_record(0)
        assert result["record"]["tile_id"] == "s1_0000001"
        assert result["selection"] == "component_presence_navigation_hint"
        assert "source_l2" not in result
        assert data.state.annotations == {}
    finally:
        data.close()


def test_roi_navigation_priority_attribute_order_changes_sampling_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "roi_candidates.json"
    queue_path.write_text(
        json.dumps(
            {
                "version": 1,
                "l2_prototypes": ROI_L2_PROTOTYPES,
                "candidates": [
                    {"tile_id": "vascular", "source_l2": ["vascular-structure-present"]},
                    {"tile_id": "bile", "source_l2": ["bile-pigment-present"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    queue = RoiCandidateQueue(queue_path)
    monkeypatch.setattr(random, "random", lambda: 0.5)
    counts = {name: 0 for name in ROI_L2_PROTOTYPES}

    vascular_first = queue.weighted_remaining(
        processed_tile_ids=set(),
        roi_positive_counts=counts,
        priority_attributes=["vascular-structure-present", "bile-pigment-present"],
    )
    bile_first = queue.weighted_remaining(
        processed_tile_ids=set(),
        roi_positive_counts=counts,
        priority_attributes=["bile-pigment-present", "vascular-structure-present"],
    )

    assert vascular_first[0]["tile_id"] == "vascular"
    assert bile_first[0]["tile_id"] == "bile"


def test_roi_navigation_has_no_fixed_positive_tile_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "roi_candidates.json"
    attribute = "vascular-structure-present"
    queue_path.write_text(
        json.dumps(
            {
                "version": 2,
                "l2_prototypes": ROI_L2_PROTOTYPES,
                "candidates": [
                    {"tile_id": "vascular", "source_l2": [attribute]},
                ],
            }
        ),
        encoding="utf-8",
    )
    queue = RoiCandidateQueue(queue_path)
    monkeypatch.setattr(random, "random", lambda: 0.5)

    remaining = queue.weighted_remaining(
        processed_tile_ids=set(),
        roi_positive_counts={attribute: 10_000},
        priority_attributes=[attribute],
    )

    assert [item["tile_id"] for item in remaining] == ["vascular"]


def _write_roi_information_report(
    path: Path,
    *,
    deficient: dict[str, str],
) -> None:
    attributes = {}
    for name in ROI_L2_PROTOTYPES:
        status = deficient.get(name, "provisionally_stable")
        support = 0.0 if status == "not_assessable" else 0.5
        attributes[name] = {
            "status": status,
            "teacher_low_gain_support_by_teacher_ratio": {
                "0.35": {
                    teacher: support
                    for teacher in (
                        "gigapath",
                        "h_optimus_1",
                        "uni2_h",
                        "virchow2",
                    )
                }
            },
        }
    path.write_text(json.dumps({"attributes": attributes}), encoding="utf-8")


def test_roi_navigation_uses_information_deficit_inside_old_l2_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "roi_candidates.json"
    report_path = tmp_path / "roi_information_report.json"
    queue_path.write_text(
        json.dumps(
            {
                "l2_prototypes": ROI_L2_PROTOTYPES,
                "candidates": [
                    {
                        "tile_id": "necrosis",
                        "source_l2": ["necrosis-present"],
                    },
                    {
                        "tile_id": "fibrous",
                        "source_l2": ["fibrous-stroma-present"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_roi_information_report(
        report_path,
        deficient={"necrosis-present": "still_growing"},
    )
    queue = RoiCandidateQueue(
        queue_path,
        information_report_path=report_path,
    )
    monkeypatch.setattr(random, "random", lambda: 0.5)

    remaining = queue.weighted_remaining(
        processed_tile_ids=set(),
        roi_positive_counts={name: 0 for name in ROI_L2_PROTOTYPES},
    )

    assert [item["tile_id"] for item in remaining] == [
        "necrosis",
        "fibrous",
    ]
    assert remaining[0]["dynamic_priority_attributes"][0] == (
        "necrosis-present"
    )


def test_roi_navigation_learns_cross_label_yield_when_direct_hints_are_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "roi_candidates.json"
    report_path = tmp_path / "roi_information_report.json"
    candidates = [
        {
            "tile_id": f"reviewed-fibrous-{index}",
            "source_l2": ["fibrous-stroma-present"],
        }
        for index in range(4)
    ]
    candidates.extend(
        {
            "tile_id": f"reviewed-ductular-{index}",
            "source_l2": ["ductular-portal-present"],
        }
        for index in range(4)
    )
    candidates.extend(
        [
            {
                "tile_id": "remaining-fibrous",
                "source_l2": ["fibrous-stroma-present"],
            },
            {
                "tile_id": "remaining-ductular",
                "source_l2": ["ductular-portal-present"],
            },
        ]
    )
    queue_path.write_text(
        json.dumps(
            {
                "l2_prototypes": ROI_L2_PROTOTYPES,
                "candidates": candidates,
            }
        ),
        encoding="utf-8",
    )
    _write_roi_information_report(
        report_path,
        deficient={"vascular-structure-present": "not_assessable"},
    )
    queue = RoiCandidateQueue(
        queue_path,
        information_report_path=report_path,
    )
    monkeypatch.setattr(random, "random", lambda: 0.5)
    processed = {
        item["tile_id"]
        for item in candidates
        if item["tile_id"].startswith("reviewed-")
    }
    positive_by_tile = {
        tile_id: (
            {"vascular-structure-present"}
            if tile_id.startswith("reviewed-fibrous-")
            else set()
        )
        for tile_id in processed
    }

    remaining = queue.weighted_remaining(
        processed_tile_ids=processed,
        roi_positive_counts={name: 0 for name in ROI_L2_PROTOTYPES},
        roi_positive_by_tile=positive_by_tile,
        priority_attributes=["vascular-structure-present"],
    )

    assert [item["tile_id"] for item in remaining] == [
        "remaining-fibrous",
        "remaining-ductular",
    ]
    policy = queue.sampling_policy()["vascular-structure-present"]
    assert policy["remaining_direct_hint_count"] == 0
    assert policy["predicted_remaining_yield"] > 0


def test_roi_progress_exposes_explicit_navigation_target(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    queue_path = tmp_path / "roi_candidates.json"
    report_path = tmp_path / "roi_information_report.json"
    _write_iac(iac_path)
    queue_path.write_text(
        json.dumps(
            {
                "l2_prototypes": ROI_L2_PROTOTYPES,
                "candidates": [
                    {
                        "tile_id": "s1_0000000",
                        "source_l2": ["steatosis-vacuolation-present"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_roi_information_report(
        report_path,
        deficient={
            "steatosis-vacuolation-present": "still_growing",
            "vascular-structure-present": "not_assessable",
        },
    )
    data = AnnotationData(
        iac_path,
        tmp_path / "roi.json",
        roi_candidate_manifest=queue_path,
        roi_information_report=report_path,
        roi_priority_attributes=["steatosis-vacuolation-present"],
        roi_mode=True,
        min_tissue_fraction=0,
    )
    try:
        assert data.progress(0)["roi_priority_attributes"] == [
            "steatosis-vacuolation-present"
        ]
    finally:
        data.close()


def test_roi_navigation_lists_direct_deficient_hints_before_inferred_hints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "roi_candidates.json"
    report_path = tmp_path / "roi_information_report.json"
    queue_path.write_text(
        json.dumps(
            {
                "l2_prototypes": ROI_L2_PROTOTYPES,
                "candidates": [
                    {
                        "tile_id": "direct",
                        "source_l2": ["vascular-structure-present"],
                    },
                    {
                        "tile_id": "inferred",
                        "source_l2": ["fibrous-stroma-present"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_roi_information_report(
        report_path,
        deficient={"vascular-structure-present": "not_assessable"},
    )
    queue = RoiCandidateQueue(
        queue_path,
        information_report_path=report_path,
    )
    random_values = iter([0.01, 0.99])
    monkeypatch.setattr(random, "random", lambda: next(random_values))

    remaining = queue.weighted_remaining(
        processed_tile_ids=set(),
        roi_positive_counts={name: 0 for name in ROI_L2_PROTOTYPES},
    )

    assert [item["tile_id"] for item in remaining] == [
        "direct",
        "inferred",
    ]
    assert remaining[0]["direct_priority_attributes"] == [
        "vascular-structure-present"
    ]


def test_roi_navigation_reports_exhausted_old_l2_class(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "roi_candidates.json"
    queue_path.write_text(
        json.dumps(
            {
                "l2_prototypes": ROI_L2_PROTOTYPES,
                "candidates": [
                    {
                        "tile_id": "vascular",
                        "source_l2": ["vascular-structure-present"],
                    },
                    {
                        "tile_id": "necrosis",
                        "source_l2": ["necrosis-present"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    queue = RoiCandidateQueue(queue_path)

    status = queue.direct_hint_status({"vascular"})

    assert status["vascular-structure-present"] == {
        "total": 1,
        "reviewed": 1,
        "remaining": 0,
        "exhausted": True,
    }
    assert status["necrosis-present"]["exhausted"] is False
    assert status["necrosis-present"]["remaining"] == 1


def test_roi_navigation_hints_stay_inside_shared_priority_boundary(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    queue_path = tmp_path / "roi_candidates.json"
    priority_path = tmp_path / "priority.json"
    _write_iac(iac_path)
    queue_path.write_text(
        json.dumps(
            {
                "version": 1,
                "l2_prototypes": ROI_L2_PROTOTYPES,
                "candidates": [
                    {"tile_id": "s1_0000001", "source_l2": ["vascular-structure-present"]},
                    {"tile_id": "s1_0000002", "source_l2": ["vascular-structure-present"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    priority_path.write_text(
        json.dumps(
            {
                "version": 1,
                "candidates": [
                    {"tile_id": "s1_0000002", "iac": "tiles.iac", "row": 2, "slide": "s1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    data = AnnotationData(
        iac_path,
        tmp_path / "roi.json",
        roi_candidate_manifest=queue_path,
        roi_priority_attributes=["vascular-structure-present"],
        priority_queue=SharedPriorityQueue(priority_path),
        min_tissue_fraction=0,
    )
    try:
        result = data.random_record("all")
        assert result["record"]["tile_id"] == "s1_0000002"
        assert result["selection"] == "component_presence_navigation_hint"
    finally:
        data.close()


def test_build_roi_queue_excludes_hyaline_and_reports_planning_coverage(tmp_path: Path) -> None:
    source = tmp_path / "annotations.json"
    source.write_text(
        json.dumps(
            {
                "annotations": {
                    "a": {"tile_id": "t1", "iac": "a.iac", "row": 1, "slide": "s1", "l2": [ROI_L2_PROTOTYPES[0], ROI_L2_PROTOTYPES[1]]},
                    "b": {"tile_id": "t2", "iac": "a.iac", "row": 2, "slide": "s1", "l2": [ROI_L2_PROTOTYPES[1], "hyaline-change-present"]},
                }
            }
        ),
        encoding="utf-8",
    )
    payload = build_roi_candidate_queue([source], planning_coverage=2)
    assert "hyaline-change-present" not in payload["l2_prototypes"]
    assert payload["candidate_count"] == 2
    assert payload["version"] == 2
    assert payload["planning_coverage_per_attribute"][ROI_L2_PROTOTYPES[0]] == 2
    assert payload["source_positive_inventory"][ROI_L2_PROTOTYPES[0]] == 1
    assert payload["unfilled_planning_coverage"][ROI_L2_PROTOTYPES[0]] == 1
    assert payload["selected_source_coverage"][ROI_L2_PROTOTYPES[1]] == 2
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(payload), encoding="utf-8")
    assert RoiCandidateQueue(queue_path).contains("t1")


def test_build_priority_manifest_keeps_only_tile_identity_and_deduplicates(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps({"annotations": {"a": {"tile_id": "t1", "iac": "a.iac", "row": 1, "slide": "s1", "l1": "x", "l2": ["auxiliary-label"]}}}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"annotations": {"a2": {"tile_id": "t1", "iac": "a.iac", "row": 1, "slide": "s1"}, "b": {"tile_id": "t2", "iac": "b.iac", "row": 2, "slide": "s2"}}}),
        encoding="utf-8",
    )

    payload = build_priority_manifest([first, second])

    assert payload["candidate_count"] == 2
    assert [item["tile_id"] for item in payload["candidates"]] == ["t1", "t2"]
    assert all("l1" not in item and "l2" not in item for item in payload["candidates"])


def test_package_list_and_progress_do_not_open_iac_viewers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "iac"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    _write_iac(root / "a" / "tiles.iac")
    _write_iac(root / "b" / "tiles.iac")
    state_path = tmp_path / "annotations.json"
    data = AnnotationData(root, state_path)
    try:
        def fail_viewer(_index: int):
            raise AssertionError("package list/progress must not open IAC viewers")

        monkeypatch.setattr(data, "viewer", fail_viewer)
        packages = data.package_json()
        assert len(packages) == 2
        assert packages[0]["total"] == 4
        progress = data.progress(0)
        assert progress["overall"]["total"] == 8
        assert progress["package"]["total"] == 4
    finally:
        data.close()


def test_auth_token_reuses_state_sidecar_and_creates_private_file(tmp_path: Path) -> None:
    state_path = tmp_path / "annotations.json"
    token_path = tmp_path / "annotations.auth-token"
    token_path.write_text("persisted-token\n", encoding="utf-8")

    assert _load_or_create_auth_token(state_path) == "persisted-token"
    assert _load_or_create_auth_token(state_path, "provided-token") == "provided-token"

    token_path.unlink()
    created = _load_or_create_auth_token(state_path)
    assert created
    assert token_path.read_text(encoding="utf-8").strip() == created
    assert token_path.stat().st_mode & 0o777 == 0o600


def test_annotation_http_requires_auth_token(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)
    data = AnnotationData(iac_path, state_path)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(data, "secret"))
    except PermissionError:
        data.close()
        pytest.skip("local socket bind is blocked in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root_url = f"http://127.0.0.1:{server.server_address[1]}/"
    url = f"{root_url}api/packages"
    try:
        with urlopen(root_url, timeout=5) as response:
            assert response.status == 200
        try:
            urlopen(url, timeout=5)
            raise AssertionError("unauthenticated request unexpectedly succeeded")
        except HTTPError as exc:
            assert exc.code == 403
        with urlopen(f"{url}?token=secret", timeout=5) as response:
            assert response.status == 200
        with urlopen(f"{url}?auth_token=secret", timeout=5) as response:
            assert response.status == 200
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        data.close()


def test_l2_roi_similarity_endpoint_uses_seeds_without_saving(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)
    data = AnnotationData(iac_path, state_path, roi_mode=True)

    class FakePlanGenerator:
        def __init__(self) -> None:
            self.tile_bytes = b""

        def generate_similar(
            self,
            tile_bytes: bytes,
            *,
            attribute: str,
            seeds: list,
            occupied: list,
        ) -> dict:
            self.tile_bytes = tile_bytes
            assert attribute == ROI_L2_PROTOTYPES[0]
            assert seeds == [[0.4, 0.6]]
            assert occupied == [[0.4, 0.6], [0.2, 0.3]]
            return {
                "version": 1,
                "suggestions": [{
                    "attribute": ROI_L2_PROTOTYPES[0],
                    "state": "positive",
                    "geometry": {
                        "type": "point",
                        "coordinate_space": "normalized",
                        "point": [0.5, 0.5],
                    },
                }],
                "summary": {"suggestion_count": 1, "seed_count": 1},
            }

    generator = FakePlanGenerator()
    try:
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(data, "secret", generator)
        )
    except PermissionError:
        data.close()
        pytest.skip("local socket bind is blocked in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = (
        f"http://127.0.0.1:{server.server_address[1]}"
        "/api/roi-similar?token=secret&mode=l2"
    )
    request = Request(
        url,
        data=json.dumps({
            "package": 0,
            "row": 0,
            "attribute": ROI_L2_PROTOTYPES[0],
            "seeds": [[0.4, 0.6]],
            "occupied": [[0.4, 0.6], [0.2, 0.3]],
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            payload = json.load(response)
        assert payload["summary"]["suggestion_count"] == 1
        assert generator.tile_bytes.startswith(b"\x89PNG")
        assert data.state.annotations == {}
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        data.close()


def test_discover_iac_packages_skips_teacher_feature_packages(tmp_path: Path) -> None:
    image_path = tmp_path / "tiles.iac"
    feature_path = tmp_path / "teacher.features.iac"
    records = _records()
    _write_iac(image_path)
    build_teacher_feature_package(
        records,
        [np.arange(4, dtype=np.float32) for _ in records],
        feature_path,
        teacher_name="smoke",
        overwrite=True,
    )

    packages = discover_iac_packages(tmp_path)
    assert [package.rel_path for package in packages] == ["tiles.iac"]


def test_annotation_state_writes_resume_json_and_csv(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)

    data = AnnotationData(iac_path, state_path)
    try:
        package = data.packages[0]
        record = data.viewer(0).records[0]
        data.state.save_annotation(
            package,
            record,
            L1_PROTOTYPES[0],
            [L2_PROTOTYPES[0], L2_PROTOTYPES[3]],
            [
                {
                    "attribute": L2_PROTOTYPES[0],
                    "state": "positive",
                    "geometry": {
                        "type": "point",
                        "coordinate_space": "normalized",
                        "point": [0.5, 0.5],
                    },
                }
            ],
        )
    finally:
        data.close()

    reloaded = AnnotationState(state_path, iac_path)
    assert len(reloaded.annotations) == 1
    assert reloaded.last_iac == "tiles.iac"
    assert next(iter(reloaded.annotations.values()))["roi"][0]["geometry"]["type"] == "point"
    csv_text = state_path.with_suffix(".csv").read_text(encoding="utf-8")
    assert "HCC-tumor" in csv_text
    assert "hepatocellular-parenchyma-present;bile-pigment-present" in csv_text


def test_annotation_state_uses_taxonomy_from_json_and_observed_labels(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)
    package = discover_iac_packages(iac_path)[0]
    viewer = AnnotationData(iac_path, state_path)
    try:
        record = viewer.viewer(0).records[0]
        state_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "input_path": str(iac_path),
                    "l1_prototypes": ["HCC-tumor", "Background-liver"],
                    "l2_prototypes": ["hepatocellular-parenchyma-present"],
                    "annotations": {
                        _annotation_key(package, record): {
                            "dataset": package.dataset,
                            "iac": package.rel_path,
                            "iac_path": str(package.path),
                            "tile_id": record.tile_id,
                            "row": record.row,
                            "slide": record.slide_label,
                            "x": record.display_x,
                            "y": record.display_y,
                            "l1": "Inflammatory-stromal",
                            "l2": ["ductular-portal-present"],
                        }
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        viewer.close()

    data = AnnotationData(iac_path, state_path)
    try:
        assert data.state.l1_prototypes == ["HCC-tumor", "Background-liver", "Inflammatory-stromal"]
        assert data.state.l2_prototypes == ["hepatocellular-parenchyma-present", "ductular-portal-present"]
        second = data.viewer(0).records[1]
        data.state.save_annotation(package, second, "Inflammatory-stromal", ["ductular-portal-present"])
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        assert payload["l1_prototypes"] == data.state.l1_prototypes
        assert payload["l2_prototypes"] == data.state.l2_prototypes
    finally:
        data.close()


def test_csv_export_preserves_review_fields_without_crashing(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)

    data = AnnotationData(iac_path, state_path)
    try:
        package = data.packages[0]
        record = data.viewer(0).records[0]
        data.state.save_annotation(package, record, L1_PROTOTYPES[0], [])
        key = _annotation_key(package, record)
        data.state.annotations[key]["reviewed"] = True
        data.state.annotations[key]["review_decision"] = "adjust"
        data.state.annotations[key]["review_suggested_l1"] = L1_PROTOTYPES[1]
        data.state.flush()
    finally:
        data.close()

    csv_text = state_path.with_suffix(".csv").read_text(encoding="utf-8")
    assert "review_decision" in csv_text
    assert "adjust" in csv_text


def test_scan_status_reports_last_iac_from_state(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "input_path": str(iac_path),
                "last_iac": "tiles.iac",
                "annotations": {},
            }
        ),
        encoding="utf-8",
    )

    data = AnnotationData(iac_path, state_path)
    try:
        assert data.scan_status()["last_iac"] == "tiles.iac"
    finally:
        data.close()


def test_progress_reports_overall_annotation_counts(tmp_path: Path) -> None:
    first_path = tmp_path / "a.iac"
    second_path = tmp_path / "cohort_b" / "b.iac"
    second_path.parent.mkdir()
    state_path = tmp_path / "annotations.json"
    _write_iac(first_path)
    _write_iac(second_path)

    data = AnnotationData(tmp_path, state_path)
    try:
        package = data.packages[0]
        record = data.viewer(0).records[0]
        data.state.save_annotation(package, record, L1_PROTOTYPES[0], [])

        progress = data.progress(0)
        assert progress["package"] == {"annotated": 1, "total": 4, "remaining": 3, "skipped": 0}
        assert progress["overall"] == {"annotated": 1, "total": 8, "remaining": 7, "skipped": 0}
    finally:
        data.close()


def test_progress_excludes_skipped_annotations(tmp_path: Path) -> None:
    first_path = tmp_path / "a.iac"
    second_path = tmp_path / "b.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(first_path)
    _write_iac(second_path)

    data = AnnotationData(tmp_path, state_path)
    try:
        first_package = data.packages[0]
        second_package = data.packages[1]
        first_records = data.viewer(0).records
        second_record = data.viewer(1).records[0]
        data.state.save_annotation(first_package, first_records[0], L1_PROTOTYPES[0], [])

        data.state.skipped.add(_annotation_key(first_package, first_records[1]))
        data.state.skipped.add(_annotation_key(second_package, second_record))

        progress = data.progress(0)
        assert progress["package"] == {"annotated": 1, "total": 4, "remaining": 2, "skipped": 1}
        assert progress["overall"] == {"annotated": 1, "total": 8, "remaining": 5, "skipped": 2}
        assert progress["l1"][L1_PROTOTYPES[0]] == 1
    finally:
        data.close()


def test_progress_label_counts_cover_all_packages(tmp_path: Path) -> None:
    first_path = tmp_path / "a.iac"
    second_path = tmp_path / "b.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(first_path)
    _write_iac(second_path)

    data = AnnotationData(tmp_path, state_path)
    try:
        first_package = data.packages[0]
        second_package = data.packages[1]
        first_record = data.viewer(0).records[0]
        second_record = data.viewer(1).records[0]
        data.state.save_annotation(first_package, first_record, L1_PROTOTYPES[0], [L2_PROTOTYPES[0]])
        data.state.save_annotation(second_package, second_record, L1_PROTOTYPES[1], [L2_PROTOTYPES[1]])

        progress = data.progress(0)
        assert progress["l1"][L1_PROTOTYPES[0]] == 1
        assert progress["l1"][L1_PROTOTYPES[1]] == 1
        assert progress["l2"][L2_PROTOTYPES[0]] == 1
        assert progress["l2"][L2_PROTOTYPES[1]] == 1
        assert progress["package_l1"][L1_PROTOTYPES[1]] == 0
    finally:
        data.close()


def test_top_level_skipped_records_are_preserved_and_not_sampled(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)

    data = AnnotationData(iac_path, state_path)
    try:
        package = data.packages[0]
        skipped_record = data.viewer(0).records[0]
        skipped_key = _annotation_key(package, skipped_record)
    finally:
        data.close()

    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "input_path": str(iac_path.resolve()),
                "last_iac": package.rel_path,
                "l1_prototypes": L1_PROTOTYPES,
                "l2_prototypes": L2_PROTOTYPES,
                "annotations": {},
                "skipped": [skipped_key],
            }
        ),
        encoding="utf-8",
    )

    data = AnnotationData(iac_path, state_path)
    try:
        package = data.packages[0]
        skipped_record = data.viewer(0).records[0]
        assert data.state.is_annotated(package, skipped_record)

        progress = data.progress(0)
        assert progress["package"] == {"annotated": 0, "total": 4, "remaining": 3, "skipped": 1}

        seen_rows = {data.random_record(0)["record"]["row"] for _ in range(30)}
        assert skipped_record.row not in seen_rows

        data.state.save_annotation(package, skipped_record, L1_PROTOTYPES[0], [])
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        assert payload["last_iac"] == package.rel_path
        assert skipped_key not in payload["skipped"]
    finally:
        data.close()


def test_random_record_skips_annotated_tiles_and_overview_is_jpg(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)

    data = AnnotationData(iac_path, state_path)
    try:
        package = data.packages[0]
        first = data.viewer(0).records[0]
        data.state.save_annotation(package, first, L1_PROTOTYPES[1], [])

        seen_rows = {data.random_record(0)["record"]["row"] for _ in range(30)}
        assert first.row not in seen_rows
        current = data.viewer(0).records[1]
        excluded_rows = {data.random_record(0, exclude_row=current.row)["record"]["row"] for _ in range(30)}
        assert current.row not in excluded_rows
        assert data.thumbnail_jpg(0).startswith(b"\xff\xd8")
        assert data.overview_cache_path(package).exists()
    finally:
        data.close()


def test_reviewed_records_lists_saved_annotations_but_not_skips(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)
    data = AnnotationData(iac_path, state_path)
    try:
        package = data.packages[0]
        records = data.viewer(0).records
        data.state.save_annotation(package, records[0], L1_PROTOTYPES[0], [])
        data.state.save_skip(package, records[1])
        data.state.save_annotation(
            package, records[2], L1_PROTOTYPES[1], [L2_PROTOTYPES[0]]
        )

        result = data.reviewed_records("all")

        assert result["total"] == 2
        assert [item["record"]["row"] for item in result["items"]] == [0, 2]
        assert result["items"][1]["l2"] == [L2_PROTOTYPES[0]]
    finally:
        data.close()


def test_context_jpg_renders_half_resolution_5x5_grid(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)

    data = AnnotationData(iac_path, state_path)
    try:
        image = Image.open(io.BytesIO(data.context_jpg(0, 0))).convert("RGB")
        assert image.size == (40, 40)
        assert image.getpixel((20, 20)) != (245, 247, 249)
        window = data.context_window(0, 0, radius=5)
        assert window["grid_size"] == 11
        assert window["visible_grid_size"] == 5
        assert len(window["cells"]) == 121
        center = next(
            item
            for item in window["cells"]
            if item["dx"] == 0 and item["dy"] == 0
        )
        right = next(
            item
            for item in window["cells"]
            if item["dx"] == 1 and item["dy"] == 0
        )
        assert center["record"]["row"] == 0
        assert right["record"]["row"] == 1
        assert data.context_center(
            0,
            0,
            grid_x=1,
            grid_y=0,
        )["record"]["row"] == 1
        prefetched = Image.open(
            io.BytesIO(
                data.context_jpg(
                    0,
                    0,
                    radius=5,
                    selected_row=0,
                )
            )
        ).convert("RGB")
        assert prefetched.size == (88, 88)
        thumb = Image.open(io.BytesIO(data.thumbnail_jpg(0, selected_row=0))).convert("RGB")
        assert thumb.size == (8, 8)
    finally:
        data.close()


def test_overview_uses_fixed_cell_grid_for_spatial_layout(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac_with_stride(iac_path, stride=64)

    data = AnnotationData(iac_path, state_path)
    try:
        image = Image.open(io.BytesIO(data.thumbnail_jpg(0))).convert("RGB")
        assert image.size == (8, 8)
        assert image.getpixel((2, 2)) != (255, 255, 255)
    finally:
        data.close()


def test_save_skip_persists_state(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)

    data = AnnotationData(iac_path, state_path)
    try:
        package = data.packages[0]
        record0 = data.viewer(0).records[0]
        record1 = data.viewer(0).records[1]

        # 1. Save annotation on record0
        data.state.save_annotation(package, record0, L1_PROTOTYPES[0], [])
        assert _annotation_key(package, record0) in data.state.annotations

        # 2. Skip record0 (should be a no-op because it's annotated)
        data.state.save_skip(package, record0)
        assert _annotation_key(package, record0) not in data.state.skipped

        # 3. Skip record1 (should succeed)
        data.state.save_skip(package, record1)
        assert _annotation_key(package, record1) in data.state.skipped

        # Verify persisted state on disk
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        assert _annotation_key(package, record1) in payload["skipped"]
        assert _annotation_key(package, record0) not in payload["skipped"]
    finally:
        data.close()


def test_skip_via_http_post(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)
    data = AnnotationData(iac_path, state_path)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(data, "secret"))
    except PermissionError:
        data.close()
        pytest.skip("local socket bind is blocked in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root_url = f"http://127.0.0.1:{server.server_address[1]}/"
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{root_url}api/skip?token=secret",
            data=json.dumps({"package": 0, "row": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            assert response.status == 200
            res_payload = json.loads(response.read().decode("utf-8"))
            assert res_payload == {"ok": True}

        package = data.packages[0]
        record = data.viewer(0).records[1]
        assert _annotation_key(package, record) in data.state.skipped
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        data.close()


def test_endpoints_via_http_get(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)
    data = AnnotationData(iac_path, state_path)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(data, "secret"))
    except PermissionError:
        data.close()
        pytest.skip("local socket bind is blocked in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root_url = f"http://127.0.0.1:{server.server_address[1]}/"
    try:
        with urlopen(f"{root_url}api/random?token=secret&package=all", timeout=5) as response:
            assert "record" in json.loads(response.read().decode("utf-8"))
        with urlopen(f"{root_url}api/progress?token=secret&package=0", timeout=5) as response:
            assert "package" in json.loads(response.read().decode("utf-8"))
        with urlopen(f"{root_url}api/annotation-state?token=secret&package=0&row=0", timeout=5) as response:
            assert "candidate" in json.loads(response.read().decode("utf-8"))
        with urlopen(f"{root_url}api/reviewed-list?token=secret&package=all", timeout=5) as response:
            assert json.loads(response.read().decode("utf-8")) == {"items": [], "total": 0}
        with urlopen(
            f"{root_url}api/context-window?token=secret&package=0&row=0&radius=5",
            timeout=5,
        ) as response:
            context = json.loads(response.read().decode("utf-8"))
            assert context["grid_size"] == 11
            assert context["visible_grid_size"] == 5
        with urlopen(
            f"{root_url}api/context-center?token=secret&package=0&row=0&grid_x=1&grid_y=0",
            timeout=5,
        ) as response:
            assert json.loads(response.read().decode("utf-8"))["record"]["row"] == 1
        with urlopen(f"{root_url}api/tile?token=secret&package=0&row=0", timeout=5) as response:
            assert len(response.read()) > 0
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        data.close()


def test_label_lifecycle_preserves_stable_ids_and_referenced_labels(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)
    data = AnnotationData(iac_path, state_path)
    try:
        package = data.package(0)
        record = data.viewer(0).records[0]
        label_id = L1_PROTOTYPES[0]
        data.state.save_annotation(package, record, label_id, [])
        data.state.change_label("l1", "rename", label_id=label_id, name="HCC tumor custom")
        recycled = data.state.change_label("l1", "add", name=L1_PROTOTYPES[0])
        assert any(item["name"] == L1_PROTOTYPES[0] for item in recycled["levels"]["l1"])
        assert data.state.annotations[_annotation_key(package, record)]["l1"] == label_id
        with pytest.raises(ValueError, match="archive it instead"):
            data.state.change_label("l1", "delete", label_id=label_id)
        data.state.change_label("l1", "archive", label_id=label_id)
        data.state.save_annotation(package, record, label_id, [])
        result = data.state.change_label("l1", "add", name="New morphology")
        added = next(item for item in result["levels"]["l1"] if item["name"] == "New morphology")
        data.state.change_label("l1", "delete", label_id=added["id"])
    finally:
        data.close()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert "label_definitions" in payload
    assert "l1_name" in state_path.with_suffix(".csv").read_text(encoding="utf-8").splitlines()[0]


def test_label_definitions_are_independent_between_versions_after_reload(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)
    main = AnnotationData(iac_path, state_path)
    archive = AnnotationArchive(main, input_path=iac_path)
    try:
        version_id = archive.create_version("marking")["created"]
        archive.data(version_id).state.change_label("l1", "rename", label_id=L1_PROTOTYPES[0], name="G1")
    finally:
        archive.close()

    reopened = AnnotationData(iac_path, state_path)
    reloaded_archive = AnnotationArchive(reopened, input_path=iac_path)
    try:
        assert reloaded_archive.default_version == version_id
        assert reloaded_archive.data("main").state.label_definitions["l1"][0]["name"] == L1_PROTOTYPES[0]
        assert reloaded_archive.data(version_id).state.label_definitions["l1"][0]["name"] == "G1"
    finally:
        reloaded_archive.close()


def test_label_editor_has_explicit_version_scoped_submit() -> None:
    assert 'id="saveLabels"' in HTML
    assert "Save label configuration" in HTML
    assert "Saved to this version." in HTML
    assert "input.dataset.labelId=item.id" in HTML
    assert "label management · ${VERSION}" in HTML
    assert "function labelDrafts()" in HTML
    assert "async function savePendingLabelNames()" in HTML
    assert "let pendingLabelAdds=[];" in HTML
    assert "Added as a draft. Save label configuration to commit." in HTML
    assert "for(const name of pendingLabelAdds)await changeLabel(MODE,'add','',name,true);" in HTML
    assert "renderLabelEditor(drafts)" in HTML


def test_previous_tile_uses_session_history_not_row_order() -> None:
    assert "tileHistory=[]" in HTML
    assert "if(current)tileHistory.push({package:pkg,record:current});" in HTML
    assert "const previous=tileHistory.pop();" in HTML
    assert "No previous tile in this session." in HTML
    assert "textContent='Previous tile'" in HTML


def test_marked_tile_list_is_collapsed_and_loaded_on_demand() -> None:
    assert '<details id="reviewedDetails" class="reviewedPanel">' in HTML
    assert '<details id="reviewedDetails" class="reviewedPanel" open>' not in HTML
    assert "/api/reviewed-list?package=all" in HTML
    assert "async function openReviewed(item)" in HTML
    assert "if(ev.target.open)loadReviewedList()" in HTML
    assert "let reviewedReturn=null,reviewedItems=[],reviewedIndex=-1;" in HTML
    assert "async function returnFromReviewed()" in HTML
    assert "else returnFromReviewed()" in HTML
    assert "height:min(36svh,360px);max-height:min(36svh,360px)" in HTML
    assert "overflow-x:hidden;overflow-y:scroll" in HTML
    assert "touch-action:pan-y" in HTML


def test_review_mode_is_shared_by_l1_and_l2_and_uses_saved_tile_order() -> None:
    assert "%REVIEW_MODE_JSON%" in HTML
    assert 'id="reviewModeBtn"' in HTML
    assert ".review-mode #packageHeader,.review-mode #packages{display:none}" in HTML
    assert "let reviewedReturn=null,reviewedItems=[],reviewedIndex=-1;" in HTML
    assert "async function reviewStep(delta)" in HTML
    assert "if(REVIEW_MODE){await reviewStep(-1);return}" in HTML
    assert "if(REVIEW_MODE){await reviewStep(1);return}" in HTML
    assert "url.searchParams.set('review','1')" in HTML
    assert "document.getElementById('reviewedDetails').open=true" in HTML


def test_clear_selected_roi_class_clears_geometry_negative_and_preview_state() -> None:
    assert "function clearSelectedRoiClass()" in HTML
    assert "roi=roi.filter(x=>x.attribute!==cleared)" in HTML
    assert "delete roiClassState[cleared]" in HTML
    assert "roiDrawing=null;roiPreview=null;roiCursor=null" in HTML
    assert "document.getElementById('roiClear').onclick=clearSelectedRoiClass" in HTML


def test_annotation_ui_remembers_the_selected_version() -> None:
    assert "function versionStorageKey()" in HTML
    assert "hcc_sempath_annotation_version:${MODE}" in HTML
    assert "localStorage.setItem(versionStorageKey(),ev.target.value)" in HTML
    assert "localStorage.setItem(`hcc_sempath_annotation_version:${MODE}`,VERSION)" in HTML


def test_unified_handler_exposes_navigation_and_versions(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    queue_path = tmp_path / "queue.json"
    _write_iac(iac_path)
    _write_roi_queue(queue_path, ["s1_0000000"])
    l1_data = AnnotationData(iac_path, tmp_path / "l1.json")
    l2_data = AnnotationData(iac_path, tmp_path / "l2.json", roi_candidate_manifest=queue_path)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler({"l1": l1_data, "l2": l2_data}, "secret"))
    except PermissionError:
        l1_data.close()
        l2_data.close()
        pytest.skip("local socket bind is blocked in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root_url = f"http://127.0.0.1:{server.server_address[1]}/"
    try:
        with urlopen(f"{root_url}?mode=l2&version=main", timeout=5) as response:
            html = response.read().decode("utf-8")
        assert 'const MODE="l2"' in html
        assert 'const MODES=["l1", "l2"]' in html
        assert 'const VERSION="main"' in html
        assert "%VERSIONS_JSON%" not in html
        with urlopen(f"{root_url}api/versions?token=secret&mode=l1&version=main", timeout=5) as response:
            versions = json.loads(response.read().decode("utf-8"))
        assert versions["versions"][0]["id"] == "main"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        l1_data.close()
        l2_data.close()


def test_annotation_versions_are_independent_and_reloadable(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    main_path = tmp_path / "l1.json"
    _write_iac(iac_path)
    initial = AnnotationData(iac_path, main_path, min_tissue_fraction=0)
    archive = AnnotationArchive(initial, input_path=iac_path, min_tissue_fraction=0)
    package = initial.package(0)
    initial.state.save_annotation(package, initial.viewer(0).records[0], L1_PROTOTYPES[0], [])
    created = archive.create_version("Second review")
    version_id = created["created"]
    second = archive.data(version_id)
    assert second.state.annotations == {}
    assert second.state.label_definitions == initial.state.label_definitions
    second.state.save_annotation(second.package(0), second.viewer(0).records[1], L1_PROTOTYPES[1], [])
    assert len(initial.state.annotations) == len(second.state.annotations) == 1
    assert set(initial.state.annotations) != set(second.state.annotations)
    archive.close()
    reloaded = AnnotationArchive(AnnotationData(iac_path, main_path, min_tissue_fraction=0), input_path=iac_path, min_tissue_fraction=0)
    try:
        assert {item["id"] for item in reloaded.versions_json()["versions"]} == {"main", version_id}
        assert len(reloaded.data("main").state.annotations) == 1
        assert len(reloaded.data(version_id).state.annotations) == 1
    finally:
        reloaded.close()


def test_random_record_filters_low_tissue_without_marking_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)
    data = AnnotationData(iac_path, state_path, min_tissue_fraction=0.30)
    monkeypatch.setattr(random, "shuffle", lambda values: None)
    monkeypatch.setattr(data, "tissue_fraction", lambda _index, record: 0.05 if record.row == 0 else 0.75)
    try:
        result = data.random_record(0)
        assert result["record"]["row"] == 1
        assert ("tiles.iac", 0) in data._auto_filtered
        assert data.state.skipped == set()
        assert data.progress(0)["auto_filtered"] == 1
    finally:
        data.close()


def test_random_record_reports_when_all_tiles_are_below_tissue_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)
    data = AnnotationData(iac_path, state_path, min_tissue_fraction=0.30)
    monkeypatch.setattr(data, "tissue_fraction", lambda _index, _record: 0.0)
    try:
        result = data.random_record(0)
        assert result == {"record": None, "done": "no_tissue_candidates"}
        assert len(data._auto_filtered) == 4
        assert data.state.skipped == set()
    finally:
        data.close()
