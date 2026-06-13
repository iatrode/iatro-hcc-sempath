from __future__ import annotations

import argparse
import csv
import json
import math
import random
import socket
import sys
import tempfile
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hcc_sempath.cli.annotate_prototypes import L1_PROTOTYPES
from hcc_sempath.cli.view_iac import IacViewerData


DEFAULT_UNSTABLE_L1 = {
    "Degenerative-material",
}


@dataclass(frozen=True)
class ReviewCandidate:
    key: str
    tile_id: str
    current_l1: str
    suggested_l1: str
    uncertainty: float
    score_summary: str


def _find_free_port(host: str, preferred: int) -> int:
    if preferred:
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _load_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("annotations"), dict):
        raise ValueError(f"annotation JSON missing annotations object: {path}")
    return payload


def _load_disagreement_payload(paths: list[Path], seed: int) -> dict:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        filename = path.name.lower()
        if "random" in filename:
            source_group = "random500"
        else:
            source_group = "top500"

        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                tile_id = str(row.get("tile_id") or "").strip()
                package_path = str(row.get("package_path") or "").strip()
                row_idx = str(row.get("row_idx") or "").strip()
                if not tile_id or not package_path or not row_idx:
                    raise ValueError(f"disagreement CSV row missing tile_id/package_path/row_idx: {path}")
                if tile_id in seen:
                    raise ValueError(f"duplicate disagreement tile_id across inputs: {tile_id}")
                seen.add(tile_id)
                item = dict(row)
                item["source_group"] = source_group
                rows.append(item)
    if not rows:
        raise ValueError("disagreement CSV inputs contain no review rows")

    random.Random(seed).shuffle(rows)
    annotations: dict[str, dict] = {}
    for index, row in enumerate(rows, start=1):
        review_id = f"TD-{index:04d}"
        item = dict(row)
        item.update(
            {
                "review_id": review_id,
                "iac_path": row["package_path"],
                "row": int(row["row_idx"]),
                "l1": "",
                "reviewed": False,
            }
        )
        annotations[review_id] = item
    return {
        "review_type": "teacher_disagreement_l1_adjudication",
        "blind_seed": int(seed),
        "source_csvs": [str(path) for path in paths],
        "l1_prototypes": list(L1_PROTOTYPES),
        "annotations": annotations,
    }


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def _package_keys(path: Path) -> list[str]:
    name = path.name
    suffixes = [
        ".prov-gigapath-local.features.iac",
        ".uni2_h.features.iac",
        ".virchow2.features.iac",
        ".features.iac",
    ]
    keys = []
    for suffix in suffixes:
        if name.endswith(suffix):
            keys.append(name[: -len(suffix)])
    keys.append(path.stem)
    return list(dict.fromkeys(key for key in keys if key))


def _tile_keys(tile_id: str) -> list[str]:
    keys = [tile_id]
    if "_" in tile_id:
        keys.append(tile_id.rsplit("_", 1)[0])
    return list(dict.fromkeys(keys))


def _discover_teacher_paths(root: Path, teachers: list[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for teacher in teachers:
        direct = root / teacher
        search_roots = [direct] if direct.exists() else [root]
        matches: list[Path] = []
        for search_root in search_roots:
            matches.extend(sorted(search_root.rglob("*.features.iac")))
            matches.extend(path for path in sorted(search_root.rglob("*features*.iac")) if path not in matches)
        teacher_matches = [path for path in matches if teacher in path.name or search_roots == [direct]]
        if teacher_matches:
            result[teacher] = teacher_matches
    return result


def _teacher_paths_from_arg(value: str) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for item in _parse_str_list(value):
        if "=" not in item:
            raise ValueError(f"teacher feature package entry must be teacher=path: {item}")
        teacher, paths = item.split("=", 1)
        parsed = [Path(path) for path in paths.split("|") if path]
        if parsed:
            result[teacher.strip()] = parsed
    return result


class FeatureL1Scorer:
    def __init__(self, payload: dict, *, teachers: list[str], teacher_paths: dict[str, list[Path]], l1_names: list[str]) -> None:
        from hcc_sempath.io.feature_cache import FeatureCacheReader

        self._reader_cls = FeatureCacheReader
        self._paths = teacher_paths
        self._path_index = {
            teacher: self._index_paths(paths)
            for teacher, paths in teacher_paths.items()
        }
        self._readers: dict[tuple[str, Path], object] = {}
        self._centers: dict[str, dict[str, np.ndarray]] = {}
        self._l1_names = [name for name in l1_names if name]
        self._build_centers(payload, teachers)

    @staticmethod
    def _index_paths(paths: list[Path]) -> dict[str, list[Path]]:
        result: dict[str, list[Path]] = {}
        for path in paths:
            for key in _package_keys(path):
                result.setdefault(key, []).append(path)
        return result

    def _reader(self, teacher: str, path: Path):
        key = (teacher, path)
        if key not in self._readers:
            self._readers[key] = self._reader_cls(path)
        return self._readers[key]

    def read(self, teacher: str, tile_id: str) -> np.ndarray | None:
        candidate_paths: list[Path] = []
        for key in _tile_keys(tile_id):
            candidate_paths.extend(self._path_index.get(teacher, {}).get(key, []))
        candidate_paths = list(dict.fromkeys([*candidate_paths, *self._paths.get(teacher, [])]))
        for path in candidate_paths:
            try:
                return self._reader(teacher, path).read_feature(tile_id)
            except FileNotFoundError:
                continue
        return None

    def _build_centers(self, payload: dict, teachers: list[str]) -> None:
        allowed = set(self._l1_names)
        for teacher in teachers:
            sums: dict[str, np.ndarray] = {}
            counts: dict[str, int] = {}
            for item in payload.get("annotations", {}).values():
                label = str(item.get("l1") or item.get("level1_label") or "")
                tile_id = str(item.get("tile_id") or "")
                if label not in allowed or not tile_id:
                    continue
                vector = self.read(teacher, tile_id)
                if vector is None:
                    continue
                vector = _normalize(vector)
                sums[label] = vector if label not in sums else sums[label] + vector
                counts[label] = counts.get(label, 0) + 1
            centers = {
                label: _normalize(total / max(1, counts[label]))
                for label, total in sums.items()
                if counts.get(label, 0) > 0
            }
            if centers:
                self._centers[teacher] = centers

    def suggest(self, item: dict) -> tuple[str, float, str]:
        current = str(item.get("l1") or item.get("level1_label") or "")
        tile_id = str(item.get("tile_id") or "")
        scores: dict[str, list[float]] = {}
        if not tile_id:
            return current, 0.0, ""
        for teacher, centers in self._centers.items():
            vector = self.read(teacher, tile_id)
            if vector is None:
                continue
            vector = _normalize(vector)
            for label, center in centers.items():
                scores.setdefault(label, []).append(float(np.dot(vector, center)))
        averaged = {
            label: sum(values) / len(values)
            for label, values in scores.items()
            if values and math.isfinite(sum(values) / len(values))
        }
        if not averaged:
            return current, 0.0, ""
        ranked = sorted(averaged.items(), key=lambda item: item[1], reverse=True)
        margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else ranked[0][1])
        uncertainty = max(0.0, 1.0 - margin)
        summary = "; ".join(f"{label}:{score:.3f}" for label, score in ranked[:3])
        return ranked[0][0], uncertainty, summary

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()


def _suggest_l1(item: dict, mode: str, binary_a: str, binary_b: str, scorer: FeatureL1Scorer | None) -> tuple[str, float, str]:
    current = str(item.get("l1") or item.get("level1_label") or "")
    if mode == "binary":
        return current, 0.0, ""
    if scorer is None:
        return current, float(item.get("l1_uncertainty", item.get("uncertainty", 0.0)) or 0.0), ""
    return scorer.suggest(item)


def build_candidates(
    payload: dict,
    *,
    mode: str = "l1",
    binary_a: str = "HCC-tumor",
    binary_b: str = "Inflammatory-stromal",
    scorer: FeatureL1Scorer | None = None,
    unstable_l1: set[str] | None = None,
) -> list[ReviewCandidate]:
    unstable_l1 = unstable_l1 or DEFAULT_UNSTABLE_L1
    candidates: list[ReviewCandidate] = []
    for key, item in payload.get("annotations", {}).items():
        if bool(item.get("reviewed")):
            continue
        current = str(item.get("l1") or item.get("level1_label") or "")
        if mode == "disagreement":
            candidates.append(
                ReviewCandidate(
                    key=str(key),
                    tile_id=str(item.get("review_id") or key),
                    current_l1=current,
                    suggested_l1="",
                    uncertainty=0.0,
                    score_summary="",
                )
            )
            continue
        if mode == "binary" and current not in {binary_a, binary_b}:
            continue
        suggested, uncertainty, score_summary = _suggest_l1(item, mode, binary_a, binary_b, scorer)
        if current in unstable_l1:
            uncertainty += 1.0
        if suggested != current:
            uncertainty += 0.5
        candidates.append(
            ReviewCandidate(
                key=str(key),
                tile_id=str(item.get("tile_id", "")),
                current_l1=current,
                suggested_l1=suggested,
                uncertainty=uncertainty,
                score_summary=score_summary,
            )
        )
    if mode == "disagreement":
        return candidates
    return sorted(candidates, key=lambda item: (-item.uncertainty, item.tile_id, item.key))


class ReviewState:
    def __init__(
        self,
        annotation_json: Path | None,
        output_json: Path | None,
        *,
        mode: str,
        binary_a: str,
        binary_b: str,
        teachers: list[str] | None = None,
        teacher_feature_root: Path | None = None,
        teacher_feature_packages: str = "",
        unstable_l1: set[str] | None = None,
        disagreement_csvs: list[Path] | None = None,
        blind_seed: int = 13,
    ) -> None:
        self.annotation_json = annotation_json
        if mode == "disagreement":
            if output_json is None:
                raise ValueError("--output-json is required for disagreement review")
            self.output_json = output_json
            if self.output_json.exists():
                self.payload = _load_payload(self.output_json)
            else:
                csvs = disagreement_csvs or []
                if not csvs:
                    raise ValueError("at least one --disagreement-csv is required")
                self.payload = _load_disagreement_payload(csvs, blind_seed)
                self.flush()
        else:
            if annotation_json is None:
                raise ValueError("--annotation-json is required for L1/binary review")
            self.output_json = output_json or annotation_json
            state_path = self.output_json if self.output_json.exists() else annotation_json
            self.payload = _load_payload(state_path)
        self.mode = mode
        self.binary_a = binary_a
        self.binary_b = binary_b
        self._viewers: dict[str, IacViewerData] = {}
        self.unstable_l1 = unstable_l1 or DEFAULT_UNSTABLE_L1
        self.scorer: FeatureL1Scorer | None = None
        self._candidate_cache: list[ReviewCandidate] | None = None
        teachers = teachers or []
        teacher_paths = _teacher_paths_from_arg(teacher_feature_packages)
        if teacher_feature_root and teachers:
            teacher_paths.update(_discover_teacher_paths(teacher_feature_root, teachers))
        if mode == "l1" and teachers and teacher_paths:
            l1_names = list(self.payload.get("l1_prototypes", L1_PROTOTYPES))
            self.scorer = FeatureL1Scorer(self.payload, teachers=teachers, teacher_paths=teacher_paths, l1_names=l1_names)

    @property
    def csv_path(self) -> Path:
        return self.output_json.with_suffix(".review.csv")

    def candidates(self) -> list[ReviewCandidate]:
        if self._candidate_cache is None:
            self._candidate_cache = build_candidates(
                self.payload,
                mode=self.mode,
                binary_a=self.binary_a,
                binary_b=self.binary_b,
                scorer=self.scorer,
                unstable_l1=self.unstable_l1,
            )
        return list(self._candidate_cache)

    def candidate(self, key: str) -> ReviewCandidate:
        candidates = {candidate.key: candidate for candidate in self.candidates()}
        return candidates[key]

    def candidate_payload(self) -> dict:
        candidates = self.candidates()
        queue_type = "standard"
        if self.mode == "disagreement":
            source_csvs = self.payload.get("source_csvs", [])
            has_random = any("random" in s.lower() for s in source_csvs)
            has_disagreement = any("top500" in s.lower() or "disagreement" in s.lower() for s in source_csvs)
            if has_random and has_disagreement:
                queue_type = "mixed"
            elif has_random:
                queue_type = "random"
            else:
                queue_type = "disagreement"
        return {
            "remaining": len(candidates),
            "candidates": [candidate.__dict__ for candidate in candidates],
            "l1_prototypes": self.payload.get("l1_prototypes", L1_PROTOTYPES),
            "mode": self.mode,
            "binary_a": self.binary_a,
            "binary_b": self.binary_b,
            "queue_type": queue_type,
        }

    def item(self, key: str) -> dict:
        return self.payload["annotations"][key]

    def public_item(self, key: str) -> dict:
        item = self.item(key)
        if self.mode == "disagreement":
            return {
                "review_id": item.get("review_id", key),
                "reviewed": bool(item.get("reviewed")),
                "l1": item.get("l1", ""),
                "source_group": item.get("source_group", ""),
            }
        return item

    def viewer(self, iac_path: str) -> IacViewerData:
        if iac_path not in self._viewers:
            self._viewers[iac_path] = IacViewerData(iac_path)
        return self._viewers[iac_path]

    def tile_png(self, key: str) -> bytes:
        item = self.item(key)
        viewer = self.viewer(str(item["iac_path"]))
        row = int(item["row"])
        return viewer.read_tile_png(row)

    def review(self, key: str, decision: str, new_l1: str = "") -> dict:
        item = self.item(key)
        if bool(item.get("reviewed")):
            return item
        current = str(item.get("l1") or "")
        suggested = _suggest_l1(item, self.mode, self.binary_a, self.binary_b, self.scorer)[0]
        if self.mode == "disagreement" and decision == "uncertain":
            item["l1"] = ""
            item["adjudication_status"] = "uncertain"
        elif self.mode == "disagreement" and decision == "adjust":
            if new_l1 not in self.payload.get("l1_prototypes", L1_PROTOTYPES):
                raise ValueError("disagreement adjudication requires a valid L1 label")
            item["l1"] = new_l1
            item["adjudication_status"] = "adjudicated"
        elif decision == "accept":
            item["l1"] = suggested
        elif decision == "adjust":
            if not new_l1:
                raise ValueError("adjust requires new_l1")
            item["l1"] = new_l1
        elif decision != "reject":
            raise ValueError(f"unknown review decision: {decision}")
        item["reviewed"] = True
        item["review_decision"] = decision
        item["review_previous_l1"] = current
        item["review_suggested_l1"] = suggested
        if self._candidate_cache is not None:
            self._candidate_cache = [candidate for candidate in self._candidate_cache if candidate.key != key]
        self.flush()
        return item

    def flush(self) -> None:
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=self.output_json.parent) as handle:
            json.dump(self.payload, handle, indent=2, sort_keys=True)
            tmp_path = Path(handle.name)
        tmp_path.replace(self.output_json)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            base_fields = [
                "review_id",
                "tile_id",
                "split",
                "rank",
                "l1",
                "adjudication_status",
                "reviewed",
                "review_decision",
                "review_previous_l1",
                "review_suggested_l1",
            ]
            extra_fields = sorted(
                {
                    field
                    for item in self.payload.get("annotations", {}).values()
                    for field in item
                    if field not in base_fields and field not in {"iac_path", "row"}
                }
            )
            writer = csv.DictWriter(
                handle,
                fieldnames=base_fields + extra_fields,
            )
            writer.writeheader()
            for item in self.payload.get("annotations", {}).values():
                writer.writerow({field: item.get(field, "") for field in writer.fieldnames})

    def close(self) -> None:
        for viewer in self._viewers.values():
            viewer.close()
        if self.scorer is not None:
            self.scorer.close()


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Annotation Review</title>
<style>
:root{color-scheme:light;--bg:#f4f6f8;--panel:#fff;--line:#d7dce2;--text:#1f2933;--muted:#687381;--blue:#1a73e8;--blue-soft:#e8f0fe;--danger:#b42318}
*{box-sizing:border-box}body{margin:0;font:14px system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--text);height:100svh;overflow:hidden}
button,select{font:inherit}button{min-height:40px;border:1px solid var(--line);background:var(--panel);color:var(--text);cursor:pointer}button.primary{background:var(--blue);border-color:var(--blue);color:#fff}button.danger{color:var(--danger)}button.ghost{background:transparent}button:disabled{opacity:.55;cursor:default}
.layout{display:grid;grid-template-columns:320px minmax(0,1fr);height:100svh}.layout.queue-collapsed{grid-template-columns:0 minmax(0,1fr)}
.queue{background:var(--panel);border-right:1px solid var(--line);overflow:hidden;display:flex;flex-direction:column}.queueHead{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:12px;border-bottom:1px solid var(--line)}.queueTitle{font-weight:650}.queueBody{overflow:auto}.layout.queue-collapsed .queue{border-right:0}.layout.queue-collapsed .queue>*{display:none}
.workspace{min-width:0;display:grid;grid-template-rows:auto minmax(0,1fr) auto;height:100svh}.topbar{display:flex;align-items:center;gap:10px;padding:10px 12px;border-bottom:1px solid var(--line);background:var(--panel)}.topbarTitle{min-width:0;font-weight:650;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.muted{color:var(--muted);font-size:12px}
.viewer{min-height:0;overflow:auto;display:flex;align-items:center;justify-content:center;padding:14px}.tileFrame{width:min(760px,100%);display:flex;align-items:center;justify-content:center}.tile{display:block;max-width:100%;max-height:calc(100svh - 250px);background:#fff;border:1px solid #b9c0c8;object-fit:contain}
.controls{background:var(--panel);border-top:1px solid var(--line);padding:12px}.controlGrid{display:grid;grid-template-columns:minmax(0,1fr) minmax(220px,300px);gap:12px;align-items:end}.labels{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}.chip{border:1px solid var(--line);border-radius:999px;padding:5px 8px;background:#fff}.chip.suggested{background:var(--blue-soft);border-color:#b8cdf8;color:#174ea6}.scores{margin-top:8px;color:var(--muted);font-size:12px;line-height:1.35}
.decisionPanel{display:grid;grid-template-columns:1fr;gap:8px}.pairButtons{display:grid;grid-template-columns:1fr 1fr;gap:8px}.choiceRow{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}.selectRow{display:grid;grid-template-columns:1fr;gap:4px}.selectLabel{font-size:12px;color:var(--muted)}select{min-height:40px;border:1px solid var(--line);background:#fff;padding:0 10px;width:100%}.status{min-height:18px;color:var(--muted);font-size:12px;white-space:pre-wrap}
.row{padding:10px 12px;border-bottom:1px solid #edf0f3;cursor:pointer}.row.active{background:var(--blue-soft)}.rowTitle{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.rowMeta{margin-top:3px;color:var(--muted);font-size:12px;line-height:1.25}.empty{padding:18px;color:var(--muted)}.hidden{display:none!important}
@media(max-width:760px){
  body{overflow:auto}.layout,.layout.queue-collapsed{display:block;height:auto;min-height:100svh}.queue{position:fixed;inset:0 0 auto 0;z-index:4;max-height:45svh;border-right:0;border-bottom:1px solid var(--line);box-shadow:0 8px 24px rgba(15,23,42,.16)}.layout.queue-collapsed .queue{display:none}.layout.queue-collapsed .queue>*{display:none}
  .workspace{height:100svh;grid-template-rows:auto minmax(0,1fr) auto}.topbar{position:sticky;top:0;z-index:3}.viewer{align-items:center;padding:8px}.tile{max-height:calc(100svh - 310px)}
  .controls{position:sticky;bottom:0;z-index:2;padding:10px}.controlGrid{grid-template-columns:1fr;gap:10px}.scores{max-height:42px;overflow:auto}.pairButtons{grid-template-columns:1fr 1fr}.choiceRow{grid-template-columns:1fr}
}
</style></head><body><div id="layout" class="layout">
<aside id="queuePanel" class="queue"><div class="queueHead"><div><div class="queueTitle">Review queue</div><div id="count" class="muted"></div></div><button id="hideQueue" class="ghost" type="button">Hide</button></div><div id="list" class="queueBody"></div></aside>
<main class="workspace"><div class="topbar"><button id="toggleQueue" type="button">Tiles</button><div class="topbarTitle" id="title">No tile selected</div><div id="progress" class="muted"></div></div>
<section class="viewer"><div class="tileFrame"><img id="tile" class="tile" alt=""></div></section>
<section class="controls"><div class="controlGrid"><div><div class="labels"><span class="chip" id="currentChip">Current: -</span><span class="chip suggested" id="suggestedChip">Suggested: -</span></div><div id="scores" class="scores"></div></div>
<div class="decisionPanel"><div id="pairButtons" class="pairButtons hidden"></div><div id="l1Choices" class="choiceRow"><button id="acceptBtn" class="primary" type="button">Accept suggested</button><button id="rejectBtn" type="button">Keep current</button><button id="adjustBtn" type="button">Use selected</button></div><div class="selectRow"><div class="selectLabel">Selected L1</div><select id="newL1"></select></div><div id="status" class="status"></div></div></div></section></main></div>
<script>
let queue=[], current=null, labels=[], mode='l1', classA='', classB='';
const layout=document.getElementById('layout');
const els={
  count:document.getElementById('count'), list:document.getElementById('list'), title:document.getElementById('title'),
  progress:document.getElementById('progress'), tile:document.getElementById('tile'), currentChip:document.getElementById('currentChip'),
  suggestedChip:document.getElementById('suggestedChip'), scores:document.getElementById('scores'), select:document.getElementById('newL1'),
  status:document.getElementById('status'), pairButtons:document.getElementById('pairButtons'), l1Choices:document.getElementById('l1Choices')
};
let busy=false;
async function api(path, opts){const r=await fetch(path,opts); if(!r.ok) throw new Error(await r.text()); return r.headers.get('content-type')?.includes('json')?r.json():r.blob();}
function setQueueOpen(open){layout.classList.toggle('queue-collapsed',!open);}
function isMobile(){return window.matchMedia('(max-width:760px)').matches;}
function text(el,value){el.textContent=value;}
async function load(selectIndex=0){
  const p=await api('/api/candidates'); queue=p.candidates; labels=p.l1_prototypes; mode=p.mode; classA=p.binary_a; classB=p.binary_b;
  let modeDesc = p.mode;
  if(p.mode==='disagreement'){
    if (p.queue_type === 'mixed') {
      modeDesc = 'blinded mixed review (混合分歧/对照)';
      document.title = 'Mixed Queue - Annotation Review';
      const qtEl = document.querySelector('.queueTitle');
      if(qtEl) qtEl.textContent = 'Mixed Queue (混合评估队列)';
    } else {
      const isRandom = p.queue_type==='random';
      modeDesc = isRandom ? 'blinded random control (随机对照)' : 'blinded high disagreement (高混淆分歧)';
      document.title = isRandom ? 'Random Queue - Annotation Review' : 'Disagreement Queue - Annotation Review';
      const qtEl = document.querySelector('.queueTitle');
      if(qtEl) qtEl.textContent = isRandom ? 'Random Queue (随机对照)' : 'Disagreement Queue (高混淆分歧)';
    }
  }
  text(els.count,`${p.remaining} remaining - ${modeDesc}`); fillLabels(); renderList();
  if(queue.length){await show(queue[Math.max(0,Math.min(selectIndex,queue.length-1))].key);} else {complete();}
}
function fillLabels(){els.select.innerHTML=''; labels.forEach(label=>{const o=document.createElement('option'); o.value=label; o.textContent=label; els.select.appendChild(o);});}
function renderList(){
  els.list.innerHTML='';
  if(!queue.length){const d=document.createElement('div'); d.className='empty'; d.textContent='No remaining tiles.'; els.list.appendChild(d); return;}
  queue.forEach((candidate,index)=>{
    const row=document.createElement('div'); row.className='row'+(current&&current.key===candidate.key?' active':'');
    const title=document.createElement('div'); title.className='rowTitle'; title.textContent=candidate.tile_id;
    const meta=document.createElement('div'); meta.className='rowMeta';
    meta.textContent=mode==='disagreement' ? `${index+1}. blinded L1 adjudication` : `${index+1}. ${candidate.current_l1} -> ${candidate.suggested_l1} - ${candidate.uncertainty.toFixed(3)}`;
    row.append(title,meta); row.addEventListener('click',()=>{show(candidate.key); if(isMobile())setQueueOpen(false);}); els.list.appendChild(row);
  });
}
async function show(key){
  const data=await api('/api/item?key='+encodeURIComponent(key)); current=data.candidate;
  const index=queue.findIndex(candidate=>candidate.key===key);
  const groupLabel = data.item.source_group === 'top500' ? ' [高混淆分歧组]' : (data.item.source_group === 'random500' ? ' [随机对照组]' : '');
  text(els.title, (data.item.review_id||data.item.tile_id||current.tile_id) + groupLabel); text(els.progress,index>=0?`${index+1}/${queue.length}`:'');
  if(mode==='disagreement'){
    text(els.currentChip,'Blinded tile'); text(els.suggestedChip,'Select adjudicated L1');
    text(els.scores,'Teacher identities, votes, confidence, split, and disagreement rank are hidden during review.');
    els.select.value=labels[0]||'';
  }else{
    text(els.currentChip,`Current: ${current.current_l1}`); text(els.suggestedChip,`Suggested: ${current.suggested_l1}`);
    text(els.scores,current.score_summary||`Uncertainty: ${current.uncertainty.toFixed(3)}`);
    els.select.value=current.suggested_l1;
  }
  els.tile.src='/api/tile?key='+encodeURIComponent(key)+'&t='+Date.now(); els.tile.alt=data.item.review_id||data.item.tile_id||current.tile_id;
  renderDecision(); renderList();
}
function renderDecision(){
  els.pairButtons.innerHTML='';
  if(mode==='disagreement'){
    els.pairButtons.classList.add('hidden'); els.l1Choices.classList.remove('hidden');
    document.getElementById('acceptBtn').classList.add('hidden');
    document.getElementById('rejectBtn').classList.remove('hidden');
    document.getElementById('rejectBtn').textContent='Unable to determine';
    document.getElementById('adjustBtn').textContent='Save selected L1';
  }else if(mode==='binary'){
    els.pairButtons.classList.remove('hidden'); els.l1Choices.classList.add('hidden');
    [classA,classB].forEach(label=>{const b=document.createElement('button'); b.type='button'; b.className='primary'; b.textContent=label; b.addEventListener('click',()=>choose(label)); els.pairButtons.appendChild(b);});
  }else{
    els.pairButtons.classList.add('hidden'); els.l1Choices.classList.remove('hidden');
    document.getElementById('acceptBtn').classList.remove('hidden');
    document.getElementById('rejectBtn').textContent='Keep current';
    document.getElementById('adjustBtn').textContent='Use selected';
  }
  setBusy(busy);
}
function complete(){
  current=null; text(els.title,'Review complete'); text(els.progress,''); text(els.currentChip,'Current: -'); text(els.suggestedChip,'Suggested: -'); text(els.scores,'No remaining tiles.'); els.tile.removeAttribute('src'); els.list.innerHTML='<div class="empty">No remaining tiles.</div>';
}
async function choose(label){els.select.value=label; await act('adjust');}
function setBusy(value){
  busy=value;
  document.querySelectorAll('button,select').forEach(el=>{el.disabled=value;});
}
async function act(decision){
  if(!current||busy)return; const doneKey=current.key; const index=queue.findIndex(candidate=>candidate.key===doneKey); const newL1=els.select.value;
  setBusy(true);
  text(els.status,'Saving...');
  try{
    await api('/api/review',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({key:doneKey,decision,new_l1:newL1})});
    text(els.status,'Saved.'); await load(Math.max(0,index));
  }catch(e){
    text(els.status,e.message||String(e));
  }finally{
    setBusy(false);
  }
}
document.getElementById('toggleQueue').addEventListener('click',()=>setQueueOpen(layout.classList.contains('queue-collapsed')));
document.getElementById('hideQueue').addEventListener('click',()=>setQueueOpen(false));
document.getElementById('acceptBtn').addEventListener('click',()=>act('accept'));
document.getElementById('rejectBtn').addEventListener('click',()=>act(mode==='disagreement'?'uncertain':'reject'));
document.getElementById('adjustBtn').addEventListener('click',()=>act('adjust'));
if(isMobile())setQueueOpen(false);
load().catch(e=>text(els.status,e.message||String(e)));
</script></body></html>"""


def _json_response(handler: BaseHTTPRequestHandler, payload: dict | list) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(state: ReviewState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    body = HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/candidates":
                    _json_response(self, state.candidate_payload())
                    return
                if parsed.path == "/api/item":
                    key = qs["key"][0]
                    _json_response(self, {"candidate": state.candidate(key).__dict__, "item": state.public_item(key)})
                    return
                if parsed.path == "/api/tile":
                    body = state.tile_png(qs["key"][0])
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/api/review":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                item = state.review(str(payload["key"]), str(payload["decision"]), str(payload.get("new_l1") or ""))
                _json_response(self, {"ok": True, "item": item})
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a focused annotation review UI.")
    parser.add_argument("--annotation-json", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--mode", choices=["l1", "binary", "disagreement"], default="l1")
    parser.add_argument("--disagreement-csv", action="append", default=[])
    parser.add_argument("--blind-seed", type=int, default=13)
    parser.add_argument("--class-a", "--binary-a", dest="binary_a", default="HCC-tumor")
    parser.add_argument("--class-b", "--binary-b", dest="binary_b", default="Inflammatory-stromal")
    parser.add_argument("--teachers", default="")
    parser.add_argument("--teacher-feature-root", default="")
    parser.add_argument("--teacher-feature-packages", default="")
    parser.add_argument("--unstable-l1", default="Degenerative-material")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--plugin", nargs="?", const="scripts/visuals_plugin.py", default="")
    args = parser.parse_args()

    # Generic external plugin extension hook
    if args.plugin:
        plugin_path = Path(args.plugin)
        if plugin_path.exists():
            print(f"[UI] Loading plugin from {plugin_path}...", flush=True)
            import importlib.util
            spec = importlib.util.spec_from_file_location("plugin_mod", str(plugin_path))
            plugin_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_mod)
            global HTML, make_handler
            HTML, make_handler = plugin_mod.apply_patch(HTML, make_handler)
        else:
            print(f"[UI] Error: plugin file not found: {plugin_path}", flush=True)
            sys.exit(1)
    else:
        default_plugin = Path("scripts/visuals_plugin.py")
        if default_plugin.exists():
            print(f"[UI] Info: Optional visuals plugin detected at {default_plugin}. Run with '--plugin' to enable it.", flush=True)

    state = ReviewState(
        Path(args.annotation_json) if args.annotation_json else None,
        Path(args.output_json) if args.output_json else None,
        mode=args.mode,
        binary_a=args.binary_a,
        binary_b=args.binary_b,
        teachers=_parse_str_list(args.teachers),
        teacher_feature_root=Path(args.teacher_feature_root) if args.teacher_feature_root else None,
        teacher_feature_packages=args.teacher_feature_packages,
        unstable_l1=set(_parse_str_list(args.unstable_l1)),
        disagreement_csvs=[Path(path) for path in args.disagreement_csv],
        blind_seed=args.blind_seed,
    )
    port = _find_free_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), make_handler(state))
    url = f"http://{args.host}:{port}/"
    print(f"annotation_review_ui url={url} state={state.output_json}", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.close()
        server.server_close()


if __name__ == "__main__":
    main()
