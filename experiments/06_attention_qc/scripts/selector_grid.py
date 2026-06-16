#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path("/Volumes/Macintosh USB/Dev/2026-CT-WSI/hcc-sempath")
sys.path.insert(0, str(REPO / "src"))

from hcc_sempath.io.tile_package import TilePackageReader
from hcc_sempath.modeling.models import load_hcc_sempath_release

WORK_DIR = Path("/tmp/hcc_sempath_exval_selector")
CACHE_DIR = WORK_DIR / "cache"
EXVAL_CASES_JSON = REPO / "experiments/06_attention_qc/configs/exval_selector_selected_cases.json"
FINAL_CHOICES_JSON = REPO / "experiments/06_attention_qc/configs/exval_selector_final_choices.json"
REVIEW_PATH = REPO / "annotations/reviews/teacher_disagreement/exval_1000/review.csv"
PREDICTION_PATH = REPO / "artifacts/caches/local_cache/teacher_disagreement/teacher_disagreement_model_predictions.csv"
L2_PATH = REPO / "artifacts/caches/local_cache/teacher_disagreement/teacher_disagreement_l2_probabilities.npz"
THRESHOLD_PATH = REPO / "artifacts/caches/local_cache/train_l2_thresholds/thresholds.json"
CONFIG_PATH = REPO / "artifacts/release/config.json"
MODEL_PATH = REPO / "artifacts/release/hcc_sempath_release.pt"

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HCC-SemPath Exhibit Case Selector</title>
<style>
:root {
  --ink: #1e293b;
  --muted: #64748b;
  --line: #cbd5e1;
  --panel: #f8fafc;
  --primary: #dc2626;
  --primary-bg: #fee2e2;
  --backup: #d97706;
  --backup-bg: #fef3c7;
  --active-row: #f1f5f9;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--ink);
  background: #0f172a;
}
header {
  height: 64px;
  padding: 0 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #1e293b;
  border-bottom: 1px solid #334155;
  position: sticky;
  top: 0;
  z-index: 100;
  color: #f1f5f9;
}
header h1 {
  font-size: 20px;
  margin: 0;
  font-weight: 700;
  background: linear-gradient(to right, #f8fafc, #94a3b8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.actions {
  display: flex;
  align-items: center;
  gap: 16px;
}
.status {
  font-size: 13px;
  color: #94a3b8;
}
button {
  height: 38px;
  border: 1px solid #475569;
  background: #334155;
  color: #f1f5f9;
  padding: 0 16px;
  font-weight: 600;
  cursor: pointer;
  border-radius: 6px;
  transition: all 0.15s ease;
}
button:hover {
  background: #475569;
  border-color: #64748b;
}
button.submit-btn {
  background: #2563eb;
  border-color: #2563eb;
}
button.submit-btn:hover {
  background: #3b82f6;
  border-color: #3b82f6;
  box-shadow: 0 0 12px rgba(59, 130, 246, 0.4);
}
.container {
  padding: 24px;
  max-width: 100%;
}
.group-section {
  background: #1e293b;
  border-radius: 12px;
  margin-bottom: 32px;
  border: 1px solid #334155;
  overflow: hidden;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
.group-header {
  padding: 16px 24px;
  background: #0f172a;
  border-bottom: 1px solid #334155;
  font-size: 16px;
  font-weight: 700;
  color: #e2e8f0;
  display: flex;
  align-items: center;
  gap: 12px;
}
.group-header .badge {
  background: #334155;
  color: #94a3b8;
  padding: 2px 8px;
  border-radius: 9999px;
  font-size: 12px;
}
.table-container {
  overflow-x: auto;
  width: 100%;
}
table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
}
th {
  background: #1e293b;
  padding: 12px 10px;
  font-size: 11px;
  text-transform: uppercase;
  color: #94a3b8;
  font-weight: 600;
  letter-spacing: 0.05em;
  border-bottom: 1px solid #334155;
  white-space: nowrap;
}
td {
  padding: 8px 10px;
  border-bottom: 1px solid #334155;
  background: #1e293b;
  vertical-align: middle;
  color: #cbd5e1;
}
tr:hover td {
  background: #243048;
}
tr.row-active td {
  background: #2b3954;
}
.case-control {
  min-width: 130px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.case-info {
  font-size: 13px;
  font-weight: 700;
  color: #f1f5f9;
}
.case-subinfo {
  font-size: 11px;
  color: #64748b;
  word-break: break-all;
}
.radio-group {
  display: flex;
  gap: 6px;
}
.radio-btn {
  flex: 1;
  height: 26px;
  font-size: 11px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #475569;
  border-radius: 4px;
  cursor: pointer;
  user-select: none;
  font-weight: 600;
  transition: all 0.1s ease;
  color: #94a3b8;
  background: #334155;
}
.radio-btn:hover {
  background: #475569;
}
.radio-btn[data-role="primary"].active {
  background: var(--primary);
  border-color: var(--primary);
  color: #fff;
}
.radio-btn[data-role="backup"].active {
  background: var(--backup);
  border-color: var(--backup);
  color: #fff;
}
.img-cell {
  position: relative;
  width: 90px;
  height: 90px;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid #475569;
  background: #0f172a;
}
.img-cell img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.img-cell .meta-tag {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(15, 23, 42, 0.85);
  font-size: 9px;
  padding: 1px 3px;
  text-align: center;
  color: #94a3b8;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.img-cell.inactive img {
  filter: grayscale(90%) opacity(25%);
}
.img-cell.inactive:hover img {
  filter: none;
}
.img-cell:hover {
  border-color: #3b82f6;
  box-shadow: 0 0 6px rgba(59, 130, 246, 0.5);
  cursor: crosshair;
}
/* Hover box for zooming */
#zoom-preview {
  position: fixed;
  display: none;
  width: 250px;
  height: 290px;
  background: #1e293b;
  border: 2px solid #3b82f6;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);
  z-index: 1000;
  pointer-events: none;
}
#zoom-preview img {
  width: 250px;
  height: 250px;
  display: block;
  background: #fff;
}
#zoom-preview .label {
  padding: 6px 10px;
  font-size: 11px;
  color: #cbd5e1;
  background: #0f172a;
  border-top: 1px solid #334155;
  height: 40px;
}
#zoom-preview .label strong {
  display: block;
  color: #f1f5f9;
}
</style>
</head>
<body>
<header>
  <h1>HCC-SemPath Case Selection Dashboard</h1>
  <div class="actions">
    <span class="status" id="status">Loading data...</span>
    <button class="submit-btn" id="submit">Submit Decisions</button>
  </div>
</header>
<div class="container" id="groups-container">
  <!-- Dynamic content -->
</div>
<div id="zoom-preview">
  <img id="zoom-img" src="">
  <div class="label" id="zoom-label"></div>
</div>

<script>
let cases = [];
let l1_names = [];
let l2_names = [];
let thresholds = [];
let choices = {};

const $ = id => document.getElementById(id);
const preview = $('zoom-preview');
const previewImg = $('zoom-img');
const previewLabel = $('zoom-label');

async function init() {
  try {
    const r = await fetch('/api/cases');
    const data = await r.json();
    cases = data.cases;
    l1_names = data.l1_names;
    l2_names = data.l2_names;
    thresholds = data.thresholds;
    choices = data.choices || {
      "HCC-tumor": {primary: "", backup: ""},
      "Background-liver": {primary: "", backup: ""},
      "Inflammatory-stromal": {primary: "", backup: ""}
    };
    $('status').textContent = `${cases.length} cases loaded`;
    render();
  } catch (e) {
    $('status').textContent = 'Error loading cases.';
  }
}

function render() {
  const container = $('groups-container');
  container.innerHTML = '';

  // Group cases by predicted L1
  const groups = {};
  l1_names.forEach(name => groups[name] = []);
  cases.forEach(c => {
    if (groups[c.pred_l1]) {
      groups[c.pred_l1].push(c);
    } else {
      groups[c.pred_l1] = [c];
    }
  });

  // Render each L1 group
  l1_names.forEach(gName => {
    const list = groups[gName] || [];
    if (list.length === 0) return;

    const section = document.createElement('div');
    section.className = 'group-section';

    const header = document.createElement('div');
    header.className = 'group-header';
    header.innerHTML = `<span>${gName}</span><span class="badge">${list.length} cases</span>`;
    section.appendChild(header);

    const tableContainer = document.createElement('div');
    tableContainer.className = 'table-container';

    // Generate table headers (L1 + 10 L2s)
    const l2_shorts = {
      'hepatocellular-parenchyma-present': 'Hep',
      'necrosis-present': 'Nec',
      'hemorrhage-present': 'Hem',
      'bile-pigment-present': 'Bile',
      'inflammatory-cell-present': 'Infl',
      'fibrous-stroma-present': 'Stroma',
      'steatosis-vacuolation-present': 'Stea',
      'hyaline-change-present': 'Hyal',
      'vascular-structure-present': 'Vasc',
      'ductular-portal-present': 'Duct'
    };
    let headersHtml = `<th>Case Details</th><th>L1 Map</th>`;
    l2_names.forEach((l2, idx) => {
      const short = l2_shorts[l2] || l2.split('-').slice(0, 2).join(' ');
      headersHtml += `<th title="${l2}">${short}</th>`;
    });

    let rowsHtml = '';
    list.forEach(c => {
      const rid = c.review_id;
      const isPrimary = choices[gName]?.primary === rid;
      const backupVal = choices[gName]?.backup;
      const isBackup = Array.isArray(backupVal) ? backupVal.includes(rid) : (backupVal === rid);

      let rowHtml = `<tr id="row-${rid}" class="${(isPrimary || isBackup) ? 'row-active' : ''}">`;
      
      // Case info column
      rowHtml += `<td>
        <div class="case-control">
          <div class="case-info">${rid}</div>
          <div class="case-subinfo">${c.source_group}<br>${c.tile_id.substring(0, 18)}...</div>
          <div class="radio-group">
            <div class="radio-btn ${isPrimary ? 'active' : ''}" data-role="primary" onclick="setRole('${gName}', '${rid}', 'primary')">Primary</div>
            <div class="radio-btn ${isBackup ? 'active' : ''}" data-role="backup" onclick="setRole('${gName}', '${rid}', 'backup')">Backup</div>
          </div>
        </div>
      </td>`;

      // Column 1: L1 Map
      const l1_url = `/cache/${rid}/l1.png`;
      rowHtml += `<td>
        <div class="img-cell" onmousemove="showZoom(event, '${l1_url}', 'L1 Map: ${c.pred_l1}', 'Consensus: ${c.pred_l1}')" onmouseleave="hideZoom()">
          <img src="${l1_url}" loading="lazy">
          <div class="meta-tag">L1 Map</div>
        </div>
      </td>`;

      // Columns 2-11: 10 L2 Maps
      c.l2_probabilities.forEach((prob, idx) => {
        const threshold = thresholds[idx];
        const isActive = prob >= threshold;
        const l2_url = `/cache/${rid}/l2_${idx}.png`;
        const l2_name = l2_names[idx];

        rowHtml += `<td>
          <div class="img-cell ${isActive ? '' : 'inactive'}" onmousemove="showZoom(event, '${l2_url}', 'L2: ${l2_name}', 'p=${prob.toFixed(3)} (threshold=${threshold.toFixed(3)})')" onmouseleave="hideZoom()">
            <img src="${l2_url}" loading="lazy">
            <div class="meta-tag">${prob.toFixed(2)}</div>
          </div>
        </td>`;
      });

      rowHtml += '</tr>';
      rowsHtml += rowHtml;
    });

    tableContainer.innerHTML = `<table><thead><tr>${headersHtml}</tr></thead><tbody>${rowsHtml}</tbody></table>`;
    section.appendChild(tableContainer);
    container.appendChild(section);
  });
}

function setRole(group, rid, role) {
  if (!choices[group]) {
    choices[group] = {primary: "", backup: []};
  }
  if (!choices[group].backup) {
    choices[group].backup = [];
  }
  if (!Array.isArray(choices[group].backup)) {
    choices[group].backup = choices[group].backup ? [choices[group].backup] : [];
  }
  
  if (role === "primary") {
    const currentVal = choices[group].primary;
    if (currentVal === rid) {
      choices[group].primary = "";
    } else {
      choices[group].primary = rid;
      // Remove from backup
      choices[group].backup = choices[group].backup.filter(x => x !== rid);
    }
  } else if (role === "backup") {
    if (choices[group].backup.includes(rid)) {
      choices[group].backup = choices[group].backup.filter(x => x !== rid);
    } else {
      choices[group].backup.push(rid);
      // Remove from primary
      if (choices[group].primary === rid) {
        choices[group].primary = "";
      }
    }
  }

  // Refresh visual state using lightweight DOM sync to prevent complete page rebuild
  syncChoicesUI();
}

function syncChoicesUI() {
  cases.forEach(c => {
    const rid = c.review_id;
    const gName = c.pred_l1;
    const isPrimary = choices[gName]?.primary === rid;
    const backupVal = choices[gName]?.backup || [];
    const isBackup = Array.isArray(backupVal) ? backupVal.includes(rid) : (backupVal === rid);

    const row = document.getElementById(`row-${rid}`);
    if (row) {
      if (isPrimary || isBackup) {
        row.classList.add('row-active');
      } else {
        row.classList.remove('row-active');
      }

      const btnPrimary = row.querySelector('.radio-btn[data-role="primary"]');
      if (btnPrimary) {
        if (isPrimary) btnPrimary.classList.add('active');
        else btnPrimary.classList.remove('active');
      }

      const btnBackup = row.querySelector('.radio-btn[data-role="backup"]');
      if (btnBackup) {
        if (isBackup) btnBackup.classList.add('active');
        else btnBackup.classList.remove('active');
      }
    }
  });
}

function showZoom(e, url, title, desc) {
  preview.style.display = 'block';
  previewImg.src = url;
  previewLabel.innerHTML = `<strong>${title}</strong><span>${desc}</span>`;
  
  // Align preview box next to cursor
  let x = e.clientX + 15;
  let y = e.clientY - 130;
  
  // Boundaries check
  if (x + 250 > window.innerWidth) x = e.clientX - 265;
  if (y < 0) y = 10;
  if (y + 290 > window.innerHeight) y = window.innerHeight - 300;
  
  preview.style.left = x + 'px';
  preview.style.top = y + 'px';
}

function hideZoom() {
  preview.style.display = 'none';
}

$('submit').onclick = async () => {
  $('status').textContent = 'Submitting decisions...';
  try {
    const r = await fetch('/api/submit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ choices })
    });
    const res = await r.json();
    if (r.ok) {
      $('status').textContent = `Saved to ${res.path}`;
    } else {
      $('status').textContent = 'Failed to submit selection.';
    }
  } catch (e) {
    $('status').textContent = 'Submit error occurred.';
  }
};

init();
</script>
</body>
</html>"""


class SelectorState:
    def __init__(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.device = self._choose_device()
        self.model, self.config = load_hcc_sempath_release(CONFIG_PATH, MODEL_PATH, self.device)
        self.model.eval()
        self.mean = torch.tensor(self.config["preprocessing"]["mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.config["preprocessing"]["std"], device=self.device).view(1, 3, 1, 1)
        self.l1_names = list(self.config["l1_names"])
        self.l2_names = list(self.config["l2_names"])
        threshold_payload = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
        self.thresholds = np.asarray(threshold_payload["thresholds"], dtype=np.float32)
        self.cases = self._load_cases()
        self.by_id = {row["review_id"]: row for row in self.cases}
        self.lock = threading.Lock()

    def _choose_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_cases(self) -> list[dict]:
        with EXVAL_CASES_JSON.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        with PREDICTION_PATH.open(newline="", encoding="utf-8") as handle:
            pred_by_id = {row["review_id"]: row for row in csv.DictReader(handle)}

        l2 = np.load(L2_PATH, allow_pickle=True)
        l2_ids = [str(x) for x in l2["review_ids"].tolist()]
        l2_by_id = {rid: l2["pred_full"][idx] for idx, rid in enumerate(l2_ids)}

        rows = []
        for c in payload["cases"]:
            rid = c["review_id"]
            probs = np.asarray(l2_by_id[rid], dtype=np.float32)
            rows.append({
                "review_id": rid,
                "tile_id": c["tile_id"],
                "package_path": c["package_path"],
                "row_idx": int(c["row_idx"]),
                "source_group": c["source_group"],
                "pred_l1": pred_by_id[rid]["pred_full"],
                "l2_probabilities": probs.tolist(),
                "l2_positive_count": int((probs >= self.thresholds).sum()),
            })
        return rows

    def load_choices(self) -> dict:
        choices = {
            "HCC-tumor": {"primary": "", "backup": []},
            "Background-liver": {"primary": "", "backup": []},
            "Inflammatory-stromal": {"primary": "", "backup": []}
        }
        if FINAL_CHOICES_JSON.exists():
            try:
                loaded = json.loads(FINAL_CHOICES_JSON.read_text(encoding="utf-8"))
                for g, vals in loaded.items():
                    if g in choices:
                        choices[g]["primary"] = vals.get("primary", "")
                        backup_val = vals.get("backup", [])
                        choices[g]["backup"] = backup_val if isinstance(backup_val, list) else ([backup_val] if backup_val else [])
            except Exception:
                pass
        return choices

    def save_choices(self, choices: dict) -> None:
        FINAL_CHOICES_JSON.write_text(json.dumps(choices, ensure_ascii=False, indent=2), encoding="utf-8")
        # Sync to /tmp directory for safety
        tmp_choices_path = WORK_DIR / "exval_selector_final_choices.json"
        tmp_choices_path.write_text(json.dumps(choices, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_image(self, case: dict) -> Image.Image:
        reader = TilePackageReader(case["package_path"])
        try:
            return reader.read_image_at(case["row_idx"]).convert("RGB").resize((224, 224))
        finally:
            reader.close()

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
        return (tensor - self.mean) / self.std

    @staticmethod
    def _normalize_map(values: torch.Tensor) -> np.ndarray:
        values = values.float().clamp_min(0)
        values = values / values.max().clamp_min(1e-6)
        values = F.interpolate(values.view(1, 1, 14, 14), size=(224, 224), mode="bilinear", align_corners=False)
        return values[0, 0].cpu().numpy()

    @staticmethod
    def _overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
        base = np.asarray(image, dtype=np.float32) / 255.0
        x = np.clip(heatmap, 0, 1)
        color = np.stack([
            np.clip(1.7 * x, 0, 1),
            np.clip(1.7 * x - 0.55, 0, 1),
            np.clip(1.6 * x - 1.0, 0, 1),
        ], axis=-1)
        alpha = (0.10 + 0.48 * x)[..., None]
        out = np.clip(base * (1 - alpha) + color * alpha, 0, 1)
        return Image.fromarray((out * 255).astype(np.uint8))

    def render(self, review_id: str) -> dict:
        case = self.by_id[review_id]
        case_dir = CACHE_DIR / review_id
        payload_path = case_dir / "payload.json"
        all_l2_exist = all((case_dir / f"l2_{idx}.png").exists() for idx in range(10))
        if payload_path.exists() and all_l2_exist:
            return json.loads(payload_path.read_text(encoding="utf-8"))
        with self.lock:
            all_l2_exist = all((case_dir / f"l2_{idx}.png").exists() for idx in range(10))
            if payload_path.exists() and all_l2_exist:
                return json.loads(payload_path.read_text(encoding="utf-8"))
            case_dir.mkdir(parents=True, exist_ok=True)
            image = self._read_image(case)
            image.save(case_dir / "original.png")
            x = self._tensor(image)
            with torch.inference_mode():
                baseline = self.model(x)
            l1_probs = baseline["l1_probabilities"][0]
            l2_probs = baseline["l2_probabilities"][0]
            order = l1_probs.argsort(descending=True)
            l1_idx, runner_idx = int(order[0]), int(order[1])
            baseline_margin = l1_probs[l1_idx] - l1_probs[runner_idx]

            variants = []
            for row in range(14):
                for col in range(14):
                    variant = x[0].clone()
                    variant[:, row * 16:(row + 1) * 16, col * 16:(col + 1) * 16] = 0
                    variants.append(variant)
            l1_drop, l2_drops = [], {idx: [] for idx in range(10)}
            with torch.inference_mode():
                for start in range(0, len(variants), 28):
                    output = self.model(torch.stack(variants[start:start + 28]).to(self.device))
                    probs1 = output["l1_probabilities"]
                    l1_drop.extend((baseline_margin - (probs1[:, l1_idx] - probs1[:, runner_idx])).cpu())
                    probs2 = output["l2_probabilities"]
                    for idx in range(10):
                        l2_drops[idx].extend((l2_probs[idx] - probs2[:, idx]).cpu())

            l1_map = self._normalize_map(torch.stack(l1_drop))
            self._overlay(image, l1_map).save(case_dir / "l1.png")
            l2_payload = []
            for idx in range(10):
                heatmap = self._normalize_map(torch.stack(l2_drops[idx]))
                filename = f"l2_{idx}.png"
                self._overlay(image, heatmap).save(case_dir / filename)
                l2_payload.append({
                    "name": self.l2_names[idx],
                    "probability": float(l2_probs[idx].cpu()),
                    "threshold": float(self.thresholds[idx]),
                    "map_url": f"/cache/{review_id}/{filename}",
                })
            payload = {
                "review_id": review_id,
                "tile_id": case["tile_id"],
                "source_group": case["source_group"],
                "original_url": f"/cache/{review_id}/original.png",
                "l1": {
                    "name": self.l1_names[l1_idx],
                    "probability": float(l1_probs[l1_idx].cpu()),
                    "margin": float(baseline_margin.cpu()),
                    "map_url": f"/cache/{review_id}/l1.png",
                },
                "l2": l2_payload,
            }
            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload


STATE: SelectorState


class Handler(BaseHTTPRequestHandler):
    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            data = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/api/cases":
            self._json({
                "cases": STATE.cases,
                "l1_names": STATE.l1_names,
                "l2_names": STATE.l2_names,
                "thresholds": STATE.thresholds.tolist(),
                "choices": STATE.load_choices()
            })
            return

        if parsed.path.startswith("/cache/"):
            relative = Path(parsed.path.removeprefix("/cache/"))
            parts = relative.parts
            if len(parts) >= 2:
                rid = parts[0]
                if rid in STATE.by_id:
                    path = (CACHE_DIR / relative).resolve()
                    if not path.exists():
                        print(f"On-demand rendering missing cache file for {rid}: {relative}", flush=True)
                        try:
                            STATE.render(rid)
                        except Exception as e:
                            print(f"Render failed for {rid}: {e}", flush=True)
                            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
                            return
            
            path = (CACHE_DIR / relative).resolve()
            if CACHE_DIR.resolve() not in path.parents or not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/submit":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        choices = request.get("choices", {})
        STATE.save_choices(choices)
        self._json({"choices": choices, "path": str(FINAL_CHOICES_JSON)})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[grid_selector] {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    global STATE
    STATE = SelectorState()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"grid_selector_ready url=http://{args.host}:{args.port} cases={len(STATE.cases)} "
        f"device={STATE.device} cache={CACHE_DIR}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
