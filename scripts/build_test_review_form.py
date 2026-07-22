#!/usr/bin/env python3
"""Generate a local dropdown-based review form for a BCE-Net audit CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REVIEW_COLUMNS = [
    "audit_id",
    "stage1_overall_gt_status",
    "stage1_omission_gt_status",
    "stage1_excess_gt_status",
    "stage1_metric_eligible",
    "stage1_confidence",
    "stage1_notes",
    "stage2_omission_model_status",
    "stage2_excess_model_status",
    "stage2_error_attribution",
    "stage2_confidence",
    "stage2_notes",
    "review_complete",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REVIEW_COLUMNS:
            raise ValueError(
                f"Unexpected review schema: {reader.fieldnames}; expected {REVIEW_COLUMNS}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Review CSV is empty")
    audit_ids = [row["audit_id"] for row in rows]
    if any(not value for value in audit_ids) or len(set(audit_ids)) != len(audit_ids):
        raise ValueError("audit_id values must be non-empty and unique")
    return rows


def render_review_form(rows: list[dict[str, str]]) -> str:
    serialized_rows = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    serialized_columns = json.dumps(REVIEW_COLUMNS, ensure_ascii=False)
    return rf"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BCE-Net test 100 review form</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, -apple-system, sans-serif; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #101419; color: #e8edf2; }}
header {{ position: sticky; top: 0; z-index: 3; padding: 12px 18px; background: #182029f5; border-bottom: 1px solid #354454; }}
.topline, .controls, .progress {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; }}
h1 {{ margin: 0 18px 0 0; font-size: 1.2rem; }}
button, select, textarea, .file-label {{ color: #eef5fb; background: #263341; border: 1px solid #496078; border-radius: 6px; padding: 8px 10px; font: inherit; }}
button, .file-label {{ cursor: pointer; }}
button:hover, .file-label:hover {{ background: #33465a; }}
button.primary {{ background: #1769aa; border-color: #3f9ce6; }}
button.stage-active {{ background: #315d3d; border-color: #59a36c; }}
input[type=file] {{ display: none; }}
.progress {{ margin-top: 8px; color: #b7c5d3; font-size: .9rem; }}
.bar {{ width: 160px; height: 8px; overflow: hidden; background: #0c0f12; border-radius: 99px; }}
.bar > span {{ display: block; height: 100%; background: #52a86a; }}
main {{ display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(310px, .7fr); gap: 18px; padding: 18px; }}
.panel, .form-panel {{ background: #1b232c; border: 1px solid #354454; border-radius: 10px; padding: 14px; }}
.panel h2, .form-panel h2 {{ margin: 0 0 10px; font-size: 1.05rem; }}
.panel img {{ display: block; width: 100%; height: auto; background: #080a0c; border-radius: 5px; }}
.source-links {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
a {{ color: #8ecbff; }}
.field {{ margin-bottom: 13px; }}
.field label {{ display: block; margin-bottom: 5px; color: #cbd7e2; font-weight: 650; }}
.field small {{ display: block; margin-top: 4px; color: #91a5b8; line-height: 1.35; }}
.field select, .field textarea {{ width: 100%; }}
.field textarea {{ min-height: 90px; resize: vertical; }}
.required-empty select {{ border-color: #ca775f; }}
.notice {{ padding: 10px 12px; margin-bottom: 12px; background: #31291d; border: 1px solid #735d35; border-radius: 7px; color: #f1d69e; }}
.saved {{ color: #78c78c; }}
.footer-actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }}
@media (max-width: 960px) {{ main {{ grid-template-columns: 1fr; }} header {{ position: static; }} }}
</style>
</head>
<body>
<header>
  <div class="topline">
    <h1>Spatial v2 test 100 검수</h1>
    <button id="stage1Button" class="stage-active">Stage 1 · GT만</button>
    <button id="stage2Button">Stage 2 · Prediction</button>
    <button id="prevButton">← 이전</button>
    <select id="samplePicker" aria-label="검수 ID"></select>
    <button id="nextButton">다음 →</button>
    <button id="nextIncompleteButton">다음 미완료</button>
  </div>
  <div class="progress">
    <span id="stage1Progress"></span><span class="bar"><span id="stage1Bar"></span></span>
    <span id="stage2Progress"></span><span class="bar"><span id="stage2Bar"></span></span>
    <span id="saveStatus" class="saved">브라우저에 자동 저장됨</span>
  </div>
</header>
<main>
  <section class="panel">
    <h2 id="imageTitle"></h2>
    <div id="stageNotice" class="notice"></div>
    <a id="panelLink"><img id="panelImage" alt="검수 패널"></a>
    <div id="sourceLinks" class="source-links"></div>
  </section>
  <section class="form-panel">
    <h2 id="formTitle"></h2>
    <div id="formFields"></div>
    <div class="footer-actions">
      <button id="markCompleteButton" class="primary">현재 항목 완료 표시</button>
      <button id="exportButton" class="primary">CSV 다운로드</button>
      <label class="file-label">기존 CSV 불러오기<input id="importInput" type="file" accept=".csv,text/csv"></label>
      <button id="resetButton">브라우저 임시저장 초기화</button>
    </div>
  </section>
</main>
<script>
const INITIAL_ROWS = {serialized_rows};
const COLUMNS = {serialized_columns};
const STORAGE_KEY = "bcenet-spatial-v2-full100-review-v1";
const OPTIONS = {{
  stage1_overall_gt_status: ["correct", "issue", "mixed", "uncertain"],
  stage1_omission_gt_status: ["correct", "missing_labels", "false_labels", "wrong_class", "boundary_issue", "mixed", "not_present", "uncertain"],
  stage1_excess_gt_status: ["correct", "missing_labels", "false_labels", "wrong_class", "boundary_issue", "mixed", "not_present", "uncertain"],
  stage1_metric_eligible: ["yes", "no", "uncertain"],
  stage1_confidence: ["high", "medium", "low"],
  stage2_omission_model_status: ["correct", "under_detection", "over_detection", "class_confusion", "boundary_fragmentation", "mixed", "not_present", "uncertain"],
  stage2_excess_model_status: ["correct", "under_detection", "over_detection", "class_confusion", "boundary_fragmentation", "mixed", "not_present", "uncertain"],
  stage2_error_attribution: ["mostly_label", "mostly_model", "both", "no_material_error", "uncertain"],
  stage2_confidence: ["high", "medium", "low"],
  review_complete: ["yes"]
}};
const LABELS = {{
  stage1_overall_gt_status: ["전체 GT 상태", "crop 전체 라벨: correct / issue / mixed / uncertain"],
  stage1_omission_gt_status: ["Omission GT 상태", "주황 라벨을 판정하고, 해당 클래스가 없으면 not_present"],
  stage1_excess_gt_status: ["Excess GT 상태", "자홍 라벨을 판정하고, 해당 클래스가 없으면 not_present"],
  stage1_metric_eligible: ["정량평가 사용 가능", "라벨이 신뢰 가능하면 yes, 명확한 라벨 오류가 수치에 영향을 주면 no"],
  stage1_confidence: ["Stage 1 확신도", "내 라벨 판정의 확신도"],
  stage1_notes: ["Stage 1 메모", "선택 사항. 위치와 판단 근거를 기록"],
  stage2_omission_model_status: ["Omission 모델 상태", "누락 검출 결과를 GT 및 영상과 비교"],
  stage2_excess_model_status: ["Excess 모델 상태", "초과 건물 검출 결과를 GT 및 영상과 비교"],
  stage2_error_attribution: ["오류 원인", "오차가 주로 label / model / both 중 어디서 왔는지"],
  stage2_confidence: ["Stage 2 확신도", "내 모델 오류 판정의 확신도"],
  stage2_notes: ["Stage 2 메모", "선택 사항. FP/FN 위치와 판단 근거를 기록"],
  review_complete: ["최종 검수 완료", "Stage 1과 Stage 2를 모두 확인했으면 yes"]
}};
const STAGE1_REQUIRED = ["stage1_overall_gt_status", "stage1_omission_gt_status", "stage1_excess_gt_status", "stage1_metric_eligible", "stage1_confidence"];
const STAGE2_REQUIRED = ["stage2_omission_model_status", "stage2_excess_model_status", "stage2_error_attribution", "stage2_confidence", "review_complete"];
let rows = loadRows();
let currentIndex = 0;
let currentStage = 1;

function clone(value) {{ return JSON.parse(JSON.stringify(value)); }}
function validRows(value) {{
  return Array.isArray(value) && value.length === INITIAL_ROWS.length &&
    value.every((row, index) => row.audit_id === INITIAL_ROWS[index].audit_id);
}}
function loadRows() {{
  try {{
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (validRows(saved)) return saved;
  }} catch (_) {{}}
  return clone(INITIAL_ROWS);
}}
function saveRows() {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(rows));
  const status = document.getElementById("saveStatus");
  status.textContent = "브라우저에 자동 저장됨";
}}
function complete(row, required) {{ return required.every(key => Boolean(row[key])); }}
function updateProgress() {{
  const s1 = rows.filter(row => complete(row, STAGE1_REQUIRED)).length;
  const s2 = rows.filter(row => complete(row, STAGE2_REQUIRED)).length;
  document.getElementById("stage1Progress").textContent = `Stage 1 ${{s1}}/${{rows.length}}`;
  document.getElementById("stage2Progress").textContent = `Stage 2 ${{s2}}/${{rows.length}}`;
  document.getElementById("stage1Bar").style.width = `${{100 * s1 / rows.length}}%`;
  document.getElementById("stage2Bar").style.width = `${{100 * s2 / rows.length}}%`;
}}
function makeSelect(key, row) {{
  const select = document.createElement("select");
  select.dataset.key = key;
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "— 선택 —";
  select.appendChild(empty);
  for (const value of OPTIONS[key]) {{
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }}
  select.value = row[key] || "";
  select.addEventListener("change", () => {{
    row[key] = select.value;
    select.parentElement.classList.toggle("required-empty", !select.value);
    saveRows(); updateProgress(); updatePicker();
  }});
  return select;
}}
function makeField(key, row) {{
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  const required = (currentStage === 1 ? STAGE1_REQUIRED : STAGE2_REQUIRED).includes(key);
  if (required && !row[key]) wrapper.classList.add("required-empty");
  const label = document.createElement("label");
  label.textContent = LABELS[key][0];
  wrapper.appendChild(label);
  if (key.endsWith("_notes")) {{
    const textarea = document.createElement("textarea");
    textarea.value = row[key] || "";
    textarea.placeholder = "예: 오른쪽 아래 작은 주황 polygon은 실제 건물로 보이며 경계는 타당함";
    textarea.addEventListener("input", () => {{ row[key] = textarea.value; saveRows(); }});
    wrapper.appendChild(textarea);
  }} else {{
    wrapper.appendChild(makeSelect(key, row));
  }}
  const help = document.createElement("small");
  help.textContent = LABELS[key][1];
  wrapper.appendChild(help);
  return wrapper;
}}
function updatePicker() {{
  const picker = document.getElementById("samplePicker");
  [...picker.options].forEach((option, index) => {{
    const row = rows[index];
    const s1 = complete(row, STAGE1_REQUIRED) ? "✓1" : "·1";
    const s2 = complete(row, STAGE2_REQUIRED) ? "✓2" : "·2";
    option.textContent = `${{row.audit_id}}  ${{s1}} ${{s2}}`;
  }});
  picker.value = String(currentIndex);
}}
function render() {{
  const row = rows[currentIndex];
  const auditId = row.audit_id;
  const stage1 = currentStage === 1;
  document.getElementById("stage1Button").classList.toggle("stage-active", stage1);
  document.getElementById("stage2Button").classList.toggle("stage-active", !stage1);
  document.getElementById("imageTitle").textContent = `${{auditId}} · ${{stage1 ? "GT-only" : "Prediction/error"}}`;
  document.getElementById("formTitle").textContent = `${{auditId}} · Stage ${{currentStage}} 판정`;
  const relative = stage1 ? `01_stage1_gt_only/${{auditId}}.png` : `02_stage2_predictions/${{auditId}}.png`;
  document.getElementById("panelImage").src = relative;
  document.getElementById("panelLink").href = relative;
  document.getElementById("stageNotice").textContent = stage1
    ? "현재는 라벨만 판정합니다. 모델 prediction은 표시하지 않습니다."
    : "GT 검수를 마친 뒤 모델 prediction과 error를 판정하는 단계입니다.";
  const links = document.getElementById("sourceLinks");
  links.innerHTML = stage1 ? [
    ["원본 영상", `source_crops/${{auditId}}/before.png`],
    ["footprint mask", `source_crops/${{auditId}}/reference.png`],
    ["raw GT state", `source_crops/${{auditId}}/state_gt_raw.png`],
    ["color GT state", `source_crops/${{auditId}}/state_gt_color.png`]
  ].map(([name, path]) => `<a href="${{path}}" target="_blank">${{name}}</a>`).join("") : "";
  const fields = stage1
    ? ["stage1_overall_gt_status", "stage1_omission_gt_status", "stage1_excess_gt_status", "stage1_metric_eligible", "stage1_confidence", "stage1_notes"]
    : ["stage2_omission_model_status", "stage2_excess_model_status", "stage2_error_attribution", "stage2_confidence", "stage2_notes", "review_complete"];
  const form = document.getElementById("formFields");
  form.innerHTML = "";
  fields.forEach(key => form.appendChild(makeField(key, row)));
  document.getElementById("markCompleteButton").style.display = stage1 ? "none" : "inline-block";
  document.getElementById("prevButton").disabled = currentIndex === 0;
  document.getElementById("nextButton").disabled = currentIndex === rows.length - 1;
  updatePicker(); updateProgress();
}}
function move(delta) {{ currentIndex = Math.max(0, Math.min(rows.length - 1, currentIndex + delta)); render(); window.scrollTo(0, 0); }}
function setStage(stage) {{
  if (stage === 2) {{
    const done = rows.filter(row => complete(row, STAGE1_REQUIRED)).length;
    if (done < rows.length && !confirm(`Stage 1이 ${{done}}/${{rows.length}}개 완료됐습니다. Prediction을 지금 공개할까요?`)) return;
  }}
  currentStage = stage; render();
}}
function nextIncomplete() {{
  const required = currentStage === 1 ? STAGE1_REQUIRED : STAGE2_REQUIRED;
  for (let offset = 1; offset <= rows.length; offset++) {{
    const candidate = (currentIndex + offset) % rows.length;
    if (!complete(rows[candidate], required)) {{ currentIndex = candidate; render(); window.scrollTo(0, 0); return; }}
  }}
  alert(`Stage ${{currentStage}} 항목이 모두 완료됐습니다.`);
}}
function csvEscape(value) {{
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${{text.replaceAll('"', '""')}}"` : text;
}}
function exportCsv() {{
  const lines = [COLUMNS.join(","), ...rows.map(row => COLUMNS.map(key => csvEscape(row[key])).join(","))];
  const blob = new Blob(["\ufeff" + lines.join("\r\n") + "\r\n"], {{type: "text/csv;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = "full100_review.csv"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}}
function parseCsv(text) {{
  const table = []; let row = []; let cell = ""; let quoted = false;
  for (let i = 0; i < text.length; i++) {{
    const char = text[i];
    if (quoted) {{
      if (char === '"' && text[i + 1] === '"') {{ cell += '"'; i++; }}
      else if (char === '"') quoted = false;
      else cell += char;
    }} else if (char === '"') quoted = true;
    else if (char === ',') {{ row.push(cell); cell = ""; }}
    else if (char === '\n') {{ row.push(cell.replace(/\r$/, "")); table.push(row); row = []; cell = ""; }}
    else cell += char;
  }}
  if (cell || row.length) {{ row.push(cell.replace(/\r$/, "")); table.push(row); }}
  return table.filter(values => values.some(value => value !== ""));
}}
function importCsv(file) {{
  const reader = new FileReader();
  reader.onload = () => {{
    try {{
      const table = parseCsv(String(reader.result).replace(/^\ufeff/, ""));
      if (JSON.stringify(table[0]) !== JSON.stringify(COLUMNS)) throw new Error("CSV 열 이름이 다릅니다.");
      const imported = table.slice(1).map(values => Object.fromEntries(COLUMNS.map((key, index) => [key, values[index] || ""])));
      if (!validRows(imported)) throw new Error("V001..V100 audit_id 또는 행 수가 다릅니다.");
      rows = imported; saveRows(); render(); alert("CSV를 불러왔습니다.");
    }} catch (error) {{ alert(`불러오기 실패: ${{error.message}}`); }}
  }};
  reader.readAsText(file, "utf-8");
}}

const picker = document.getElementById("samplePicker");
rows.forEach((row, index) => {{ const option = document.createElement("option"); option.value = String(index); picker.appendChild(option); }});
picker.addEventListener("change", () => {{ currentIndex = Number(picker.value); render(); }});
document.getElementById("stage1Button").addEventListener("click", () => setStage(1));
document.getElementById("stage2Button").addEventListener("click", () => setStage(2));
document.getElementById("prevButton").addEventListener("click", () => move(-1));
document.getElementById("nextButton").addEventListener("click", () => move(1));
document.getElementById("nextIncompleteButton").addEventListener("click", nextIncomplete);
document.getElementById("markCompleteButton").addEventListener("click", () => {{ rows[currentIndex].review_complete = "yes"; saveRows(); render(); }});
document.getElementById("exportButton").addEventListener("click", exportCsv);
document.getElementById("importInput").addEventListener("change", event => {{ if (event.target.files[0]) importCsv(event.target.files[0]); event.target.value = ""; }});
document.getElementById("resetButton").addEventListener("click", () => {{
  if (confirm("브라우저에 저장된 검수 내용을 초기 CSV 상태로 되돌릴까요? 먼저 CSV를 다운로드했는지 확인하세요.")) {{ rows = clone(INITIAL_ROWS); saveRows(); render(); }}
}});
document.addEventListener("keydown", event => {{
  if (event.target.matches("textarea, select")) return;
  if (event.key === "ArrowLeft") move(-1);
  if (event.key === "ArrowRight") move(1);
}});
render();
</script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    review_csv = Path(args.review_csv).resolve()
    output = Path(args.output).resolve()
    if not review_csv.is_file():
        raise FileNotFoundError(review_csv)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    rows = read_review_rows(review_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_review_form(rows), encoding="utf-8")
    print(f"Wrote {output} with {len(rows)} review rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
