#!/usr/bin/env python3
"""Build a portable full-test qualitative audit package from a frozen evaluation."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for value in (REPO_ROOT, REPO_ROOT / "DCNv2"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from dataset.bcenet_map_ortho import MapOrthoBCENetDataset
from evaluate_bcenet_map_ortho import (
    CROP_SIZE,
    EXPECTED_SAMPLES,
    PROTOCOL,
    THRESHOLD,
    build_model,
    infer_qualitative_record,
    prepare_output_dir,
    seed_everything,
    sha256,
)
from scripts.build_test_pilot_audit import (
    REVIEW_COLUMNS,
    read_csv,
    save_rgb,
    stage1_row,
    write_csv,
    write_source_crops,
)
from scripts.build_test_review_form import render_review_form
from utils.bcenet_visualization import qualitative_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-report")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--stage1-panel-size", type=int, default=512)
    parser.add_argument("--stage2-panel-size", type=int, default=320)
    return parser.parse_args()


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def review_guide() -> str:
    return """# Spatial split v2 — full test 100 qualitative audit

이 패키지는 spatial split v2의 frozen test 중앙 512×512 crop 100개 전부를
정량 결과와 같은 평가 범위로 검수하기 위한 독립 자료다. 원본 라벨,
checkpoint, 기존 평가 산출물은 수정하지 않았다.

## 가장 빠르게 전체 결과 보기

`results_gallery.html`을 브라우저로 열면 익명 ID `V001..V100` 순서로 GT와
prediction/error 패널을 함께 볼 수 있다. VS Code에서는 파일을 우클릭해
**Open with Default Browser** 또는 HTML preview를 사용한다.

직접 CSV 값을 입력하지 않으려면 `review_form.html`을 브라우저로 연다.
모든 categorical field를 dropdown으로 선택할 수 있고, 입력은 브라우저에
자동 저장된다. 작업 중간과 완료 시 `CSV 다운로드`를 눌러 보관한다.

## 편향을 줄인 정식 검수 순서

1. `01_stage1_gt_only/`만 열고 100장의 `stage1_*`를
   `full100_review.csv`에 작성한다.
2. stage 1이 끝나기 전에는 `02_stage2_predictions/`,
   `03_unblind/`, `results_gallery.html`을 열지 않는다.
3. `02_stage2_predictions/`를 열고 `stage2_*`를 작성한다.
4. `review_complete=yes`를 기록하고 CSV를 UTF-8로 저장한다.
5. 실제 sample ID와 sample별 수치는 마지막에
   `03_unblind/selection_manifest.csv`에서 확인한다.

`source_crops/<audit_id>/`에는 prediction이 없는 lossless 중앙 crop
(`before`, `reference`, raw/color GT state)이 있다. 확대 판정은 이 파일을
사용한다. 작성해야 할 파일은 `full100_review.csv` 하나뿐이다.

## 정량 결과

- `quantitative/overall_metrics.json`: test 100장 전체 pixel-micro 지표
- `quantitative/per_sample_metrics.csv`: 실제 sample ID별 지표
- `03_unblind/selection_manifest.csv`: 익명 ID, 실제 ID, spatial component,
  파일 경로, sample별 지표의 통합표
- `quantitative/SPATIAL_V2_BASELINE_REPORT.md`: split·학습·test 해석 보고서

threshold는 validation에서 고정한 0.5이며 test로 모델이나 threshold를
선택하지 않았다. 이 패키지의 100장은 train/validation과 source-1024 및
center-crop overlap이 0이다. 다만 test 내부에서는 23개 spatial component와
48쌍의 center-crop overlap이 있어 100개의 완전 독립 공간으로 간주하지 않는다.

## 색상

- footprint: cyan
- GT unchanged: green
- GT omission: orange
- GT excess: magenta
- stage 2 error: TP=green, FP=red, FN=blue, class confusion=yellow

## CSV 허용값

- overall GT: `correct`, `issue`, `mixed`, `uncertain`
- class GT: `correct`, `missing_labels`, `false_labels`, `wrong_class`,
  `boundary_issue`, `mixed`, `not_present`, `uncertain`
- metric eligible: `yes`, `no`, `uncertain`
- model: `correct`, `under_detection`, `over_detection`,
  `class_confusion`, `boundary_fragmentation`, `mixed`, `not_present`,
  `uncertain`
- attribution: `mostly_label`, `mostly_model`, `both`,
  `no_material_error`, `uncertain`
- confidence: `high`, `medium`, `low`

객체마다 판정이 다르면 `mixed`를 선택하고 notes에 위치별 판단을 적는다.
열 이름과 `audit_id`는 변경하지 않는다. 이 결과는 “BCE-Net 기반
noisy-label robust baseline”이며 논문 BCE-Net 정확 재현 성능이 아니다.
"""


def gallery_html(cards: list[dict[str, str]]) -> str:
    rendered_cards: list[str] = []
    for card in cards:
        audit_id = html.escape(card["audit_id"])
        metric = (
            f"omission F1 {float(card['omission_f1']):.4f} · "
            f"excess F1 {float(card['excess_f1']):.4f} · "
            f"combined F1 {float(card['combined_f1']):.4f}"
        )
        rendered_cards.append(
            f"""<article class="card" id="{audit_id}">
  <h2>{audit_id}</h2>
  <p>{html.escape(metric)}</p>
  <h3>GT only</h3>
  <a href="01_stage1_gt_only/{audit_id}.png"><img loading="lazy" src="01_stage1_gt_only/{audit_id}.png" alt="{audit_id} GT-only panel"></a>
  <h3>Prediction and error</h3>
  <a href="02_stage2_predictions/{audit_id}.png"><img loading="lazy" src="02_stage2_predictions/{audit_id}.png" alt="{audit_id} prediction panel"></a>
</article>"""
        )
    return """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spatial v2 full test audit</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
body { margin: 0; background: #11151a; color: #e8edf2; }
header { position: sticky; top: 0; z-index: 2; padding: 14px 22px; background: #182029ee; border-bottom: 1px solid #354454; }
header h1 { margin: 0 0 4px; font-size: 1.2rem; }
header p { margin: 0; color: #b7c5d3; }
main { display: grid; gap: 18px; padding: 20px; }
.card { padding: 16px; background: #1b232c; border: 1px solid #354454; border-radius: 10px; }
.card h2 { margin: 0; font-size: 1.15rem; }
.card h3 { margin: 14px 0 6px; color: #b7c5d3; font-size: .9rem; }
.card p { margin: 5px 0; color: #9fb2c5; font-variant-numeric: tabular-nums; }
img { display: block; width: 100%; height: auto; border-radius: 4px; background: #0b0e11; }
a { color: #8ecbff; }
</style>
</head>
<body>
<header><h1>Spatial split v2 — test 100</h1><p>GT와 prediction/error 전체 갤러리 · threshold 0.5 · 중앙 512×512</p></header>
<main>
""" + "\n".join(rendered_cards) + """
</main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    evaluation_dir = Path(args.evaluation_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    summary_report = Path(args.summary_report).resolve() if args.summary_report else None
    if checkpoint.name != "checkpoint-best.pth":
        raise ValueError("Full test package accepts checkpoint-best.pth only")
    if args.stage1_panel_size <= 0 or args.stage2_panel_size <= 0:
        raise ValueError("Panel sizes must be positive")

    metrics_path = evaluation_dir / "metrics.json"
    per_sample_path = evaluation_dir / "per_sample_metrics.csv"
    required = [manifest, checkpoint, metrics_path, per_sample_path]
    if summary_report is not None:
        required.append(summary_report)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))
    if evaluation.get("protocol") != PROTOCOL:
        raise ValueError(f"Unexpected frozen evaluation protocol: {evaluation.get('protocol')}")
    scope = evaluation.get("scope", {})
    if scope.get("sample_count") != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} evaluation samples")
    if scope.get("evaluated_crop_shape") != [CROP_SIZE, CROP_SIZE]:
        raise ValueError("Frozen evaluation is not the central 512 crop protocol")
    if scope.get("crop_origin") != [256, 256] or scope.get("sliding_window"):
        raise ValueError("Frozen evaluation scope is not central-crop-only")
    if evaluation.get("selection_policy", {}).get("threshold") != THRESHOLD:
        raise ValueError("Frozen evaluation threshold is not 0.5")
    if evaluation["checkpoint"]["sha256"] != sha256(checkpoint):
        raise ValueError("Checkpoint hash differs from frozen evaluation")
    if evaluation["manifest"]["sha256"] != sha256(manifest):
        raise ValueError("Manifest hash differs from frozen evaluation")

    source_rows = [
        row for row in read_manifest_rows(manifest)
        if row["split"].strip().lower() == "test"
    ]
    source_by_id = {row["sample_id"]: row for row in source_rows}
    component_counts = Counter(row["spatial_component_v2"] for row in source_rows)
    if len(source_rows) != EXPECTED_SAMPLES or len(source_by_id) != EXPECTED_SAMPLES:
        raise ValueError("Manifest must contain 100 unique test samples")

    prepare_output_dir(output_dir)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, checkpoint_metadata = build_model(checkpoint)
    model = model.to(device).eval()
    dataset = MapOrthoBCENetDataset(
        manifest,
        split="test",
        crop_size=CROP_SIZE,
        train_jitter=0,
        augment=False,
        imagenet_normalize=checkpoint_metadata["imagenet_normalize"],
    )
    metric_rows = read_csv(per_sample_path)
    row_by_id = {row["sample_id"]: row for row in metric_rows}
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    if len(metric_rows) != EXPECTED_SAMPLES or len(row_by_id) != EXPECTED_SAMPLES:
        raise ValueError("Per-sample metrics must contain 100 unique samples")
    if set(row_by_id) != set(sample_by_id) or set(row_by_id) != set(source_by_id):
        raise ValueError("Evaluation, dataset, and manifest test IDs do not match")

    presentation_order = sorted(sample_by_id)
    random.Random(args.seed).shuffle(presentation_order)
    sample_to_index = {
        sample.sample_id: index for index, sample in enumerate(dataset.samples)
    }
    review_rows: list[dict[str, str]] = []
    unblind_rows: list[dict[str, Any]] = []
    gallery_cards: list[dict[str, str]] = []
    for order_index, sample_id in enumerate(presentation_order, start=1):
        audit_id = f"V{order_index:03d}"
        record = infer_qualitative_record(
            model=model,
            dataset=dataset,
            index=sample_to_index[sample_id],
            device=device,
            amp=args.amp,
        )
        blinded_record = dict(record)
        blinded_record["sample_id"] = audit_id
        save_rgb(
            output_dir / "01_stage1_gt_only" / f"{audit_id}.png",
            stage1_row(blinded_record, audit_id, args.stage1_panel_size),
        )
        save_rgb(
            output_dir / "02_stage2_predictions" / f"{audit_id}.png",
            qualitative_row(
                blinded_record,
                threshold=THRESHOLD,
                imagenet_normalized=checkpoint_metadata["imagenet_normalize"],
                panel_size=args.stage2_panel_size,
            ),
        )
        write_source_crops(output_dir, audit_id, record)
        review_rows.append(
            {
                column: audit_id if column == "audit_id" else ""
                for column in REVIEW_COLUMNS
            }
        )

        source = source_by_id[sample_id]
        metric = row_by_id[sample_id]
        metric_payload = {
            key: value for key, value in metric.items() if key != "sample_id"
        }
        unblind_rows.append(
            {
                "audit_id": audit_id,
                "sample_id": sample_id,
                "center_class": source["center_class"],
                "center_class_value": source["center_class_value"],
                "spatial_component_v2": source["spatial_component_v2"],
                "component_sample_count": component_counts[source["spatial_component_v2"]],
                "source_raster_id": source["source_raster_id"],
                "xoff": source["xoff"],
                "yoff": source["yoff"],
                "source_gsd_m": source["source_gsd_m"],
                "image_path": source["image_path"],
                "map_mask_path": source["map_mask_path"],
                "label_mask_path": source["label_mask_path"],
                "stage1_panel": f"01_stage1_gt_only/{audit_id}.png",
                "stage2_panel": f"02_stage2_predictions/{audit_id}.png",
                "source_crop_dir": f"source_crops/{audit_id}",
                **metric_payload,
            }
        )
        gallery_cards.append({"audit_id": audit_id, **metric})

    write_csv(output_dir / "full100_review.csv", review_rows, REVIEW_COLUMNS)
    write_csv(
        output_dir / "03_unblind" / "selection_manifest.csv",
        unblind_rows,
        list(unblind_rows[0]),
    )
    quantitative_dir = output_dir / "quantitative"
    quantitative_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(metrics_path, quantitative_dir / "overall_metrics.json")
    shutil.copy2(per_sample_path, quantitative_dir / "per_sample_metrics.csv")
    if summary_report is not None:
        shutil.copy2(summary_report, quantitative_dir / "SPATIAL_V2_BASELINE_REPORT.md")

    package_metadata = {
        "package": "map-ortho-test-audit-full100-spatial-v2",
        "sample_count": EXPECTED_SAMPLES,
        "audit_id_range": "V001..V100",
        "presentation_order_blinded": True,
        "presentation_seed": args.seed,
        "canonical_review_csv": "full100_review.csv",
        "interactive_review_form": "review_form.html",
        "results_gallery": "results_gallery.html",
        "scope": evaluation["scope"],
        "selection_policy": evaluation["selection_policy"],
        "checkpoint": evaluation["checkpoint"],
        "manifest": evaluation["manifest"],
        "threshold": THRESHOLD,
        "stage1_contains_predictions": False,
        "source_labels_modified": False,
        "spatial_independence": {
            "cross_split_source_overlap_pairs": 0,
            "cross_split_center_crop_overlap_pairs": 0,
            "within_test_spatial_components": len(component_counts),
            "within_test_center_crop_overlap_pairs": 48,
            "interpretation": "cross_split_disjoint_within_test_correlated",
        },
        "model_description": evaluation["model_description"],
        "paper_exact_reproduction": evaluation["paper_exact_reproduction"],
    }
    (output_dir / "package_metadata.json").write_text(
        json.dumps(package_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(review_guide(), encoding="utf-8")
    (output_dir / "review_form.html").write_text(
        render_review_form(review_rows), encoding="utf-8"
    )
    (output_dir / "results_gallery.html").write_text(
        gallery_html(gallery_cards), encoding="utf-8"
    )
    print(json.dumps(package_metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
