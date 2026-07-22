#!/usr/bin/env python3
"""Build a blinded two-stage audit package for the 70 test samples outside pilot."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
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
from utils.bcenet_visualization import qualitative_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="dataset/map_ortho_manifest.csv")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pilot-selection-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--stage1-panel-size", type=int, default=512)
    parser.add_argument("--stage2-panel-size", type=int, default=320)
    return parser.parse_args()


def review_guide() -> str:
    return """# Remaining 70 test qualitative audit v1

이 패키지는 pilot 30장을 제외한 나머지 test 70개의 중앙 512×512 crop을
같은 기준으로 검수하기 위한 자료다. 가능하면 pilot 30장 검수를 먼저
마치고 판정 기준을 고정한 뒤 이 패키지를 시작한다.

## 반드시 지킬 순서

1. `01_stage1_gt_only/`만 열고 `remaining70_review.csv`의 `stage1_*` 열을
   모두 작성한다.
2. stage 1이 끝나기 전에는 `02_stage2_predictions/`와 `03_unblind/`를
   열지 않는다.
3. `02_stage2_predictions/`를 열고 `stage2_*` 열을 작성한다.
4. `review_complete=yes`를 기록하고 UTF-8 CSV로 저장한다.
5. Codex에게 아래 경로의 작성 완료를 알리거나 CSV를 대화에 첨부한다.

```text
training_monitor/test-audit-remaining70-v1-20260722/remaining70_review.csv
```

실제 sample ID와 frozen model metric은
`03_unblind/selection_manifest.csv`에 있으며 마지막에만 확인한다.
`source_crops/<audit_id>/`는 prediction이 없는 lossless zoom 자료다.

## 색상

- footprint: cyan
- GT unchanged: green
- GT omission: orange
- GT excess: magenta
- stage 2 error: TP=green, FP=red, FN=blue, class confusion=yellow

## stage 1 허용값

- `stage1_overall_gt_status`: `correct`, `issue`, `mixed`, `uncertain`
- class별 GT status: `correct`, `missing_labels`, `false_labels`,
  `wrong_class`, `boundary_issue`, `mixed`, `not_present`, `uncertain`
- `stage1_metric_eligible`: `yes`, `no`, `uncertain`
- confidence: `high`, `medium`, `low`

## stage 2 허용값

- class별 model status: `correct`, `under_detection`, `over_detection`,
  `class_confusion`, `boundary_fragmentation`, `mixed`, `not_present`,
  `uncertain`
- `stage2_error_attribution`: `mostly_label`, `mostly_model`, `both`,
  `no_material_error`, `uncertain`
- confidence: `high`, `medium`, `low`
- `review_complete`: `yes` 또는 검수 중이면 빈 칸

객체별 판단이 다르면 `mixed`를 선택하고 notes에 위치와 판단을 적는다.
열 이름과 `audit_id`는 변경하지 않는다.
"""


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    evaluation_dir = Path(args.evaluation_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    pilot_manifest_path = Path(args.pilot_selection_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    if checkpoint.name != "checkpoint-best.pth":
        raise ValueError("Remaining audit package accepts checkpoint-best.pth only")
    required = [
        manifest,
        checkpoint,
        pilot_manifest_path,
        evaluation_dir / "metrics.json",
        evaluation_dir / "per_sample_metrics.csv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    evaluation = json.loads(
        (evaluation_dir / "metrics.json").read_text(encoding="utf-8")
    )
    if evaluation.get("protocol") != "map-ortho-test-center-crop-512-v1":
        raise ValueError("Unexpected frozen evaluation protocol")
    if evaluation["checkpoint"]["sha256"] != sha256(checkpoint):
        raise ValueError("Checkpoint hash differs from frozen evaluation")
    if evaluation["manifest"]["sha256"] != sha256(manifest):
        raise ValueError("Manifest hash differs from frozen evaluation")

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
    metric_rows = read_csv(evaluation_dir / "per_sample_metrics.csv")
    row_by_id = {row["sample_id"]: row for row in metric_rows}
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    if set(row_by_id) != set(sample_by_id):
        raise ValueError("Per-sample metrics and test dataset IDs do not match")

    pilot_rows = read_csv(pilot_manifest_path)
    pilot_ids = {row["sample_id"] for row in pilot_rows}
    if len(pilot_rows) != 30 or len(pilot_ids) != 30:
        raise ValueError("Pilot selection manifest must contain 30 unique samples")
    if not pilot_ids.issubset(sample_by_id):
        raise ValueError("Pilot manifest contains IDs outside the test split")
    remaining_ids = sorted(set(sample_by_id) - pilot_ids)
    if len(remaining_ids) != 70:
        raise ValueError(f"Expected 70 remaining test samples, found {len(remaining_ids)}")
    random.Random(args.seed).shuffle(remaining_ids)

    sample_to_index = {
        sample.sample_id: index for index, sample in enumerate(dataset.samples)
    }
    review_rows: list[dict[str, str]] = []
    unblind_rows: list[dict[str, Any]] = []
    for order_index, sample_id in enumerate(remaining_ids, start=1):
        audit_id = f"R{order_index:03d}"
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
        sample = sample_by_id[sample_id]
        unblind_rows.append(
            {
                "audit_id": audit_id,
                "sample_id": sample_id,
                "center_class_value": sample.center_class_value,
                "selection_stratum": "remaining_test",
                "crop_y": 256,
                "crop_x": 256,
                "image_path": sample.image_path,
                "map_mask_path": sample.map_mask_path,
                "label_mask_path": sample.label_mask_path,
                **row_by_id[sample_id],
            }
        )

    write_csv(
        output_dir / "remaining70_review.csv", review_rows, REVIEW_COLUMNS
    )
    write_csv(
        output_dir / "03_unblind" / "selection_manifest.csv",
        unblind_rows,
        list(unblind_rows[0]),
    )
    metadata = {
        "package": "map-ortho-test-audit-remaining70-v1",
        "sample_count": 70,
        "selection": {
            "definition": "test split minus the frozen pilot 30 IDs",
            "presentation_order_blinded": True,
            "seed": args.seed,
            "pilot_selection_manifest": str(pilot_manifest_path),
        },
        "scope": evaluation["scope"],
        "checkpoint": evaluation["checkpoint"],
        "manifest": evaluation["manifest"],
        "threshold": THRESHOLD,
        "stage1_contains_predictions": False,
        "source_labels_modified": False,
    }
    (output_dir / "package_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(review_guide(), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
