#!/usr/bin/env python3
"""Build a blinded, two-stage qualitative audit package for map-ortho test data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
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
from utils.bcenet_visualization import (
    STATE_COLORS,
    _blend_mask,
    _image_from_tensor,
    _mask,
    _titled,
    qualitative_row,
)


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
    parser.add_argument("--manifest", default="dataset/map_ortho_manifest.csv")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--stage1-panel-size", type=int, default=512)
    parser.add_argument("--stage2-panel-size", type=int, default=320)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write image: {path}")


def save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def stage1_row(
    record: dict[str, Any], audit_id: str, panel_size: int
) -> np.ndarray:
    image = _image_from_tensor(record["image"], imagenet_normalized=False)
    reference = _mask(record["reference"])
    target_existing = _mask(record["target_existing"])
    target_omission = _mask(record["target_omission"])
    target_excess = _mask(record["target_excess"])
    target_unchanged = target_existing & reference & ~target_excess
    reference_panel = _blend_mask(
        image,
        [(reference, np.array([35, 205, 235], dtype=np.uint8))],
    )
    target_panel = _blend_mask(
        image,
        [
            (target_unchanged, STATE_COLORS["unchanged"]),
            (target_omission, STATE_COLORS["omission"]),
            (target_excess, STATE_COLORS["excess"]),
        ],
    )
    return np.concatenate(
        [
            _titled(image, f"Before: {audit_id}", panel_size),
            _titled(reference_panel, "After footprint (cyan)", panel_size),
            _titled(target_panel, "GT state", panel_size),
        ],
        axis=1,
    )


def write_source_crops(
    output_dir: Path,
    audit_id: str,
    record: dict[str, Any],
) -> None:
    sample_dir = output_dir / "source_crops" / audit_id
    image = _image_from_tensor(record["image"], imagenet_normalized=False)
    reference = _mask(record["reference"])
    target_existing = _mask(record["target_existing"])
    target_omission = _mask(record["target_omission"])
    target_excess = _mask(record["target_excess"])
    target_unchanged = target_existing & reference & ~target_excess
    state = np.zeros(reference.shape, dtype=np.uint8)
    state[target_unchanged] = 1
    state[target_omission] = 2
    state[target_excess] = 3
    state_color = np.zeros((*reference.shape, 3), dtype=np.uint8)
    state_color[:] = np.array([35, 35, 35], dtype=np.uint8)
    state_color[target_unchanged] = STATE_COLORS["unchanged"]
    state_color[target_omission] = STATE_COLORS["omission"]
    state_color[target_excess] = STATE_COLORS["excess"]
    save_rgb(sample_dir / "before.png", image)
    save_gray(sample_dir / "reference.png", reference.astype(np.uint8) * 255)
    save_gray(sample_dir / "state_gt_raw.png", state)
    save_rgb(sample_dir / "state_gt_color.png", state_color)


def select_samples(
    *,
    dataset: MapOrthoBCENetDataset,
    metric_rows: list[dict[str, str]],
    qualitative_selections: dict[str, list[dict[str, Any]]],
    seed: int,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    row_by_id = {row["sample_id"]: row for row in metric_rows}
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    if set(row_by_id) != set(sample_by_id):
        raise ValueError("Per-sample metrics and test dataset IDs do not match")

    reasons: defaultdict[str, list[str]] = defaultdict(list)
    error_ranked: list[str] = []
    for selection_name, items in qualitative_selections.items():
        if not selection_name.startswith("top_"):
            continue
        for item in items:
            sample_id = str(item["sample_id"])
            reasons[sample_id].append(selection_name)
            if sample_id not in error_ranked:
                error_ranked.append(sample_id)
    if len(error_ranked) != 16:
        raise ValueError(
            f"Expected 16 unique error-ranked samples from frozen evaluation, "
            f"found {len(error_ranked)}"
        )

    selected = set(error_ranked)
    high_agreement: list[str] = []
    for center_class in (2, 3):
        candidates = [
            sample.sample_id
            for sample in dataset.samples
            if sample.center_class_value == center_class
            and sample.sample_id not in selected
        ]
        candidates.sort(
            key=lambda sample_id: (
                -float(row_by_id[sample_id]["combined_f1"]),
                -float(row_by_id[sample_id]["combined_iou"]),
                sample_id,
            )
        )
        chosen = candidates[:2]
        high_agreement.extend(chosen)
        selected.update(chosen)

    rng = random.Random(seed)
    random_controls: list[str] = []
    for center_class in (2, 3):
        candidates = [
            sample.sample_id
            for sample in dataset.samples
            if sample.center_class_value == center_class
            and sample.sample_id not in selected
        ]
        chosen = rng.sample(candidates, 5)
        random_controls.extend(chosen)
        selected.update(chosen)

    if len(selected) != 30:
        raise RuntimeError(f"Expected 30 unique pilot samples, got {len(selected)}")
    presentation_order = sorted(selected)
    rng.shuffle(presentation_order)

    selection_metadata: dict[str, dict[str, str]] = {}
    for sample_id in presentation_order:
        if sample_id in error_ranked:
            stratum = "error_ranked"
        elif sample_id in high_agreement:
            stratum = "high_agreement_control"
        else:
            stratum = "random_control"
        selection_metadata[sample_id] = {
            "selection_stratum": stratum,
            "selection_reasons": "|".join(reasons.get(sample_id, [])),
        }
    return presentation_order, selection_metadata


def review_guide() -> str:
    return """# Test pilot qualitative audit v1

이 패키지는 30개의 중앙 512×512 test crop을 두 단계로 검수하기 위한
자료다. 원본 라벨이나 기존 평가 결과를 수정하지 않는다.

## 반드시 지킬 순서

1. `01_stage1_gt_only/`만 열고 `pilot_review.csv`의 `stage1_*` 열을 모두 작성한다.
2. stage 1 작성이 끝나기 전에는 `02_stage2_predictions/`와
   `03_unblind/`를 열지 않는다.
3. 그 다음 `02_stage2_predictions/`를 열고 `stage2_*` 열을 작성한다.
4. `review_complete=yes`를 기록하고 CSV를 UTF-8 형식으로 저장한다.
5. Codex에게 파일 경로를 알려주거나 `pilot_review.csv`를 대화에 첨부한다.

실제 sample ID, 표본 선정 이유와 모델 지표는 `03_unblind/selection_manifest.csv`
에 있다. stage 1 라벨 판정이 모델 오류 순위에 끌리지 않도록 마지막에만 연다.

## 색상

- footprint: cyan
- GT unchanged: green
- GT omission: orange
- GT excess: magenta
- stage 2 error: TP=green, FP=red, FN=blue, class confusion=yellow

`source_crops/<audit_id>/`에는 zoom 검수용 lossless before, reference,
raw state label과 color state label이 있다. 이 폴더에는 prediction이 없다.

## stage 1 허용값

`stage1_overall_gt_status`:

- `correct`: metric에 사용할 수 있는 정상 GT
- `issue`: 명확한 GT 오류 또는 누락
- `mixed`: 정상 객체와 오류 객체가 함께 있음
- `uncertain`: 영상만으로 판정 불가

`stage1_omission_gt_status`, `stage1_excess_gt_status`:

- `correct`
- `missing_labels`
- `false_labels`
- `wrong_class`
- `boundary_issue`
- `mixed`
- `not_present`
- `uncertain`

`stage1_metric_eligible`: `yes`, `no`, `uncertain`

`stage1_confidence`: `high`, `medium`, `low`

## stage 2 허용값

`stage2_omission_model_status`, `stage2_excess_model_status`:

- `correct`
- `under_detection`
- `over_detection`
- `class_confusion`
- `boundary_fragmentation`
- `mixed`
- `not_present`
- `uncertain`

`stage2_error_attribution`:

- `mostly_label`
- `mostly_model`
- `both`
- `no_material_error`
- `uncertain`

`stage2_confidence`: `high`, `medium`, `low`

`review_complete`: `yes` 또는 검수 중이면 빈 칸

객체가 여러 개여서 하나의 enum으로 부족하면 `mixed`를 선택하고 notes에
위치와 판단을 적는다. 예: `왼쪽 큰 주황 polygon은 테니스 코트로 보여
false label, 오른쪽 작은 지붕 omission은 정상`.

## 결과를 Codex에 전달하는 방법

공유 workspace에서 직접 편집했다면 다음 파일을 그대로 저장한 뒤
"pilot_review.csv 작성 완료"라고 알려주면 된다.

```text
training_monitor/test-pilot-audit-v1-20260721/pilot_review.csv
```

다른 로컬 PC로 복사해 작성했다면 완성된 `pilot_review.csv`를 대화에
첨부한다. 열 이름과 `audit_id`는 변경하지 않는다.
"""


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    evaluation_dir = Path(args.evaluation_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if checkpoint.name != "checkpoint-best.pth":
        raise ValueError("Pilot package accepts checkpoint-best.pth only")
    if args.stage1_panel_size <= 0 or args.stage2_panel_size <= 0:
        raise ValueError("Panel sizes must be positive")
    metrics_path = evaluation_dir / "metrics.json"
    per_sample_path = evaluation_dir / "per_sample_metrics.csv"
    selections_path = evaluation_dir / "qualitative" / "selections.json"
    for path in (manifest, checkpoint, metrics_path, per_sample_path, selections_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))
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
    metric_rows = read_csv(per_sample_path)
    qualitative_selections = json.loads(selections_path.read_text(encoding="utf-8"))
    presentation_order, selection_metadata = select_samples(
        dataset=dataset,
        metric_rows=metric_rows,
        qualitative_selections=qualitative_selections,
        seed=args.seed,
    )
    sample_to_index = {
        sample.sample_id: index for index, sample in enumerate(dataset.samples)
    }
    row_by_id = {row["sample_id"]: row for row in metric_rows}
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}

    review_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, Any]] = []
    for order_index, sample_id in enumerate(presentation_order, start=1):
        audit_id = f"P{order_index:03d}"
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
        review_rows.append({column: audit_id if column == "audit_id" else "" for column in REVIEW_COLUMNS})
        sample = sample_by_id[sample_id]
        manifest_rows.append(
            {
                "audit_id": audit_id,
                "sample_id": sample_id,
                "center_class_value": sample.center_class_value,
                "selection_stratum": selection_metadata[sample_id]["selection_stratum"],
                "selection_reasons": selection_metadata[sample_id]["selection_reasons"],
                "crop_y": 256,
                "crop_x": 256,
                "image_path": sample.image_path,
                "map_mask_path": sample.map_mask_path,
                "label_mask_path": sample.label_mask_path,
                **row_by_id[sample_id],
            }
        )

    write_csv(output_dir / "pilot_review.csv", review_rows, REVIEW_COLUMNS)
    manifest_fields = list(manifest_rows[0])
    write_csv(
        output_dir / "03_unblind" / "selection_manifest.csv",
        manifest_rows,
        manifest_fields,
    )
    package_metadata = {
        "package": "map-ortho-test-pilot-audit-v1",
        "sample_count": len(presentation_order),
        "selection": {
            "error_ranked": 16,
            "high_agreement_control": 4,
            "random_control": 10,
            "presentation_order_blinded": True,
            "seed": args.seed,
        },
        "scope": evaluation["scope"],
        "checkpoint": evaluation["checkpoint"],
        "manifest": evaluation["manifest"],
        "threshold": THRESHOLD,
        "stage1_contains_predictions": False,
        "source_labels_modified": False,
    }
    (output_dir / "package_metadata.json").write_text(
        json.dumps(package_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(review_guide(), encoding="utf-8")
    print(json.dumps(package_metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
