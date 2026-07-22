#!/usr/bin/env python3
"""Combine the blinded 30-sample pilot and remaining 70 into one test package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


STAGE_DIRECTORIES = (
    "01_stage1_gt_only",
    "02_stage2_predictions",
    "source_crops",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", required=True)
    parser.add_argument("--remaining-dir", required=True)
    parser.add_argument("--output-dir", required=True)
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


def prepare_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"Output directory must be new or empty: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def merge_fields(rows: list[dict[str, str]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def guide() -> str:
    return """# Full test 100 qualitative audit v1

이 디렉터리는 pilot 30장과 remaining 70장을 합친 canonical 검수
패키지다. `P001..P030`과 `R001..R070` ID를 유지하므로 기존 자료와
대응이 끊기지 않는다.

## 작성 파일

앞으로 다음 CSV 하나만 작성한다.

```text
training_monitor/test-audit-full100-v1-20260722/full100_review.csv
```

## 검수 순서

1. `01_stage1_gt_only/`만 열고 100장 모두의 `stage1_*` 열을 작성한다.
2. stage 1 완료 전에는 `02_stage2_predictions/`와 `03_unblind/`를 열지 않는다.
3. stage 1 완료 후 prediction 패널을 열고 `stage2_*` 열을 작성한다.
4. `review_complete=yes`를 기록하고 UTF-8 CSV로 저장한다.
5. Codex에게 `full100_review.csv 작성 완료`라고 알려주거나 파일을 첨부한다.

`source_crops/<audit_id>/`에는 prediction이 없는 lossless before/reference/
state 자료가 있다. 실제 sample ID와 frozen metric은 마지막에 확인할
`03_unblind/selection_manifest.csv`에 있다.

## 허용값

- overall GT: `correct`, `issue`, `mixed`, `uncertain`
- class GT: `correct`, `missing_labels`, `false_labels`, `wrong_class`,
  `boundary_issue`, `mixed`, `not_present`, `uncertain`
- metric eligible: `yes`, `no`, `uncertain`
- model: `correct`, `under_detection`, `over_detection`, `class_confusion`,
  `boundary_fragmentation`, `mixed`, `not_present`, `uncertain`
- attribution: `mostly_label`, `mostly_model`, `both`, `no_material_error`,
  `uncertain`
- confidence: `high`, `medium`, `low`

객체마다 판정이 다르면 `mixed`를 선택하고 notes에 위치별 판단을 적는다.
열 이름과 `audit_id`는 변경하지 않는다.

## 해석 제한

현재 v1 split은 파일 경로는 분리되어 있지만 cross-split exact pixel
overlap이 있다. 이 패키지는 라벨 품질과 모델 오류를 전수 검수하기 위한
자료이며, 현재 test metric을 strict spatial generalization으로 만드는
자료는 아니다. overlap 근거는 다음 위치에 있다.

```text
training_monitor/test-split-overlap-audit-20260722/REPORT.md
```
"""


def main() -> int:
    args = parse_args()
    pilot_dir = Path(args.pilot_dir).resolve()
    remaining_dir = Path(args.remaining_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    for directory in (pilot_dir, remaining_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    prepare_output_dir(output_dir)

    pilot_metadata = json.loads(
        (pilot_dir / "package_metadata.json").read_text(encoding="utf-8")
    )
    remaining_metadata = json.loads(
        (remaining_dir / "package_metadata.json").read_text(encoding="utf-8")
    )
    for key in ("checkpoint", "manifest", "threshold", "scope"):
        if pilot_metadata[key] != remaining_metadata[key]:
            raise ValueError(f"Source package metadata differs for {key}")

    pilot_reviews = read_csv(pilot_dir / "pilot_review.csv")
    remaining_reviews = read_csv(remaining_dir / "remaining70_review.csv")
    review_rows = pilot_reviews + remaining_reviews
    if len(review_rows) != 100:
        raise ValueError(f"Expected 100 review rows, found {len(review_rows)}")
    audit_ids = [row["audit_id"] for row in review_rows]
    if len(set(audit_ids)) != 100:
        raise ValueError("Review audit IDs are not unique")
    review_fields = list(review_rows[0])
    if any(list(row) != review_fields for row in review_rows):
        raise ValueError("Review CSV schemas differ")

    pilot_unblind = read_csv(
        pilot_dir / "03_unblind" / "selection_manifest.csv"
    )
    remaining_unblind = read_csv(
        remaining_dir / "03_unblind" / "selection_manifest.csv"
    )
    unblind_rows = pilot_unblind + remaining_unblind
    sample_ids = [row["sample_id"] for row in unblind_rows]
    if len(unblind_rows) != 100 or len(set(sample_ids)) != 100:
        raise ValueError("Unblind manifests do not contain 100 unique samples")
    if {row["audit_id"] for row in unblind_rows} != set(audit_ids):
        raise ValueError("Review and unblind audit IDs differ")

    for stage_directory in STAGE_DIRECTORIES:
        destination = output_dir / stage_directory
        destination.mkdir(parents=True, exist_ok=False)
        for source_root in (pilot_dir, remaining_dir):
            for source in sorted((source_root / stage_directory).iterdir()):
                target = destination / source.name
                if target.exists():
                    raise FileExistsError(f"Duplicate package item: {target}")
                if source.is_dir():
                    shutil.copytree(source, target, copy_function=shutil.copy2)
                else:
                    shutil.copy2(source, target)

    write_csv(
        output_dir / "full100_review.csv",
        review_rows,
        review_fields,
    )
    unblind_fields = merge_fields(unblind_rows)
    write_csv(
        output_dir / "03_unblind" / "selection_manifest.csv",
        [
            {field: row.get(field, "") for field in unblind_fields}
            for row in unblind_rows
        ],
        unblind_fields,
    )
    metadata = {
        "package": "map-ortho-test-audit-full100-v1",
        "sample_count": 100,
        "source_packages": [str(pilot_dir), str(remaining_dir)],
        "audit_id_ranges": ["P001..P030", "R001..R070"],
        "canonical_review_csv": "full100_review.csv",
        "scope": pilot_metadata["scope"],
        "checkpoint": pilot_metadata["checkpoint"],
        "manifest": pilot_metadata["manifest"],
        "threshold": pilot_metadata["threshold"],
        "stage1_contains_predictions": False,
        "source_labels_modified": False,
        "spatial_independence": False,
        "spatial_overlap_report": (
            "training_monitor/test-split-overlap-audit-20260722/REPORT.md"
        ),
    }
    (output_dir / "package_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(guide(), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
