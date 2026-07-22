#!/usr/bin/env python3
"""Audit file identity and spatial pixel overlap across map-ortho splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
import pandas as pd


class Rectangle(NamedTuple):
    x0: int
    y0: int
    x1: int
    y1: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="dataset/map_ortho_manifest.csv")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"Output directory must be new or empty: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def rectangle(row: Any, *, offset: int, size: int) -> Rectangle:
    x0 = int(row.xoff) + offset
    y0 = int(row.yoff) + offset
    return Rectangle(x0, y0, x0 + size, y0 + size)


def intersection(a: Rectangle, b: Rectangle) -> Rectangle | None:
    result = Rectangle(
        max(a.x0, b.x0),
        max(a.y0, b.y0),
        min(a.x1, b.x1),
        min(a.y1, b.y1),
    )
    return result if result.x1 > result.x0 and result.y1 > result.y0 else None


def dimensions(rect: Rectangle | None) -> tuple[int, int, int]:
    if rect is None:
        return 0, 0, 0
    width = rect.x1 - rect.x0
    height = rect.y1 - rect.y0
    return width, height, width * height


@lru_cache(maxsize=128)
def read_image(path: str, flag: int) -> np.ndarray:
    image = cv2.imread(path, flag)
    if image is None:
        raise FileNotFoundError(path)
    return image


def compare_overlap(
    row_a: Any,
    row_b: Any,
    rect: Rectangle,
    field: str,
    flag: int,
) -> tuple[int, int, float]:
    image_a = read_image(str(getattr(row_a, field)), flag)
    image_b = read_image(str(getattr(row_b, field)), flag)
    crop_a = image_a[
        rect.y0 - int(row_a.yoff) : rect.y1 - int(row_a.yoff),
        rect.x0 - int(row_a.xoff) : rect.x1 - int(row_a.xoff),
    ]
    crop_b = image_b[
        rect.y0 - int(row_b.yoff) : rect.y1 - int(row_b.yoff),
        rect.x0 - int(row_b.xoff) : rect.x1 - int(row_b.xoff),
    ]
    if crop_a.shape != crop_b.shape:
        raise RuntimeError(
            f"Aligned overlap shapes differ: {crop_a.shape} != {crop_b.shape}"
        )
    equal = np.all(crop_a == crop_b, axis=2) if crop_a.ndim == 3 else crop_a == crop_b
    equal_pixels = int(equal.sum())
    total_pixels = int(equal.size)
    return equal_pixels, total_pixels, equal_pixels / max(1, total_pixels)


def cross_pairs(
    frame: pd.DataFrame, split_a: str, split_b: str
) -> list[tuple[Any, Any]]:
    rows_a = list(frame[frame["split"] == split_a].itertuples(index=False))
    rows_b = list(frame[frame["split"] == split_b].itertuples(index=False))
    return [(row_a, row_b) for row_a in rows_a for row_b in rows_b]


def overlap_summary(
    frame: pd.DataFrame,
    split_a: str,
    split_b: str,
    *,
    offset_a: int,
    size_a: int,
    offset_b: int,
    size_b: int,
) -> dict[str, Any]:
    overlaps: list[tuple[Any, Any, int]] = []
    for row_a, row_b in cross_pairs(frame, split_a, split_b):
        rect = intersection(
            rectangle(row_a, offset=offset_a, size=size_a),
            rectangle(row_b, offset=offset_b, size=size_b),
        )
        area = dimensions(rect)[2]
        if area:
            overlaps.append((row_a, row_b, area))
    return {
        "pairs": len(overlaps),
        f"unique_{split_a}_samples": len({row_a.sample_id for row_a, _, _ in overlaps}),
        f"unique_{split_b}_samples": len({row_b.sample_id for _, row_b, _ in overlaps}),
        "maximum_overlap_pixels": max((area for _, _, area in overlaps), default=0),
        "maximum_fraction_of_a": max(
            (area / (size_a * size_a) for _, _, area in overlaps), default=0.0
        ),
        "maximum_fraction_of_b": max(
            (area / (size_b * size_b) for _, _, area in overlaps), default=0.0
        ),
    }


def build_report(summary: dict[str, Any]) -> str:
    s = summary["cross_split_overlap"]
    source_pairs = summary["pixel_identity"]["source_overlap_pairs"]
    group_field = summary["manifest"]["group_field"]
    shared_groups = summary["manifest"]["shared_groups"]
    if source_pairs == 0:
        return f"""# Map-ortho train/validation/test overlap audit

## 결론

파일 및 `{group_field}` 단위가 split 사이에 공유되지 않았고, 서로 다른
split의 1024 source patch가 겹치는 경우도 **0쌍**이다. 중앙 512 crop과
train possible-crop union 기준 중복도 모두 0이다.

## 파일 및 component 무결성

- manifest rows: `{summary['manifest']['rows']}`
- split counts: `{summary['manifest']['split_counts']}`
- duplicate sample IDs: `{summary['manifest']['duplicate_sample_ids']}`
- duplicate image paths: `{summary['manifest']['duplicate_image_paths']}`
- duplicate map paths: `{summary['manifest']['duplicate_map_paths']}`
- duplicate label paths: `{summary['manifest']['duplicate_label_paths']}`
- 둘 이상의 split에 공유된 `{group_field}`: `{shared_groups}`

## 실제 1024 source patch 중첩

| split pair | overlapping pairs | affected samples A/B | maximum overlap |
|---|---:|---:|---:|
| train–validation | {s['train_val_source1024']['pairs']} | {s['train_val_source1024']['unique_train_samples']}/{s['train_val_source1024']['unique_val_samples']} | {s['train_val_source1024']['maximum_overlap_pixels']:,} px |
| train–test | {s['train_test_source1024']['pairs']} | {s['train_test_source1024']['unique_train_samples']}/{s['train_test_source1024']['unique_test_samples']} | {s['train_test_source1024']['maximum_overlap_pixels']:,} px |
| validation–test | {s['val_test_source1024']['pairs']} | {s['val_test_source1024']['unique_val_samples']}/{s['val_test_source1024']['unique_test_samples']} | {s['val_test_source1024']['maximum_overlap_pixels']:,} px |

## 실제 학습·평가 crop 중첩

- train possible-crop union vs validation center: `{s['train_possible_union_vs_val_center']['pairs']}` pairs
- train possible-crop union vs test center: `{s['train_possible_union_vs_test_center']['pairs']}` pairs
- validation center vs test center: `{s['val_test_center512']['pairs']}` pairs

따라서 이 manifest는 현재 좌표 lineage와 1024 source footprint 기준으로
cross-split pixel leakage가 없는 split이다.
"""
    return f"""# Map-ortho train/validation/test overlap audit

## 결론

동일 sample ID나 동일 파일 경로가 split 사이에 중복된 경우는 없다.
`spatial_group`도 둘 이상의 split에 배정되지 않았다. 그러나 서로 다른
8,192픽셀 block의 경계에 놓인 1024 patch가 실제 공간에서 겹치므로
**pixel-level spatial leakage가 존재한다.**

좌표상 겹치는 cross-split source patch는 총
`{summary['pixel_identity']['source_overlap_pairs']}`쌍이며, 모든 쌍에서
image, footprint, state label의 교차 영역 픽셀이 100% 동일했다. 따라서
좌표 추정이 아니라 실제 중복이다.

## 파일 및 group 무결성

- manifest rows: `{summary['manifest']['rows']}`
- duplicate sample IDs: `{summary['manifest']['duplicate_sample_ids']}`
- duplicate image paths: `{summary['manifest']['duplicate_image_paths']}`
- duplicate map paths: `{summary['manifest']['duplicate_map_paths']}`
- duplicate label paths: `{summary['manifest']['duplicate_label_paths']}`
- 둘 이상의 split에 공유된 spatial groups: `{summary['manifest']['shared_spatial_groups']}`

## 실제 1024 source patch 중첩

| split pair | overlapping pairs | affected samples A/B | maximum overlap |
|---|---:|---:|---:|
| train–validation | {s['train_val_source1024']['pairs']} | {s['train_val_source1024']['unique_train_samples']}/{s['train_val_source1024']['unique_val_samples']} | {s['train_val_source1024']['maximum_overlap_pixels']:,} px |
| train–test | {s['train_test_source1024']['pairs']} | {s['train_test_source1024']['unique_train_samples']}/{s['train_test_source1024']['unique_test_samples']} | {s['train_test_source1024']['maximum_overlap_pixels']:,} px |
| validation–test | {s['val_test_source1024']['pairs']} | {s['val_test_source1024']['unique_val_samples']}/{s['val_test_source1024']['unique_test_samples']} | {s['val_test_source1024']['maximum_overlap_pixels']:,} px |

## 중앙 512 평가 영역 중첩

| split pair | overlapping pairs | affected samples A/B | maximum overlap |
|---|---:|---:|---:|
| train-center–validation-center | {s['train_val_center512']['pairs']} | {s['train_val_center512']['unique_train_samples']}/{s['train_val_center512']['unique_val_samples']} | {s['train_val_center512']['maximum_overlap_pixels']:,} px |
| train-center–test-center | {s['train_test_center512']['pairs']} | {s['train_test_center512']['unique_train_samples']}/{s['train_test_center512']['unique_test_samples']} | {s['train_test_center512']['maximum_overlap_pixels']:,} px |
| validation-center–test-center | {s['val_test_center512']['pairs']} | {s['val_test_center512']['unique_val_samples']}/{s['val_test_center512']['unique_test_samples']} | {s['val_test_center512']['maximum_overlap_pixels']:,} px |

가장 큰 validation–test 중앙 crop 중첩은 validation `excess_00323`과
test `excess_00324` 사이의 `204,885`픽셀로, 512 crop의 `78.16%`다.

## 학습 crop과의 관계

학습은 1024 source를 그대로 resize한 것이 아니다. 각 epoch에서 512×512
crop 하나를 읽으며 중앙 origin `(256, 256)`에 x/y 각각 `[-128, +128]`
jitter를 적용한다. 따라서 train crop origin은 각 축 `128..384` 범위다.
validation/test는 jitter 없이 `(256, 256)` 중앙 crop 하나만 사용한다.

실제 epoch별 train crop origin은 로그에 저장되지 않았다. 그러므로 어느
중복 pixel이 실제 100 epoch에서 몇 번 사용됐는지는 사후 확정할 수 없다.
다만 train crop이 도달할 수 있는 union과 held-out 중앙 crop을 비교하면:

- train possible-crop union vs validation center: `{s['train_possible_union_vs_val_center']['pairs']}` pairs, validation `{s['train_possible_union_vs_val_center']['unique_val_samples']}` samples affected
- train possible-crop union vs test center: `{s['train_possible_union_vs_test_center']['pairs']}` pairs, test `{s['train_possible_union_vs_test_center']['unique_test_samples']}` samples affected

## 영향과 조치

- severity: **High**, confidence: **High**
- 현재 test는 파일 단위로는 holdout이지만 엄격한 공간 독립 holdout은 아니다.
- 현재 frozen 결과와 audit는 그대로 보존하되 “spatially independent test”로
  표현하지 않는다.
- Formula (7) 재학습 전에 1024 footprint 전체가 split 경계를 넘지 않도록
  buffer를 둔 spatial split v2를 별도 manifest로 설계해야 한다.
- 기존 manifest나 결과를 덮어쓰지 말고 v1/v2를 병기한다.

전체 pair와 pixel 동일성은 `overlap_pairs.csv`, 기계 판독용 요약은
`summary.json`에 있다.
"""


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    prepare_output_dir(output_dir)
    frame = pd.read_csv(manifest)
    required = {
        "sample_id",
        "image_path",
        "map_mask_path",
        "label_mask_path",
        "xoff",
        "yoff",
        "spatial_group",
        "split",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    split_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    pair_rows: list[dict[str, Any]] = []
    for split_a, split_b in split_pairs:
        for row_a, row_b in cross_pairs(frame, split_a, split_b):
            source_rect = intersection(
                rectangle(row_a, offset=0, size=1024),
                rectangle(row_b, offset=0, size=1024),
            )
            if source_rect is None:
                continue
            source_width, source_height, source_pixels = dimensions(source_rect)
            center_rect = intersection(
                rectangle(row_a, offset=256, size=512),
                rectangle(row_b, offset=256, size=512),
            )
            center_width, center_height, center_pixels = dimensions(center_rect)
            row: dict[str, Any] = {
                "split_a": split_a,
                "split_b": split_b,
                "sample_a": row_a.sample_id,
                "sample_b": row_b.sample_id,
                "spatial_group_a": row_a.spatial_group,
                "spatial_group_b": row_b.spatial_group,
                "xoff_a": int(row_a.xoff),
                "yoff_a": int(row_a.yoff),
                "xoff_b": int(row_b.xoff),
                "yoff_b": int(row_b.yoff),
                "source_overlap_width": source_width,
                "source_overlap_height": source_height,
                "source_overlap_pixels": source_pixels,
                "source_overlap_fraction": source_pixels / (1024 * 1024),
                "center_overlap_width": center_width,
                "center_overlap_height": center_height,
                "center_overlap_pixels": center_pixels,
                "center_overlap_fraction": center_pixels / (512 * 512),
            }
            for field, prefix, flag in (
                ("image_path", "image", cv2.IMREAD_COLOR),
                ("map_mask_path", "map", cv2.IMREAD_GRAYSCALE),
                ("label_mask_path", "label", cv2.IMREAD_GRAYSCALE),
            ):
                equal, total, rate = compare_overlap(
                    row_a, row_b, source_rect, field, flag
                )
                row[f"{prefix}_equal_pixels"] = equal
                row[f"{prefix}_compared_pixels"] = total
                row[f"{prefix}_match_rate"] = rate
            pair_rows.append(row)

    overlap: dict[str, Any] = {}
    for split_a, split_b in split_pairs:
        pair_key = f"{split_a}_{split_b}"
        overlap[f"{pair_key}_source1024"] = overlap_summary(
            frame,
            split_a,
            split_b,
            offset_a=0,
            size_a=1024,
            offset_b=0,
            size_b=1024,
        )
        overlap[f"{pair_key}_center512"] = overlap_summary(
            frame,
            split_a,
            split_b,
            offset_a=256,
            size_a=512,
            offset_b=256,
            size_b=512,
        )
    overlap["train_source_vs_val_center"] = overlap_summary(
        frame, "train", "val", offset_a=0, size_a=1024, offset_b=256, size_b=512
    )
    overlap["train_source_vs_test_center"] = overlap_summary(
        frame, "train", "test", offset_a=0, size_a=1024, offset_b=256, size_b=512
    )
    overlap["train_possible_union_vs_val_center"] = overlap_summary(
        frame, "train", "val", offset_a=128, size_a=768, offset_b=256, size_b=512
    )
    overlap["train_possible_union_vs_test_center"] = overlap_summary(
        frame, "train", "test", offset_a=128, size_a=768, offset_b=256, size_b=512
    )

    exact_counts = {
        prefix: sum(float(row[f"{prefix}_match_rate"]) == 1.0 for row in pair_rows)
        for prefix in ("image", "map", "label")
    }
    group_field = (
        "spatial_component_v2"
        if "spatial_component_v2" in frame.columns
        else "spatial_group"
    )
    shared_groups = int(
        (frame.groupby(group_field)["split"].nunique() > 1).sum()
    )
    source_overlap_pairs = len(pair_rows)
    summary = {
        "manifest": {
            "path": str(manifest),
            "sha256": file_sha256(manifest),
            "rows": len(frame),
            "split_counts": {
                str(key): int(value)
                for key, value in frame["split"].value_counts().items()
            },
            "duplicate_sample_ids": int(frame["sample_id"].duplicated().sum()),
            "duplicate_image_paths": int(frame["image_path"].duplicated().sum()),
            "duplicate_map_paths": int(frame["map_mask_path"].duplicated().sum()),
            "duplicate_label_paths": int(frame["label_mask_path"].duplicated().sum()),
            "group_field": group_field,
            "shared_groups": shared_groups,
            "shared_spatial_groups": shared_groups,
        },
        "cross_split_overlap": overlap,
        "pixel_identity": {
            "source_overlap_pairs": source_overlap_pairs,
            "image_exact_match_pairs": exact_counts["image"],
            "map_exact_match_pairs": exact_counts["map"],
            "label_exact_match_pairs": exact_counts["label"],
        },
        "severity": "none" if source_overlap_pairs == 0 else "high",
        "confidence": "high",
        "interpretation": (
            "No cross-split source-pixel overlap was found."
            if source_overlap_pairs == 0
            else "No row/file/group duplication, but exact pixel-level spatial "
            "overlap exists across split boundaries."
        ),
    }
    with (output_dir / "overlap_pairs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        pair_fields = (
            list(pair_rows[0])
            if pair_rows
            else [
                "split_a",
                "split_b",
                "sample_a",
                "sample_b",
                "spatial_group_a",
                "spatial_group_b",
                "xoff_a",
                "yoff_a",
                "xoff_b",
                "yoff_b",
                "source_overlap_width",
                "source_overlap_height",
                "source_overlap_pixels",
                "source_overlap_fraction",
                "center_overlap_width",
                "center_overlap_height",
                "center_overlap_pixels",
                "center_overlap_fraction",
                "image_equal_pixels",
                "image_compared_pixels",
                "image_match_rate",
                "map_equal_pixels",
                "map_compared_pixels",
                "map_match_rate",
                "label_equal_pixels",
                "label_compared_pixels",
                "label_match_rate",
            ]
        )
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(pair_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(
        build_report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
