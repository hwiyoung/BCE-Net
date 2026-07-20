#!/usr/bin/env python3
"""Validate map-ortho PNG triples and create a spatially grouped manifest."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="/home/work/data/change_detection/building/map-ortho",
    )
    parser.add_argument(
        "--output",
        default="dataset/map_ortho_manifest.csv",
    )
    parser.add_argument("--report", default="dataset/map_ortho_audit.json")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--spatial-block-pixels",
        type=int,
        default=8192,
        help="All patch centers in the same source-raster block stay in one split.",
    )
    return parser.parse_args()


def assign_groups(
    frame: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, str]:
    test_fraction = 1.0 - train_fraction - val_fraction
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError("train/val/test fractions must all be positive")
    fractions = {
        "train": train_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in frame["spatial_group"].items():
        grouped[str(group)].append(index)

    rng = random.Random(seed)
    group_items = list(grouped.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    total = {
        "rows": len(frame),
        "class_2": int((frame["center_class_value"] == 2).sum()),
        "class_3": int((frame["center_class_value"] == 3).sum()),
    }
    targets = {
        split: {metric: max(1.0, fraction * value) for metric, value in total.items()}
        for split, fraction in fractions.items()
    }
    current = {
        split: {"rows": 0, "class_2": 0, "class_3": 0} for split in fractions
    }
    assignments: dict[str, str] = {}

    for group, indices in group_items:
        group_values = frame.loc[indices, "center_class_value"]
        contribution = {
            "rows": len(indices),
            "class_2": int((group_values == 2).sum()),
            "class_3": int((group_values == 3).sum()),
        }
        scores: dict[str, float] = {}
        for split in fractions:
            score = 0.0
            for metric, value in contribution.items():
                projected = current[split][metric] + value
                score += (projected / targets[split][metric]) ** 2
            scores[split] = score
        selected = min(scores, key=scores.get)
        assignments[group] = selected
        for metric, value in contribution.items():
            current[selected][metric] += value
    return assignments


def main() -> int:
    args = parse_args()
    root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    patches_path = root / "patches.csv"
    frame = pd.read_csv(patches_path)

    records: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    unique_patterns: Counter[tuple[int, ...]] = Counter()
    pixel_totals: Counter[int] = Counter()
    map_mismatch_pixels = 0
    invalid_value_files = 0

    for row in frame.to_dict(orient="records"):
        sample_id = str(row["patch_id"])
        image = root / str(row["image_png"])
        map_mask = root / str(row["map_mask_png"])
        label_mask = root / str(row["label_mask_png"])
        missing = [str(path) for path in (image, map_mask, label_mask) if not path.is_file()]
        if missing:
            excluded.append({"sample_id": sample_id, "reason": f"missing: {missing}"})
            continue
        image_array = cv2.imread(str(image), cv2.IMREAD_COLOR)
        map_array = cv2.imread(str(map_mask), cv2.IMREAD_GRAYSCALE)
        label_array = cv2.imread(str(label_mask), cv2.IMREAD_GRAYSCALE)
        if image_array is None or map_array is None or label_array is None:
            excluded.append({"sample_id": sample_id, "reason": "OpenCV read failure"})
            continue
        shapes = (image_array.shape[:2], map_array.shape, label_array.shape)
        if len(set(shapes)) != 1:
            excluded.append({"sample_id": sample_id, "reason": f"shape mismatch: {shapes}"})
            continue

        values, counts = np.unique(label_array, return_counts=True)
        pattern = tuple(int(value) for value in values)
        unique_patterns[pattern] += 1
        for value, count in zip(values, counts):
            pixel_totals[int(value)] += int(count)
        if not set(pattern).issubset({0, 1, 2, 3}):
            invalid_value_files += 1

        expected_map = (label_array == 1) | (label_array == 3)
        map_mismatch_pixels += int(((map_array > 0) != expected_map).sum())
        group_x = int(row["xoff"]) // args.spatial_block_pixels
        group_y = int(row["yoff"]) // args.spatial_block_pixels
        records.append(
            {
                "sample_id": sample_id,
                "image_path": str(image),
                "map_mask_path": str(map_mask),
                "label_mask_path": str(label_mask),
                "width": int(image_array.shape[1]),
                "height": int(image_array.shape[0]),
                "center_class": str(row["center_class"]),
                "center_class_value": int(row["center_class_value"]),
                "center_area_m2": float(row["center_area_m2"]),
                "xoff": int(row["xoff"]),
                "yoff": int(row["yoff"]),
                "spatial_group": f"{group_x}_{group_y}",
            }
        )

    manifest = pd.DataFrame.from_records(records)
    if manifest.empty:
        raise RuntimeError(f"No usable triples found below {root}")
    assignments = assign_groups(
        manifest,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    manifest["split"] = manifest["spatial_group"].map(assignments)
    manifest = manifest.sort_values(["split", "sample_id"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)

    split_summary: dict[str, dict[str, int]] = {}
    for split, rows in manifest.groupby("split"):
        split_summary[str(split)] = {
            "samples": len(rows),
            "center_omission": int((rows["center_class_value"] == 2).sum()),
            "center_excess": int((rows["center_class_value"] == 3).sum()),
            "spatial_groups": int(rows["spatial_group"].nunique()),
        }
    report = {
        "data_root": str(root),
        "manifest": str(output),
        "source_rows": len(frame),
        "usable_rows": len(manifest),
        "excluded": excluded,
        "invalid_value_files": invalid_value_files,
        "unique_label_patterns": {
            ",".join(map(str, key)): value for key, value in unique_patterns.items()
        },
        "pixel_totals": {str(key): value for key, value in pixel_totals.items()},
        "map_vs_label_1_or_3_mismatch_pixels": map_mismatch_pixels,
        "spatial_block_pixels": args.spatial_block_pixels,
        "seed": args.seed,
        "splits": split_summary,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
