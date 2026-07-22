#!/usr/bin/env python3
"""Create a leakage-free buffered spatial split for map-ortho patches."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_TARGET_ROWS = {"train": 800, "val": 100, "test": 100}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest", default="dataset/map_ortho_manifest.csv"
    )
    parser.add_argument(
        "--patches-csv",
        default="/home/work/data/change_detection/building/map-ortho/patches.csv",
    )
    parser.add_argument(
        "--output", default="dataset/map_ortho_manifest_spatial_v2.csv"
    )
    parser.add_argument(
        "--report", default="dataset/map_ortho_spatial_split_v2_report.json"
    )
    parser.add_argument(
        "--report-md", default="dataset/map_ortho_spatial_split_v2_report.md"
    )
    parser.add_argument("--buffer-pixels", type=int, default=256)
    parser.add_argument(
        "--candidate-buffers", default="0,128,256,512,1024"
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--source-raster-id",
        default="orthophoto_2020_seocho_gangnam_merged_5186_cog",
    )
    parser.add_argument("--source-gsd-m", type=float, default=0.12)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def build_components(
    frame: pd.DataFrame, buffer_pixels: int
) -> tuple[list[list[int]], dict[str, int]]:
    """Connect overlapping or buffer-near 1024 source rectangles."""

    if buffer_pixels < 0:
        raise ValueError("buffer_pixels must be non-negative")
    x0 = frame["xoff"].to_numpy(dtype=np.int64)
    y0 = frame["yoff"].to_numpy(dtype=np.int64)
    x1 = x0 + frame["width"].to_numpy(dtype=np.int64)
    y1 = y0 + frame["height"].to_numpy(dtype=np.int64)
    dsu = DisjointSet(len(frame))
    overlap_edges = 0
    buffer_only_edges = 0

    for left in range(len(frame) - 1):
        overlap_width = (
            np.minimum(x1[left], x1[left + 1 :])
            - np.maximum(x0[left], x0[left + 1 :])
        )
        overlap_height = (
            np.minimum(y1[left], y1[left + 1 :])
            - np.maximum(y0[left], y0[left + 1 :])
        )
        overlaps = (overlap_width > 0) & (overlap_height > 0)
        related = overlaps.copy()
        if buffer_pixels > 0:
            gap_x = np.maximum(
                0,
                np.maximum(x0[left], x0[left + 1 :])
                - np.minimum(x1[left], x1[left + 1 :]),
            )
            gap_y = np.maximum(
                0,
                np.maximum(y0[left], y0[left + 1 :])
                - np.minimum(y1[left], y1[left + 1 :]),
            )
            buffered = (gap_x < buffer_pixels) & (gap_y < buffer_pixels)
            related |= buffered
            buffer_only_edges += int((buffered & ~overlaps).sum())
        overlap_edges += int(overlaps.sum())
        for right in np.flatnonzero(related) + left + 1:
            dsu.union(left, int(right))

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(frame)):
        grouped[dsu.find(index)].append(index)
    components = sorted(
        grouped.values(),
        key=lambda indices: (
            int(frame.loc[indices, "xoff"].min()),
            int(frame.loc[indices, "yoff"].min()),
            str(frame.loc[indices, "sample_id"].min()),
        ),
    )
    return components, {
        "overlap_edges": overlap_edges,
        "buffer_only_edges": buffer_only_edges,
    }


def component_summary(
    components: list[list[int]], edge_counts: dict[str, int]
) -> dict[str, Any]:
    sizes = sorted((len(indices) for indices in components), reverse=True)
    return {
        "components": len(components),
        **edge_counts,
        "largest_component_samples": sizes[0],
        "largest_component_fraction": sizes[0] / sum(sizes),
        "top_10_component_sizes": sizes[:10],
        "singleton_components": int(sum(size == 1 for size in sizes)),
    }


def load_and_validate(
    source_manifest: Path, patches_csv: Path
) -> pd.DataFrame:
    frame = pd.read_csv(source_manifest).reset_index(drop=True)
    required = {
        "sample_id",
        "image_path",
        "map_mask_path",
        "label_mask_path",
        "width",
        "height",
        "center_class_value",
        "xoff",
        "yoff",
        "spatial_group",
        "split",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")
    if len(frame) != 1000:
        raise ValueError(f"Expected 1000 rows, found {len(frame)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs in source manifest")
    if not bool(((frame["width"] == 1024) & (frame["height"] == 1024)).all()):
        raise ValueError("Every source patch must be 1024x1024")
    for field in ("image_path", "map_mask_path", "label_mask_path"):
        if frame[field].duplicated().any():
            raise ValueError(f"Duplicate path in {field}")
        missing_paths = [value for value in frame[field] if not Path(value).is_file()]
        if missing_paths:
            raise FileNotFoundError(missing_paths[0])

    patches = pd.read_csv(patches_csv, encoding="utf-8-sig")
    patch_fields = [
        "patch_id",
        "center_x",
        "center_y",
        "background_pixels",
        "no_change_pixels",
        "omission_pixels",
        "excess_pixels",
    ]
    missing_patch_fields = set(patch_fields) - set(patches.columns)
    if missing_patch_fields:
        raise ValueError(
            f"patches.csv is missing columns: {sorted(missing_patch_fields)}"
        )
    patches = patches[patch_fields].rename(columns={"patch_id": "sample_id"})
    if patches["sample_id"].duplicated().any():
        raise ValueError("Duplicate patch IDs in patches.csv")
    frame = frame.merge(patches, on="sample_id", how="left", validate="one_to_one")
    if frame["center_x"].isna().any():
        raise ValueError("Some manifest rows lack patches.csv lineage")

    center_counts = {
        "center_existing_pixels": [],
        "center_omission_pixels": [],
        "center_excess_pixels": [],
    }
    for row in frame.itertuples(index=False):
        label = cv2.imread(str(row.label_mask_path), cv2.IMREAD_GRAYSCALE)
        if label is None or label.shape != (1024, 1024):
            raise ValueError(f"Invalid label image: {row.label_mask_path}")
        center = label[256:768, 256:768]
        center_counts["center_existing_pixels"].append(
            int(((center == 1) | (center == 2)).sum())
        )
        center_counts["center_omission_pixels"].append(int((center == 2).sum()))
        center_counts["center_excess_pixels"].append(int((center == 3).sum()))
    for field, values in center_counts.items():
        frame[field] = values
    frame["center_omission_samples"] = (
        frame["center_class_value"].astype(int) == 2
    ).astype(int)

    x_center = frame["xoff"].to_numpy(dtype=float) + 512.0
    y_center = frame["yoff"].to_numpy(dtype=float) + 512.0
    x_edges = np.quantile(x_center, [0.25, 0.5, 0.75])
    y_edges = np.quantile(y_center, [0.25, 0.5, 0.75])
    frame["x_quartile"] = np.digitize(x_center, x_edges, right=False)
    frame["y_quartile"] = np.digitize(y_center, y_edges, right=False)
    return frame


def aggregate_components(
    frame: pd.DataFrame, components: list[list[int]]
) -> tuple[pd.DataFrame, dict[int, str]]:
    metric_fields = [
        "center_omission_samples",
        "center_existing_pixels",
        "center_omission_pixels",
        "center_excess_pixels",
    ]
    records: list[dict[str, Any]] = []
    row_to_component: dict[int, str] = {}
    for number, indices in enumerate(components, start=1):
        component_id = f"spv2_{number:04d}"
        record: dict[str, Any] = {
            "component_id": component_id,
            "rows": len(indices),
        }
        for field in metric_fields:
            record[field] = int(frame.loc[indices, field].sum())
        for axis in ("x", "y"):
            for quartile in range(4):
                record[f"{axis}_quartile_{quartile}"] = int(
                    (frame.loc[indices, f"{axis}_quartile"] == quartile).sum()
                )
        records.append(record)
        for index in indices:
            row_to_component[index] = component_id
    return pd.DataFrame.from_records(records), row_to_component


def assign_components(
    component_frame: pd.DataFrame, seed: int
) -> tuple[dict[str, str], dict[str, Any]]:
    """Use MILP to keep components intact and balance label/spatial marginals."""

    components = list(component_frame["component_id"])
    component_count = len(components)
    split_count = len(SPLITS)
    balance_fields = {
        "center_omission_samples": 4.0,
        "center_existing_pixels": 1.0,
        "center_omission_pixels": 2.0,
        "center_excess_pixels": 2.0,
        **{f"x_quartile_{index}": 0.25 for index in range(4)},
        **{f"y_quartile_{index}": 0.25 for index in range(4)},
    }
    deviations = [
        (split, field)
        for split in SPLITS
        for field in balance_fields
    ]
    assignment_vars = component_count * split_count
    total_vars = assignment_vars + len(deviations)
    objective = np.zeros(total_vars, dtype=float)
    rng = np.random.default_rng(seed)
    objective[:assignment_vars] = rng.uniform(0.0, 1.0e-8, assignment_vars)
    for offset, (_, field) in enumerate(deviations):
        objective[assignment_vars + offset] = balance_fields[field]

    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def assignment_index(component: int, split_index: int) -> int:
        return component * split_count + split_index

    for component in range(component_count):
        rows.append(
            {
                assignment_index(component, split_index): 1.0
                for split_index in range(split_count)
            }
        )
        lower.append(1.0)
        upper.append(1.0)

    for split_index, split in enumerate(SPLITS):
        rows.append(
            {
                assignment_index(component, split_index): float(
                    component_frame.iloc[component]["rows"]
                )
                for component in range(component_count)
            }
        )
        target = float(SPLIT_TARGET_ROWS[split])
        lower.append(target)
        upper.append(target)

    for deviation_offset, (split, field) in enumerate(deviations):
        split_index = SPLITS.index(split)
        total = float(component_frame[field].sum())
        if total <= 0:
            raise ValueError(f"Cannot balance empty metric: {field}")
        deviation_index = assignment_vars + deviation_offset
        normalized = component_frame[field].to_numpy(dtype=float) / total
        positive_row = {
            assignment_index(component, split_index): float(normalized[component])
            for component in range(component_count)
            if normalized[component] != 0.0
        }
        positive_row[deviation_index] = -1.0
        rows.append(positive_row)
        lower.append(-np.inf)
        upper.append(SPLIT_FRACTIONS[split])

        negative_row = {
            assignment_index(component, split_index): -float(normalized[component])
            for component in range(component_count)
            if normalized[component] != 0.0
        }
        negative_row[deviation_index] = -1.0
        rows.append(negative_row)
        lower.append(-np.inf)
        upper.append(-SPLIT_FRACTIONS[split])

    matrix = lil_matrix((len(rows), total_vars), dtype=float)
    for row_index, values in enumerate(rows):
        for column, value in values.items():
            matrix[row_index, column] = value
    variable_lower = np.zeros(total_vars, dtype=float)
    variable_upper = np.full(total_vars, np.inf, dtype=float)
    variable_upper[:assignment_vars] = 1.0
    integrality = np.zeros(total_vars, dtype=int)
    integrality[:assignment_vars] = 1
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 30.0, "mip_rel_gap": 0.01},
    )
    if result.x is None:
        raise RuntimeError(f"Split MILP failed: {result.message}")

    assignments: dict[str, str] = {}
    for component, component_id in enumerate(components):
        values = [
            result.x[assignment_index(component, split_index)]
            for split_index in range(split_count)
        ]
        selected = int(np.argmax(values))
        if values[selected] < 0.5:
            raise RuntimeError(f"Non-integral assignment for {component_id}: {values}")
        assignments[component_id] = SPLITS[selected]
    return assignments, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun) if result.fun is not None else None,
        "mip_gap": (
            float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None
        ),
    }


def cross_split_checks(frame: pd.DataFrame, buffer_pixels: int) -> dict[str, Any]:
    records: dict[str, dict[str, int]] = {}
    overall_overlap = 0
    overall_buffer = 0
    for split_a, split_b in (("train", "val"), ("train", "test"), ("val", "test")):
        left = frame[frame["split"] == split_a]
        right = frame[frame["split"] == split_b]
        overlap_pairs = 0
        buffer_violations = 0
        center_pairs = 0
        possible_train_pairs = 0
        for row_a in left.itertuples(index=False):
            ax0, ay0 = int(row_a.xoff), int(row_a.yoff)
            ax1, ay1 = ax0 + int(row_a.width), ay0 + int(row_a.height)
            for row_b in right.itertuples(index=False):
                bx0, by0 = int(row_b.xoff), int(row_b.yoff)
                bx1, by1 = bx0 + int(row_b.width), by0 + int(row_b.height)
                overlap_width = min(ax1, bx1) - max(ax0, bx0)
                overlap_height = min(ay1, by1) - max(ay0, by0)
                if overlap_width > 0 and overlap_height > 0:
                    overlap_pairs += 1
                gap_x = max(0, max(ax0, bx0) - min(ax1, bx1))
                gap_y = max(0, max(ay0, by0) - min(ay1, by1))
                if gap_x < buffer_pixels and gap_y < buffer_pixels:
                    buffer_violations += 1

                ac = (ax0 + 256, ay0 + 256, ax0 + 768, ay0 + 768)
                bc = (bx0 + 256, by0 + 256, bx0 + 768, by0 + 768)
                if min(ac[2], bc[2]) > max(ac[0], bc[0]) and min(
                    ac[3], bc[3]
                ) > max(ac[1], bc[1]):
                    center_pairs += 1
                if split_a == "train":
                    au = (ax0 + 128, ay0 + 128, ax0 + 896, ay0 + 896)
                    if min(au[2], bc[2]) > max(au[0], bc[0]) and min(
                        au[3], bc[3]
                    ) > max(au[1], bc[1]):
                        possible_train_pairs += 1
        key = f"{split_a}_{split_b}"
        records[key] = {
            "source1024_overlap_pairs": overlap_pairs,
            "buffer_violations": buffer_violations,
            "center512_overlap_pairs": center_pairs,
            "train_possible_union_vs_held_center_pairs": possible_train_pairs,
        }
        overall_overlap += overlap_pairs
        overall_buffer += buffer_violations
    records["totals"] = {
        "source1024_overlap_pairs": overall_overlap,
        "buffer_violations": overall_buffer,
    }
    return records


def split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pixels_per_crop = 512 * 512
    for split in SPLITS:
        rows = frame[frame["split"] == split]
        omission_pixels = int(rows["center_omission_pixels"].sum())
        excess_pixels = int(rows["center_excess_pixels"].sum())
        result[split] = {
            "samples": len(rows),
            "components": int(rows["spatial_component_v2"].nunique()),
            "center_omission_samples": int(
                (rows["center_class_value"] == 2).sum()
            ),
            "center_excess_samples": int(
                (rows["center_class_value"] == 3).sum()
            ),
            "center_omission_target_rate": omission_pixels
            / (len(rows) * pixels_per_crop),
            "center_excess_target_rate": excess_pixels
            / (len(rows) * pixels_per_crop),
            "center_combined_target_rate": (omission_pixels + excess_pixels)
            / (len(rows) * pixels_per_crop),
        }
    return result


def report_markdown(report: dict[str, Any]) -> str:
    splits = report["splits"]
    checks = report["cross_split_checks"]
    candidates = report["candidate_buffers"]
    candidate_rows = "\n".join(
        "| {buffer} | {components} | {largest} | {fraction:.1%} |".format(
            buffer=buffer,
            components=values["components"],
            largest=values["largest_component_samples"],
            fraction=values["largest_component_fraction"],
        )
        for buffer, values in candidates.items()
    )
    split_rows = "\n".join(
        "| {split} | {samples} | {components} | {omission}/{excess} | "
        "{omission_rate:.6f} | {excess_rate:.6f} | {combined_rate:.6f} |".format(
            split=split,
            samples=values["samples"],
            components=values["components"],
            omission=values["center_omission_samples"],
            excess=values["center_excess_samples"],
            omission_rate=values["center_omission_target_rate"],
            excess_rate=values["center_excess_target_rate"],
            combined_rate=values["center_combined_target_rate"],
        )
        for split, values in splits.items()
    )
    check_rows = "\n".join(
        f"| {pair} | {values['source1024_overlap_pairs']} | "
        f"{values['buffer_violations']} | {values.get('center512_overlap_pairs', '-')} | "
        f"{values.get('train_possible_union_vs_held_center_pairs', '-')} |"
        for pair, values in checks.items()
        if pair != "totals"
    )
    return f"""# Map-ortho spatial split v2

## 결론

기존 v1 manifest는 수정하지 않았다. 동일한 source mosaic의 1024 patch가
겹치거나 axis-aligned gap이 `{report['buffer_pixels']}`픽셀 미만이면 같은
spatial component로 묶고, component 전체를 하나의 split에 배정했다.

- source 1024 cross-split overlap: **0 pairs**
- cross-split buffer violations: **0 pairs**
- train possible 512 crop union과 held-out center crop overlap: **0 pairs**
- validation/test center crop overlap: **0 pairs**

## Buffer 후보

| buffer px | components | largest samples | largest fraction |
|---:|---:|---:|---:|
{candidate_rows}

`256px`는 GSD 0.12m 기준 30.72m다. `512px`부터 component가 276장으로
급증하므로, split 구성 가능성과 주변 문맥 분리를 함께 고려해 256px를
선택했다.

## Split 분포

| split | samples | components | center omission/excess | omission rate | excess rate | combined rate |
|---|---:|---:|---:|---:|---:|---:|
{split_rows}

## 독립 검증

| split pair | source overlap | buffer violation | center overlap | train-union vs held-center |
|---|---:|---:|---:|---:|
{check_rows}

`xoff/yoff`는 단일 merged source raster의 공통 원점에 대한 픽셀 offset이다.
원본 COG는 현재 작업공간에 없지만, 기존 audit에서 좌표로 정렬한 88개
교차 영역의 image/map/label 픽셀이 모두 100% 일치했다.
"""


def main() -> int:
    args = parse_args()
    source_manifest = Path(args.source_manifest).resolve()
    patches_csv = Path(args.patches_csv).resolve()
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    report_md_path = Path(args.report_md).resolve()
    for source in (source_manifest, patches_csv):
        if not source.is_file():
            raise FileNotFoundError(source)
    for destination in (output, report_path, report_md_path):
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite: {destination}")

    frame = load_and_validate(source_manifest, patches_csv)
    candidate_buffers = [
        int(value.strip()) for value in args.candidate_buffers.split(",")
    ]
    if args.buffer_pixels not in candidate_buffers:
        candidate_buffers.append(args.buffer_pixels)
    candidates: dict[str, Any] = {}
    chosen_components: list[list[int]] | None = None
    chosen_edges: dict[str, int] | None = None
    for buffer_pixels in sorted(set(candidate_buffers)):
        components, edge_counts = build_components(frame, buffer_pixels)
        candidates[str(buffer_pixels)] = component_summary(components, edge_counts)
        if buffer_pixels == args.buffer_pixels:
            chosen_components = components
            chosen_edges = edge_counts
    assert chosen_components is not None and chosen_edges is not None

    component_frame, row_to_component = aggregate_components(
        frame, chosen_components
    )
    assignments, optimizer_summary = assign_components(component_frame, args.seed)
    frame["split_v1"] = frame["split"]
    frame["spatial_component_v2"] = [
        row_to_component[index] for index in range(len(frame))
    ]
    frame["spatial_buffer_pixels_v2"] = args.buffer_pixels
    frame["source_raster_id"] = args.source_raster_id
    frame["source_gsd_m"] = args.source_gsd_m
    frame["split"] = frame["spatial_component_v2"].map(assignments)
    if frame["split"].isna().any():
        raise RuntimeError("Incomplete component assignment")

    checks = cross_split_checks(frame, args.buffer_pixels)
    if checks["totals"]["source1024_overlap_pairs"] != 0:
        raise RuntimeError("v2 split still has cross-split source overlap")
    if checks["totals"]["buffer_violations"] != 0:
        raise RuntimeError("v2 split still violates the selected buffer")
    if (frame.groupby("spatial_component_v2")["split"].nunique() > 1).any():
        raise RuntimeError("A spatial component was split across partitions")
    actual_counts = frame["split"].value_counts().to_dict()
    if actual_counts != SPLIT_TARGET_ROWS:
        raise RuntimeError(
            f"Unexpected split counts: {actual_counts} != {SPLIT_TARGET_ROWS}"
        )

    output_columns = [
        "sample_id",
        "image_path",
        "map_mask_path",
        "label_mask_path",
        "width",
        "height",
        "center_class",
        "center_class_value",
        "center_area_m2",
        "center_x",
        "center_y",
        "xoff",
        "yoff",
        "source_raster_id",
        "source_gsd_m",
        "spatial_group",
        "split_v1",
        "spatial_component_v2",
        "spatial_buffer_pixels_v2",
        "split",
    ]
    result_frame = frame[output_columns].sort_values(
        ["split", "spatial_component_v2", "sample_id"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output, index=False)

    report = {
        "protocol": "map-ortho-buffered-spatial-split-v2",
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": sha256(source_manifest),
        },
        "patches_csv": {"path": str(patches_csv), "sha256": sha256(patches_csv)},
        "output_manifest": {"path": str(output), "sha256": sha256(output)},
        "seed": args.seed,
        "buffer_pixels": args.buffer_pixels,
        "buffer_meters": args.buffer_pixels * args.source_gsd_m,
        "buffer_definition": (
            "Two 1024 source rectangles are connected when they overlap, or "
            "when both axis-aligned gaps are smaller than buffer_pixels."
        ),
        "candidate_buffers": candidates,
        "selected_components": component_summary(chosen_components, chosen_edges),
        "assignment_optimizer": optimizer_summary,
        "splits": split_summary(frame),
        "cross_split_checks": checks,
        "integrity": {
            "rows": len(frame),
            "duplicate_sample_ids": int(frame["sample_id"].duplicated().sum()),
            "duplicate_image_paths": int(frame["image_path"].duplicated().sum()),
            "duplicate_map_paths": int(frame["map_mask_path"].duplicated().sum()),
            "duplicate_label_paths": int(
                frame["label_mask_path"].duplicated().sum()
            ),
            "components_shared_across_splits": int(
                (frame.groupby("spatial_component_v2")["split"].nunique() > 1).sum()
            ),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_md_path.write_text(report_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
