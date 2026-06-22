#!/usr/bin/env python3
"""Vectorize BCE-Net building missing/excess candidate evidence.

This script creates reviewer-facing candidate layers only. It does not mark any
candidate as a confirmed error and does not evaluate model accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape
from shapely.ops import unary_union

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - fallback only
    ndimage = None


MISSING_LAYER = "building_missing_candidates"
EXCESS_LAYER = "building_excess_candidates"
REQUIRED_COLUMNS = [
    "candidate_id",
    "candidate_type",
    "model_output",
    "old_building_id",
    "confidence_mean",
    "confidence_p90",
    "confidence_max",
    "area_m2",
    "source_tile",
    "review_status",
    "review_comment",
    "is_synthetic",
]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-index", required=True)
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--old-buildings", required=True)
    parser.add_argument("--old-id-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold-new", type=float, default=0.5)
    parser.add_argument("--threshold-removed", type=float, default=0.5)
    parser.add_argument("--threshold-stat", default="p90", choices=["mean", "p90", "max"])
    parser.add_argument("--min-area-m2", type=float, default=10.0)
    parser.add_argument("--merge-missing", type=parse_bool, default=True)
    parser.add_argument("--allow-empty", type=parse_bool, default=True)
    parser.add_argument("--out-summary", default=None)
    return parser.parse_args()


def detect_id_col(gdf: gpd.GeoDataFrame) -> str:
    for candidate in ("BLDG_ID", "building_id", "old_building_id", "id", "ID"):
        if candidate in gdf.columns:
            return candidate
    raise ValueError(f"Could not detect old building ID column from columns: {list(gdf.columns)}")


def stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def empty_candidates(crs: Any) -> gpd.GeoDataFrame:
    data = {column: [] for column in REQUIRED_COLUMNS}
    return gpd.GeoDataFrame(data, geometry=[], crs=crs)


def write_layer(gdf: gpd.GeoDataFrame, out_path: Path, layer: str, warnings: list[str]) -> None:
    if gdf.empty:
        # Keep a stable empty schema. Some GPKG stacks dislike all-null dtypes,
        # so use explicit typed empty columns before writing.
        gdf = empty_candidates(gdf.crs)
    try:
        gdf.to_file(out_path, layer=layer, driver="GPKG", engine="pyogrio")
    except Exception as exc:
        warnings.append(f"pyogrio write failed for {layer}: {type(exc).__name__}: {exc}; retrying default engine")
        gdf.to_file(out_path, layer=layer, driver="GPKG")


def component_masks(binary: np.ndarray) -> list[np.ndarray]:
    if not binary.any():
        return []
    if ndimage is None:
        return [binary.astype(bool)]
    labeled, count = ndimage.label(binary.astype(np.uint8))
    return [(labeled == idx) for idx in range(1, count + 1)]


def polygonize_component(mask: np.ndarray, transform: Any) -> list[Any]:
    polygons = []
    for geom, value in features.shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform):
        if int(value) == 1:
            polygons.append(shape(geom).buffer(0))
    return [geom for geom in polygons if not geom.is_empty and geom.is_valid]


def vectorize_missing(tile_ids: list[str], inference_dir: Path, threshold: float, min_area: float, crs: Any) -> gpd.GeoDataFrame:
    records: list[dict[str, Any]] = []
    for tile_id in tile_ids:
        prob_path = inference_dir / "prob" / "new" / f"{tile_id}.tif"
        if not prob_path.exists():
            raise FileNotFoundError(f"Missing new probability GeoTIFF: {prob_path}")
        with rasterio.open(prob_path) as src:
            prob = src.read(1).astype(np.float32)
            transform = src.transform
            tile_crs = src.crs
        if crs is None:
            crs = tile_crs
        binary = prob >= threshold
        for component in component_masks(binary):
            values = prob[component]
            for geom in polygonize_component(component, transform):
                area = float(geom.area)
                if area < min_area:
                    continue
                score = stats(values)
                records.append(
                    {
                        "candidate_type": "MISSING",
                        "model_output": "newly_constructed",
                        "old_building_id": None,
                        "confidence_mean": score["mean"],
                        "confidence_p90": score["p90"],
                        "confidence_max": score["max"],
                        "area_m2": area,
                        "source_tile": tile_id,
                        "review_status": "UNREVIEWED",
                        "review_comment": None,
                        "is_synthetic": True,
                        "geometry": geom,
                    }
                )
    if not records:
        return empty_candidates(crs)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def merge_missing_candidates(gdf: gpd.GeoDataFrame, crs: Any) -> tuple[gpd.GeoDataFrame, str]:
    if gdf.empty:
        return empty_candidates(crs), "empty"
    unioned = unary_union(list(gdf.geometry))
    if unioned.is_empty:
        return empty_candidates(crs), "unary_union_empty"
    geometries = list(unioned.geoms) if hasattr(unioned, "geoms") else [unioned]
    records = []
    for idx, geom in enumerate(geometries, start=1):
        intersecting = gdf[gdf.intersects(geom)]
        if intersecting.empty:
            continue
        records.append(
            {
                "candidate_id": f"MISSING_{idx:05d}",
                "candidate_type": "MISSING",
                "model_output": "newly_constructed",
                "old_building_id": None,
                "confidence_mean": float(intersecting["confidence_mean"].max()),
                "confidence_p90": float(intersecting["confidence_p90"].max()),
                "confidence_max": float(intersecting["confidence_max"].max()),
                "area_m2": float(geom.area),
                "source_tile": ";".join(sorted(set(intersecting["source_tile"].astype(str)))),
                "review_status": "UNREVIEWED",
                "review_comment": None,
                "is_synthetic": True,
                "geometry": geom.buffer(0),
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs), "unary_union_max_confidence"


def vectorize_excess(
    tile_ids: list[str],
    inference_dir: Path,
    old_id_dir: Path,
    old_buildings: gpd.GeoDataFrame,
    id_col: str,
    threshold: float,
    threshold_stat: str,
    crs: Any,
) -> gpd.GeoDataFrame:
    per_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for tile_id in tile_ids:
        prob_path = inference_dir / "prob" / "removed" / f"{tile_id}.tif"
        id_path = old_id_dir / f"{tile_id}.tif"
        if not prob_path.exists():
            raise FileNotFoundError(f"Missing removed probability GeoTIFF: {prob_path}")
        if not id_path.exists():
            raise FileNotFoundError(f"Missing old ID raster: {id_path}")
        with rasterio.open(prob_path) as src:
            prob = src.read(1).astype(np.float32)
        with rasterio.open(id_path) as src:
            old_ids = src.read(1)
        if prob.shape != old_ids.shape:
            raise ValueError(f"Shape mismatch for {tile_id}: prob={prob.shape}, old_id={old_ids.shape}")
        for old_id in sorted(int(v) for v in np.unique(old_ids) if int(v) > 0):
            mask = old_ids == old_id
            values = prob[mask]
            score = stats(values)
            selected = score[threshold_stat]
            if selected >= threshold:
                per_id[old_id].append(
                    {
                        "tile_id": tile_id,
                        "pixel_count": int(mask.sum()),
                        **score,
                    }
                )

    records = []
    for idx, (old_id, items) in enumerate(sorted(per_id.items()), start=1):
        geom_rows = old_buildings[old_buildings[id_col].astype(str) == str(old_id)]
        if geom_rows.empty:
            continue
        geom = geom_rows.geometry.iloc[0].buffer(0)
        # Explicit aggregation choice: tile-level means are combined by max so
        # overlapping tiles preserve the strongest reviewer-facing evidence.
        records.append(
            {
                "candidate_id": f"EXCESS_{idx:05d}",
                "candidate_type": "EXCESS",
                "model_output": "removed",
                "old_building_id": str(old_id),
                "confidence_mean": float(max(item["mean"] for item in items)),
                "confidence_p90": float(max(item["p90"] for item in items)),
                "confidence_max": float(max(item["max"] for item in items)),
                "area_m2": float(geom.area),
                "source_tile": ";".join(sorted(set(item["tile_id"] for item in items))),
                "review_status": "UNREVIEWED",
                "review_comment": None,
                "is_synthetic": True,
                "geometry": geom,
            }
        )
    if not records:
        return empty_candidates(crs)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def normalize_columns(gdf: gpd.GeoDataFrame, crs: Any) -> gpd.GeoDataFrame:
    if gdf.empty:
        return empty_candidates(crs)
    for column in REQUIRED_COLUMNS:
        if column not in gdf.columns:
            gdf[column] = None
    return gdf[REQUIRED_COLUMNS + ["geometry"]].copy()


def main() -> int:
    args = parse_args()
    tile_index_path = Path(args.tile_index).resolve()
    inference_dir = Path(args.inference_dir).resolve()
    old_id_dir = Path(args.old_id_dir).resolve()
    old_buildings_path = Path(args.old_buildings).resolve()
    out_path = Path(args.out).resolve()
    out_summary = Path(args.out_summary).resolve() if args.out_summary else out_path.parent / "vectorization_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    failures: list[str] = []
    tile_index = gpd.read_file(tile_index_path)
    if "tile_id" not in tile_index.columns:
        raise ValueError("tile index is missing tile_id column")
    tile_ids = [str(v) for v in tile_index["tile_id"].tolist()]
    crs = tile_index.crs

    old_buildings = gpd.read_file(old_buildings_path)
    id_col = detect_id_col(old_buildings)
    if old_buildings.crs != crs:
        old_buildings = old_buildings.to_crs(crs)
    old_buildings["geometry"] = old_buildings.geometry.buffer(0)

    missing_raw = vectorize_missing(tile_ids, inference_dir, args.threshold_new, args.min_area_m2, crs)
    if args.merge_missing:
        missing, missing_merge_method = merge_missing_candidates(missing_raw, crs)
    else:
        missing = missing_raw.copy()
        missing["candidate_id"] = [f"MISSING_{idx:05d}" for idx in range(1, len(missing) + 1)]
        missing_merge_method = "none"
    excess = vectorize_excess(
        tile_ids,
        inference_dir,
        old_id_dir,
        old_buildings,
        id_col,
        args.threshold_removed,
        args.threshold_stat,
        crs,
    )
    if not excess.empty:
        excess["candidate_id"] = [f"EXCESS_{idx:05d}" for idx in range(1, len(excess) + 1)]

    missing = normalize_columns(missing, crs)
    excess = normalize_columns(excess, crs)
    if not args.allow_empty and (missing.empty or excess.empty):
        failures.append("empty candidate layer while allow-empty=false")

    if out_path.exists():
        out_path.unlink()
    write_layer(missing, out_path, MISSING_LAYER, warnings)
    write_layer(excess, out_path, EXCESS_LAYER, warnings)

    try:
        layers = fiona.listlayers(out_path)
    except Exception as exc:
        layers = []
        failures.append(f"could not list output layers: {type(exc).__name__}: {exc}")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "blocked",
        "input_paths": {
            "tile_index": str(tile_index_path),
            "inference_dir": str(inference_dir),
            "old_buildings": str(old_buildings_path),
            "old_id_dir": str(old_id_dir),
        },
        "threshold_new": args.threshold_new,
        "threshold_removed": args.threshold_removed,
        "threshold_stat": args.threshold_stat,
        "min_area_m2": args.min_area_m2,
        "merge_missing": args.merge_missing,
        "missing_candidate_count": int(len(missing)),
        "excess_candidate_count": int(len(excess)),
        "total_candidate_count": int(len(missing) + len(excess)),
        "empty_layers": {
            MISSING_LAYER: bool(missing.empty),
            EXCESS_LAYER: bool(excess.empty),
        },
        "missing_merge_method": missing_merge_method,
        "excess_duplicate_method": "old_building_id aggregation; confidence_mean=max(tile means), p90=max, max=max",
        "crs": crs.to_string() if crs else None,
        "output_path": str(out_path),
        "layers": list(layers),
        "candidate_schema": REQUIRED_COLUMNS,
        "warnings": warnings,
        "failures": failures,
        "results_are_confirmed_errors": False,
        "real_data_processed": False,
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote candidate vectorization summary: {out_summary}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
