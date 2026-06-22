#!/usr/bin/env python3
"""Create controlled probability maps for candidate vectorization smoke tests."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.affinity import translate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-index", required=True)
    parser.add_argument("--old-buildings", required=True)
    parser.add_argument("--old-id-dir", required=True)
    parser.add_argument("--reference-changes", default=None)
    parser.add_argument("--out-dir", default="../results/dev_synthetic/res-korea-controlled")
    parser.add_argument("--id-col", default="BLDG_ID")
    return parser.parse_args()


def write_raster(path: Path, array: np.ndarray, profile: dict[str, Any], dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=dtype, compress="deflate", nodata=None)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array.astype(dtype), 1)


def main() -> int:
    args = parse_args()
    tile_index_path = Path(args.tile_index).resolve()
    old_id_dir = Path(args.old_id_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    for subdir in ("prob/new", "prob/removed", "prob/building", "mask/new", "mask/removed_raw", "mask/building"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    tile_index = gpd.read_file(tile_index_path)
    old_buildings = gpd.read_file(args.old_buildings)
    if tile_index.crs and old_buildings.crs != tile_index.crs:
        old_buildings = old_buildings.to_crs(tile_index.crs)
    if args.id_col not in old_buildings.columns:
        raise ValueError(f"old buildings missing id column {args.id_col}")

    ref = None
    if args.reference_changes and Path(args.reference_changes).exists():
        ref = gpd.read_file(args.reference_changes)
        if tile_index.crs and ref.crs != tile_index.crs:
            ref = ref.to_crs(tile_index.crs)

    if ref is not None and "change_type" in ref.columns:
        missing_geoms = list(ref[ref["change_type"].astype(str).str.contains("new", case=False, na=False)].geometry)
        excess_ids = sorted(
            {
                int(value)
                for value in ref[ref["change_type"].astype(str).str.contains("removed", case=False, na=False)][args.id_col].dropna().tolist()
            }
        )
    else:
        missing_geoms = []
        excess_ids = []

    if not missing_geoms:
        bounds = old_buildings.total_bounds
        width = (bounds[2] - bounds[0]) * 0.12
        missing_geoms = [translate(old_buildings.geometry.iloc[0], xoff=width * 3, yoff=-width * 2)]
    if not excess_ids:
        excess_ids = [int(old_buildings[args.id_col].iloc[0])]

    written_tiles = []
    for _, row in tile_index.iterrows():
        tile_id = str(row["tile_id"])
        old_id_path = old_id_dir / f"{tile_id}.tif"
        with rasterio.open(old_id_path) as src:
            old_ids = src.read(1)
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            out_shape = (src.height, src.width)

        prob_new = np.full(out_shape, 0.05, dtype=np.float32)
        prob_removed = np.full(out_shape, 0.05, dtype=np.float32)
        prob_building = np.where(old_ids > 0, 0.80, 0.05).astype(np.float32)

        missing_mask = features.rasterize(
            [(geom, 1) for geom in missing_geoms if geom is not None and not geom.is_empty],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        )
        prob_new[missing_mask > 0] = 0.95
        prob_building[missing_mask > 0] = 0.85
        for old_id in excess_ids:
            prob_removed[old_ids == old_id] = 0.95

        write_raster(out_dir / "prob" / "new" / f"{tile_id}.tif", prob_new, profile, "float32")
        write_raster(out_dir / "prob" / "removed" / f"{tile_id}.tif", prob_removed, profile, "float32")
        write_raster(out_dir / "prob" / "building" / f"{tile_id}.tif", prob_building, profile, "float32")
        write_raster(out_dir / "mask" / "new" / f"{tile_id}.tif", (prob_new >= 0.5).astype(np.uint8) * 255, profile, "uint8")
        write_raster(out_dir / "mask" / "removed_raw" / f"{tile_id}.tif", (prob_removed >= 0.5).astype(np.uint8) * 255, profile, "uint8")
        write_raster(out_dir / "mask" / "building" / f"{tile_id}.tif", (prob_building >= 0.5).astype(np.uint8) * 255, profile, "uint8")
        written_tiles.append(tile_id)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "tile_index": str(tile_index_path),
        "old_buildings": str(Path(args.old_buildings).resolve()),
        "old_id_dir": str(old_id_dir),
        "reference_changes": str(Path(args.reference_changes).resolve()) if args.reference_changes else None,
        "out_dir": str(out_dir),
        "controlled_missing_count": len(missing_geoms),
        "controlled_excess_old_ids": [str(v) for v in excess_ids],
        "tile_count": len(written_tiles),
        "written_tiles": written_tiles,
        "probability_values": {"candidate": 0.95, "background": 0.05, "old_building_reference": 0.80},
        "crs": tile_index.crs.to_string() if tile_index.crs else None,
        "real_data_processed": False,
    }
    summary_path = out_dir / "controlled_probability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote controlled probability summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
