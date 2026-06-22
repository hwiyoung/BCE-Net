#!/usr/bin/env python3
"""Prepare BCE-Net-style 512x512 tiles from synthetic Korea PoC inputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import array_bounds
from rasterio.windows import Window
from shapely.geometry import box


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
    parser.add_argument("--ortho", required=True)
    parser.add_argument("--buildings", required=True)
    parser.add_argument("--out-dir", default="../results/dev_synthetic/korea_poc")
    parser.add_argument("--id-col", default="BLDG_ID")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--min-valid-ratio", type=float, default=0.1)
    parser.add_argument("--image-format", default="tif", choices=["tif", "tiff"])
    parser.add_argument("--mask-format", default="tif", choices=["tif", "tiff"])
    parser.add_argument("--create-dummy-labels", type=parse_bool, default=True)
    return parser.parse_args()


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length < tile_size:
        raise ValueError(f"image length {length} is smaller than tile size {tile_size}")
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def write_single_band(path: Path, array: np.ndarray, crs: Any, transform: Any, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[1],
        height=array.shape[0],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(array.astype(dtype), 1)


def write_image(path: Path, image: np.ndarray, crs: Any, transform: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=image.shape[2],
        height=image.shape[1],
        count=image.shape[0],
        dtype=image.dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(image)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    image_dir = out_dir / "tiles" / "images"
    footprint_dir = out_dir / "tiles" / "old_footprint"
    old_id_dir = out_dir / "tiles" / "old_id"
    dummy_root = out_dir / "tiles" / "dummy_labels"
    metadata_dir = out_dir / "metadata"
    dataset_dir = out_dir / "dataset"
    for path in (image_dir, footprint_dir, old_id_dir, dummy_root, metadata_dir, dataset_dir):
        path.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.ortho) as src:
        if src.count < 3:
            raise ValueError(f"expected at least 3 bands in ortho, got {src.count}")
        crs = src.crs
        src_bounds = box(*src.bounds)
        gdf = gpd.read_file(args.buildings)
        if args.id_col not in gdf.columns:
            raise ValueError(f"building vector missing id column {args.id_col!r}")
        if gdf.crs is None:
            raise ValueError("building vector CRS is missing")
        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf["geometry"] = gdf.geometry.buffer(0)
        gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()
        gdf = gdf[gdf.intersects(src_bounds)].copy()

        stride = args.tile_size - args.overlap
        if stride <= 0:
            raise ValueError("tile_size must be larger than overlap")
        row_starts = tile_starts(src.height, args.tile_size, stride)
        col_starts = tile_starts(src.width, args.tile_size, stride)

        rows: list[dict[str, Any]] = []
        tile_features: list[dict[str, Any]] = []
        zero = np.zeros((args.tile_size, args.tile_size), dtype=np.uint8)

        for row_off in row_starts:
            for col_off in col_starts:
                window = Window(col_off, row_off, args.tile_size, args.tile_size)
                tile_transform = src.window_transform(window)
                left, bottom, right, top = array_bounds(args.tile_size, args.tile_size, tile_transform)
                geom = box(left, bottom, right, top)
                valid_ratio = 1.0
                if valid_ratio < args.min_valid_ratio:
                    continue
                tile_id = f"tile_r{row_off:04d}_c{col_off:04d}"
                image_path = image_dir / f"{tile_id}.{args.image_format}"
                footprint_path = footprint_dir / f"{tile_id}.{args.mask_format}"
                old_id_path = old_id_dir / f"{tile_id}.{args.mask_format}"
                image = src.read(indexes=[1, 2, 3], window=window)
                write_image(image_path, image, crs, tile_transform)

                tile_gdf = gdf[gdf.intersects(geom)].copy()
                if len(tile_gdf) > 0:
                    footprint_shapes = [(geometry, 1) for geometry in tile_gdf.geometry]
                    id_shapes = [
                        (geometry, int(value))
                        for geometry, value in zip(tile_gdf.geometry, tile_gdf[args.id_col], strict=False)
                    ]
                    old_mask = features.rasterize(
                        footprint_shapes,
                        out_shape=(args.tile_size, args.tile_size),
                        transform=tile_transform,
                        fill=0,
                        dtype="uint8",
                    )
                    old_id = features.rasterize(
                        id_shapes,
                        out_shape=(args.tile_size, args.tile_size),
                        transform=tile_transform,
                        fill=0,
                        dtype="int32",
                    )
                else:
                    old_mask = zero.copy()
                    old_id = np.zeros((args.tile_size, args.tile_size), dtype=np.int32)

                write_single_band(footprint_path, old_mask, crs, tile_transform, "uint8")
                write_single_band(old_id_path, old_id, crs, tile_transform, "int32")

                dummy_paths: dict[str, Path] = {}
                for label_name in ("new", "removed", "building", "change"):
                    label_path = dummy_root / label_name / f"{tile_id}.{args.mask_format}"
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    write_single_band(label_path, zero, crs, tile_transform, "uint8")
                    dummy_paths[label_name] = label_path

                record = {
                    "tile_id": tile_id,
                    "image_path": str(image_path),
                    "old_footprint_path": str(footprint_path),
                    "old_id_path": str(old_id_path),
                    "dummy_new_path": str(dummy_paths["new"]),
                    "dummy_removed_path": str(dummy_paths["removed"]),
                    "dummy_building_path": str(dummy_paths["building"]),
                    "dummy_change_path": str(dummy_paths["change"]),
                    "height": args.tile_size,
                    "width": args.tile_size,
                    "crs": crs.to_string() if crs else None,
                    "row_off": row_off,
                    "col_off": col_off,
                    "valid_ratio": valid_ratio,
                    "old_positive_pixels": int(old_mask.sum()),
                }
                rows.append(record)
                tile_features.append(
                    {
                        **record,
                        "transform": json.dumps(list(tile_transform)[:6]),
                        "geometry": geom,
                    }
                )

    csv_path = dataset_dir / "test_korea.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tile_index = gpd.GeoDataFrame(tile_features, crs=crs)
    tile_index_path = metadata_dir / "tile_index.geojson"
    tile_index.to_file(tile_index_path, driver="GeoJSON")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "ortho": str(Path(args.ortho).resolve()),
        "buildings": str(Path(args.buildings).resolve()),
        "out_dir": str(out_dir),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "stride": stride,
        "generated_tile_count": len(rows),
        "building_feature_count": int(len(gdf)),
        "old_footprint_value_convention": "0/1 uint8 on disk; dataloader returns 0/1 float32 [H,W]",
        "dummy_labels_note": "Dummy label rasters are placeholders for loader compatibility, not metric ground truth.",
        "manifest_csv": str(csv_path),
        "tile_index": str(tile_index_path),
    }
    summary_path = metadata_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote Korea synthetic tile CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
