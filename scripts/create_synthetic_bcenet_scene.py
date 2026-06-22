#!/usr/bin/env python3
"""Create a synthetic ortho/vector scene for BCE-Net Korea PoC smoke tests."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="../results/dev_synthetic/raw")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--crs", default="EPSG:5186")
    parser.add_argument("--pixel-size", type=float, default=0.5)
    parser.add_argument("--id-col", default="BLDG_ID")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pixel_box_to_geom(
    row0: int,
    col0: int,
    row1: int,
    col1: int,
    origin_x: float,
    origin_y: float,
    pixel_size: float,
):
    x0 = origin_x + col0 * pixel_size
    x1 = origin_x + col1 * pixel_size
    y0 = origin_y - row0 * pixel_size
    y1 = origin_y - row1 * pixel_size
    return box(x0, y1, x1, y0)


def paint_rect(
    image: np.ndarray,
    row0: int,
    col0: int,
    row1: int,
    col1: int,
    color: tuple[int, int, int],
    edge: tuple[int, int, int] | None = None,
) -> None:
    image[row0:row1, col0:col1, :] = np.array(color, dtype=np.uint8)
    if edge is not None:
        image[row0 : row0 + 3, col0:col1, :] = edge
        image[row1 - 3 : row1, col0:col1, :] = edge
        image[row0:row1, col0 : col0 + 3, :] = edge
        image[row0:row1, col1 - 3 : col1, :] = edge


def safe_to_file(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> str:
    try:
        gdf.to_file(path, layer=layer, driver="GPKG", engine="pyogrio")
        return "pyogrio"
    except Exception:
        gdf.to_file(path, layer=layer, driver="GPKG")
        return "default"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    origin_x = 200000.0
    origin_y = 600000.0
    transform = from_origin(origin_x, origin_y, args.pixel_size, args.pixel_size)

    base = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    gradient_x = np.linspace(0, 18, args.width, dtype=np.float32)[None, :, None]
    gradient_y = np.linspace(0, 14, args.height, dtype=np.float32)[:, None, None]
    noise = rng.normal(0, 7, size=base.shape)
    base_rgb = np.array([96, 118, 104], dtype=np.float32)
    image = np.clip(base_rgb + gradient_x + gradient_y + noise, 0, 255).astype(np.uint8)

    # Old vector buildings: some stay visible, some are removed-like in current image.
    old_specs = [
        {"id": 101, "kind": "existing_like", "rect": (120, 130, 260, 270), "color": (168, 176, 184)},
        {"id": 102, "kind": "existing_like", "rect": (320, 650, 450, 790), "color": (174, 160, 150)},
        {"id": 103, "kind": "removed_like", "rect": (600, 160, 750, 310), "color": None},
        {"id": 104, "kind": "removed_like", "rect": (720, 700, 850, 850), "color": None},
        {"id": 105, "kind": "existing_like", "rect": (150, 760, 250, 930), "color": (158, 168, 175)},
    ]
    current_new_specs = [
        {"id": 201, "kind": "newly_constructed_like", "rect": (500, 500, 650, 660), "color": (190, 168, 132)},
        {"id": 202, "kind": "newly_constructed_like", "rect": (820, 120, 950, 260), "color": (182, 186, 171)},
    ]

    old_records: list[dict[str, Any]] = []
    ref_records: list[dict[str, Any]] = []

    for spec in old_specs:
        row0, col0, row1, col1 = spec["rect"]
        geom = pixel_box_to_geom(row0, col0, row1, col1, origin_x, origin_y, args.pixel_size)
        old_records.append({args.id_col: spec["id"], "synthetic_role": spec["kind"], "geometry": geom})
        if spec["color"] is not None:
            paint_rect(image, row0, col0, row1, col1, spec["color"], edge=(95, 95, 95))
        else:
            image[row0:row1, col0:col1, :] = np.clip(
                image[row0:row1, col0:col1, :].astype(np.int16) + rng.normal(0, 3, size=(row1 - row0, col1 - col0, 3)),
                0,
                255,
            ).astype(np.uint8)
            ref_records.append(
                {
                    "CHANGE_ID": f"removed_{spec['id']}",
                    "change_type": "removed_like_reference",
                    args.id_col: spec["id"],
                    "geometry": geom,
                }
            )

    for spec in current_new_specs:
        row0, col0, row1, col1 = spec["rect"]
        geom = pixel_box_to_geom(row0, col0, row1, col1, origin_x, origin_y, args.pixel_size)
        paint_rect(image, row0, col0, row1, col1, spec["color"], edge=(92, 86, 74))
        ref_records.append(
            {
                "CHANGE_ID": f"new_{spec['id']}",
                "change_type": "newly_constructed_like_reference",
                args.id_col: spec["id"],
                "geometry": geom,
            }
        )

    # Add simple road-like strips and parcel texture so the image is not a single flat field.
    image[:, 470:486, :] = np.array([128, 128, 122], dtype=np.uint8)
    image[470:486, :, :] = np.array([132, 132, 126], dtype=np.uint8)
    for offset in range(80, args.width, 160):
        image[:, offset : offset + 2, :] = np.clip(image[:, offset : offset + 2, :] + 18, 0, 255)

    ortho_path = out_dir / "synthetic_current_ortho.tif"
    with rasterio.open(
        ortho_path,
        "w",
        driver="GTiff",
        width=args.width,
        height=args.height,
        count=3,
        dtype="uint8",
        crs=args.crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(np.moveaxis(image, -1, 0))

    old_gdf = gpd.GeoDataFrame(old_records, crs=args.crs)
    old_path = out_dir / "synthetic_old_buildings.gpkg"
    old_engine = safe_to_file(old_gdf, old_path, "old_buildings")

    ref_gdf = gpd.GeoDataFrame(ref_records, crs=args.crs)
    ref_path = out_dir / "synthetic_reference_changes.gpkg"
    ref_engine = safe_to_file(ref_gdf, ref_path, "reference_changes")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "note": "Synthetic reference changes are smoke-test references, not performance ground truth.",
        "ortho_path": str(ortho_path),
        "old_buildings_path": str(old_path),
        "reference_changes_path": str(ref_path),
        "old_buildings_layer": "old_buildings",
        "reference_changes_layer": "reference_changes",
        "write_engines": {"old_buildings": old_engine, "reference_changes": ref_engine},
        "width": args.width,
        "height": args.height,
        "crs": args.crs,
        "transform": list(transform)[:6],
        "pixel_size": args.pixel_size,
        "id_col": args.id_col,
        "old_building_count": len(old_records),
        "reference_change_count": len(ref_records),
        "roles": {
            "existing_like": sum(1 for spec in old_specs if spec["kind"] == "existing_like"),
            "removed_like": sum(1 for spec in old_specs if spec["kind"] == "removed_like"),
            "newly_constructed_like": len(current_new_specs),
        },
    }
    manifest_path = out_dir / "synthetic_scene_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote synthetic scene manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
