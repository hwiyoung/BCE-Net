#!/usr/bin/env python3
"""Create quicklook PNGs for synthetic candidate vector smoke outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ortho", required=True)
    parser.add_argument("--old-buildings", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-previews", type=int, default=5)
    return parser.parse_args()


def polygon_parts(geom) -> Iterable[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        return [part for part in geom.geoms if isinstance(part, Polygon)]
    return []


def geom_to_pixels(geom, inv_transform) -> list[list[tuple[float, float]]]:
    rings = []
    for poly in polygon_parts(geom):
        coords = []
        for x, y in poly.exterior.coords:
            col, row = inv_transform * (x, y)
            coords.append((float(col), float(row)))
        rings.append(coords)
    return rings


def draw_gdf(draw: ImageDraw.ImageDraw, gdf: gpd.GeoDataFrame, inv_transform, outline: tuple[int, int, int], width: int) -> None:
    for geom in gdf.geometry:
        for ring in geom_to_pixels(geom, inv_transform):
            if len(ring) >= 2:
                draw.line(ring + [ring[0]], fill=outline, width=width)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.ortho) as src:
        rgb = np.moveaxis(src.read(indexes=[1, 2, 3]), 0, -1).astype(np.uint8)
        inv_transform = ~src.transform
        crs = src.crs

    old = gpd.read_file(args.old_buildings)
    if crs and old.crs != crs:
        old = old.to_crs(crs)

    candidate_path = Path(args.candidates).resolve()
    layers = fiona.listlayers(candidate_path)
    missing = gpd.read_file(candidate_path, layer="building_missing_candidates") if "building_missing_candidates" in layers else gpd.GeoDataFrame(geometry=[], crs=crs)
    excess = gpd.read_file(candidate_path, layer="building_excess_candidates") if "building_excess_candidates" in layers else gpd.GeoDataFrame(geometry=[], crs=crs)
    if crs and missing.crs != crs:
        missing = missing.to_crs(crs)
    if crs and excess.crs != crs:
        excess = excess.to_crs(crs)

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw_gdf(draw, old, inv_transform, (0, 220, 255), 2)
    draw_gdf(draw, missing, inv_transform, (255, 64, 32), 4)
    draw_gdf(draw, excess, inv_transform, (210, 70, 255), 4)
    overview = out_dir / "candidate_vector_preview_overview.png"
    image.save(overview)

    combined = []
    if not missing.empty:
        combined.append(missing.assign(_preview_type="MISSING"))
    if not excess.empty:
        combined.append(excess.assign(_preview_type="EXCESS"))
    detail_paths = []
    if combined:
        import pandas as pd

        all_candidates = gpd.GeoDataFrame(pd.concat(combined, ignore_index=True), geometry="geometry", crs=crs)
        for idx, row in all_candidates.head(args.max_previews).iterrows():
            minx, miny, maxx, maxy = row.geometry.bounds
            c0, r1 = inv_transform * (minx, miny)
            c1, r0 = inv_transform * (maxx, maxy)
            pad = 60
            left = max(0, int(min(c0, c1)) - pad)
            right = min(image.width, int(max(c0, c1)) + pad)
            top = max(0, int(min(r0, r1)) - pad)
            bottom = min(image.height, int(max(r0, r1)) + pad)
            crop = image.crop((left, top, right, bottom))
            detail = out_dir / f"candidate_preview_{idx + 1:02d}_{row.get('_preview_type', 'candidate')}.png"
            crop.save(detail)
            detail_paths.append(str(detail))

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "ortho": str(Path(args.ortho).resolve()),
        "old_buildings": str(Path(args.old_buildings).resolve()),
        "candidates": str(candidate_path),
        "out_dir": str(out_dir),
        "overview": str(overview),
        "detail_previews": detail_paths,
        "missing_candidate_count": int(len(missing)),
        "excess_candidate_count": int(len(excess)),
        "note": "Preview is a smoke-test quicklook, not a reviewer UI.",
    }
    summary_path = out_dir / "candidate_vector_preview_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote candidate vector preview summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
