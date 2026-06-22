#!/usr/bin/env python3
"""Smoke test raster/vector geospatial operations with synthetic data only."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = (REPO_ROOT / "../results").resolve()
DEFAULT_OUT_DIR = RESULTS_DIR / "geospatial_smoke"
DEFAULT_OUT_JSON = RESULTS_DIR / "geospatial_smoke_result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    return parser.parse_args()


def exception_record(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def package_versions() -> dict[str, Any]:
    import fiona
    import geopandas as gpd
    import pyogrio
    import pyproj
    import rasterio
    import shapely

    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rasterio": rasterio.__version__,
        "rasterio_gdal": getattr(rasterio, "__gdal_version__", None),
        "geopandas": gpd.__version__,
        "pyogrio": pyogrio.__version__,
        "pyogrio_gdal": getattr(pyogrio, "__gdal_version__", None),
        "shapely": shapely.__version__,
        "shapely_geos": getattr(shapely, "geos_version_string", None),
        "fiona": fiona.__version__,
        "fiona_gdal": getattr(fiona, "__gdal_version__", None),
        "pyproj": pyproj.__version__,
        "pyproj_proj": getattr(pyproj, "proj_version_str", None),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_json = Path(args.out_json).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "package_versions": {},
        "outputs": {},
        "checks": {},
        "failures": [],
        "status": "failed",
    }

    try:
        import geopandas as gpd
        import rasterio
        from rasterio import features
        from rasterio.transform import from_origin
        from shapely.geometry import Polygon, shape

        report["package_versions"] = package_versions()

        crs = "EPSG:5186"
        width = height = 128
        transform = from_origin(200000.0, 600000.0, 1.0, 1.0)
        raster_path = out_dir / "synthetic_ortho.tif"
        vector_path = out_dir / "synthetic_buildings.gpkg"
        mask_path = out_dir / "synthetic_old_footprint.tif"
        polygonized_path = out_dir / "synthetic_polygonized_mask.gpkg"

        rgb = np.zeros((3, height, width), dtype=np.uint8)
        yy, xx = np.mgrid[0:height, 0:width]
        rgb[0] = (xx * 2).astype(np.uint8)
        rgb[1] = (yy * 2).astype(np.uint8)
        rgb[2] = ((xx + yy) % 256).astype(np.uint8)

        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype=rgb.dtype,
            crs=crs,
            transform=transform,
        ) as dataset:
            dataset.write(rgb)

        with rasterio.open(raster_path) as dataset:
            data = dataset.read()
            report["checks"]["geotiff_read_write"] = {
                "ok": True,
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "crs": str(dataset.crs),
                "transform": list(dataset.transform)[:6],
            }

        polygons = [
            Polygon([(200010, 599990), (200035, 599990), (200035, 599965), (200010, 599965)]),
            Polygon([(200050, 599980), (200075, 599980), (200075, 599950), (200050, 599950)]),
            Polygon([(200085, 599940), (200110, 599940), (200110, 599915), (200085, 599915)]),
        ]
        gdf = gpd.GeoDataFrame(
            {"BLDG_ID": ["B001", "B002", "B003"], "source": ["synthetic"] * 3},
            geometry=polygons,
            crs=crs,
        )

        vector_write_engine = "pyogrio"
        try:
            gdf.to_file(vector_path, driver="GPKG", engine="pyogrio")
        except Exception as exc:
            report["failures"].append(
                {
                    "step": "geopackage_write_pyogrio",
                    "error": exception_record(exc),
                    "fallback": "fiona",
                }
            )
            vector_write_engine = "fiona"
            gdf.to_file(vector_path, driver="GPKG", engine="fiona")

        read_gdf = gpd.read_file(vector_path)
        report["checks"]["geopackage_read_write"] = {
            "ok": True,
            "write_engine": vector_write_engine,
            "feature_count": int(len(read_gdf)),
            "crs": str(read_gdf.crs),
            "all_valid": bool(read_gdf.geometry.is_valid.all()),
            "columns": list(read_gdf.columns),
        }

        shapes = [(geom, 255) for geom in read_gdf.geometry]
        mask = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            fill=0,
            transform=transform,
            dtype="uint8",
        )
        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=mask.dtype,
            crs=crs,
            transform=transform,
        ) as dataset:
            dataset.write(mask, 1)
        positive_pixels = int((mask > 0).sum())
        report["checks"]["rasterize"] = {
            "ok": positive_pixels > 0,
            "mask_shape": list(mask.shape),
            "positive_pixel_count": positive_pixels,
        }

        polygonized_records = []
        for geom, value in features.shapes(mask, mask=mask > 0, transform=transform):
            polygonized_records.append({"value": int(value), "geometry": shape(geom)})
        polygonized_gdf = gpd.GeoDataFrame(
            {"value": [record["value"] for record in polygonized_records]},
            geometry=[record["geometry"] for record in polygonized_records],
            crs=crs,
        )
        polygonized_gdf.to_file(polygonized_path, driver="GPKG", engine="pyogrio")
        polygonized_read = gpd.read_file(polygonized_path)
        report["checks"]["polygonize"] = {
            "ok": len(polygonized_read) > 0,
            "feature_count": int(len(polygonized_read)),
            "crs": str(polygonized_read.crs),
            "all_valid": bool(polygonized_read.geometry.is_valid.all()),
        }

        report["outputs"] = {
            "synthetic_ortho": str(raster_path),
            "synthetic_buildings": str(vector_path),
            "synthetic_old_footprint": str(mask_path),
            "synthetic_polygonized_mask": str(polygonized_path),
        }
        report["checks"]["output_files_exist"] = {
            key: Path(value).exists() for key, value in report["outputs"].items()
        }

        required = [
            report["checks"]["geotiff_read_write"]["ok"],
            report["checks"]["geopackage_read_write"]["ok"],
            report["checks"]["rasterize"]["ok"],
            report["checks"]["polygonize"]["ok"],
            all(report["checks"]["output_files_exist"].values()),
        ]
        report["status"] = "pass" if all(required) else "partial"
    except Exception as exc:
        report["failures"].append({"step": "geospatial_smoke", "error": exception_record(exc)})
        report["status"] = "failed"

    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote geospatial smoke JSON: {out_json}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
