#!/usr/bin/env python3
"""Validate synthetic BCE-Net Korea inference output files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-root", default="../results/dev_synthetic/korea_poc")
    parser.add_argument("--inference-dir", default="../results/dev_synthetic/res-korea")
    parser.add_argument("--out-json", default="../results/dev_synthetic/res-korea/output_check.json")
    return parser.parse_args()


def read_raster_info(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        return {
            "path": str(path),
            "shape": [src.height, src.width],
            "dtype": str(arr.dtype),
            "crs": src.crs.to_string() if src.crs else None,
            "transform": list(src.transform)[:6],
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "has_nan": bool(np.isnan(arr).any()) if np.issubdtype(arr.dtype, np.floating) else False,
            "has_inf": bool(np.isinf(arr).any()) if np.issubdtype(arr.dtype, np.floating) else False,
        }


def main() -> int:
    args = parse_args()
    tile_root = Path(args.tile_root).resolve()
    inference_dir = Path(args.inference_dir).resolve()
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    csv_path = tile_root / "dataset" / "test_korea.csv"
    tile_index = tile_root / "metadata" / "tile_index.geojson"
    summary_path = inference_dir / "summary.json"
    failures: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tile_root": str(tile_root),
        "inference_dir": str(inference_dir),
        "manifest_exists": csv_path.exists(),
        "tile_index_exists": tile_index.exists(),
        "summary_exists": summary_path.exists(),
        "checks": [],
        "status": "blocked",
        "failures": failures,
    }

    if not csv_path.exists():
        failures.append({"path": str(csv_path), "error": "missing dataset CSV"})
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    df = pd.read_csv(csv_path)
    report["tile_count"] = int(len(df))
    expected_subdirs = [
        "prob/building",
        "prob/removed",
        "prob/new",
        "mask/building",
        "mask/removed_raw",
        "mask/new",
        "preview",
    ]
    report["directory_counts"] = {
        subdir: len(list((inference_dir / subdir).glob("*.png" if subdir == "preview" else "*.tif")))
        for subdir in expected_subdirs
    }

    for _, row in df.iterrows():
        tile_id = str(row["tile_id"])
        image_path = Path(row["image_path"])
        with rasterio.open(image_path) as src:
            expected_shape = [src.height, src.width]
            expected_crs = src.crs.to_string() if src.crs else None
            expected_transform = list(src.transform)[:6]
        tile_record = {"tile_id": tile_id, "ok": True, "rasters": {}}
        for kind, rel in {
            "prob_building": f"prob/building/{tile_id}.tif",
            "prob_removed": f"prob/removed/{tile_id}.tif",
            "prob_new": f"prob/new/{tile_id}.tif",
            "mask_building": f"mask/building/{tile_id}.tif",
            "mask_removed_raw": f"mask/removed_raw/{tile_id}.tif",
            "mask_new": f"mask/new/{tile_id}.tif",
        }.items():
            path = inference_dir / rel
            if not path.exists():
                failures.append({"tile_id": tile_id, "path": str(path), "error": "missing output"})
                tile_record["ok"] = False
                continue
            info = read_raster_info(path)
            tile_record["rasters"][kind] = info
            if info["shape"] != expected_shape:
                failures.append({"tile_id": tile_id, "path": str(path), "error": "shape mismatch", "info": info})
                tile_record["ok"] = False
            if info["crs"] != expected_crs or info["transform"] != expected_transform:
                failures.append({"tile_id": tile_id, "path": str(path), "error": "georeference mismatch", "info": info})
                tile_record["ok"] = False
            if kind.startswith("prob") and info["dtype"] != "float32":
                failures.append({"tile_id": tile_id, "path": str(path), "error": "probability dtype is not float32", "info": info})
                tile_record["ok"] = False
            if kind.startswith("prob") and not (0.0 <= info["min"] <= info["max"] <= 1.0):
                failures.append({"tile_id": tile_id, "path": str(path), "error": "probability outside [0,1]", "info": info})
                tile_record["ok"] = False
            if kind.startswith("mask") and info["dtype"] != "uint8":
                failures.append({"tile_id": tile_id, "path": str(path), "error": "mask dtype is not uint8", "info": info})
                tile_record["ok"] = False
        preview = inference_dir / "preview" / f"{tile_id}_quicklook.png"
        if not preview.exists():
            failures.append({"tile_id": tile_id, "path": str(preview), "error": "missing preview"})
            tile_record["ok"] = False
        else:
            with Image.open(preview) as img:
                tile_record["preview"] = {"path": str(preview), "size": list(img.size), "mode": img.mode}
        report["checks"].append(tile_record)

    report["status"] = "pass" if not failures else "partial"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote synthetic output check JSON: {out_json}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
