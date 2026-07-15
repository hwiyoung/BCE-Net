#!/usr/bin/env python3
"""Verify the reusable BCE-Net managed-container environment."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "torch",
    "torchvision",
    "rasterio",
    "geopandas",
    "pyogrio",
    "shapely",
    "fiona",
    "pyproj",
    "scipy",
    "skimage",
    "cv2",
)


def main() -> int:
    report: dict[str, object] = {
        "status": "failed",
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "modules": {},
        "checks": {},
        "failures": [],
    }
    failures: list[str] = report["failures"]  # type: ignore[assignment]
    modules: dict[str, object] = report["modules"]  # type: ignore[assignment]
    checks: dict[str, object] = report["checks"]  # type: ignore[assignment]

    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            modules[name] = {
                "ok": True,
                "version": getattr(module, "__version__", None),
            }
        except Exception as exc:
            modules[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(f"import:{name}")

    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        checks["cuda"] = {
            "ok": cuda_ok,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda_ok else None,
            "capability": list(torch.cuda.get_device_capability(0)) if cuda_ok else None,
        }
        if not cuda_ok:
            failures.append("cuda")
    except Exception as exc:
        checks["cuda"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        failures.append("cuda")

    try:
        import torch
        from DCNv2.dcn_v2 import DCN

        device = torch.device("cuda:0")
        model = DCN(3, 4, kernel_size=3, stride=1, padding=1, deformable_groups=1).to(device).eval()
        x = torch.randn(1, 3, 16, 16, device=device)
        with torch.no_grad():
            y = model(x)
        torch.cuda.synchronize()
        checks["dcnv2"] = {"ok": True, "input_shape": list(x.shape), "output_shape": list(y.shape)}
    except Exception as exc:
        checks["dcnv2"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        failures.append("dcnv2")

    try:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import box

        with tempfile.TemporaryDirectory(prefix="bcenet-env-") as tmp:
            raster_path = Path(tmp) / "smoke.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=8,
                height=8,
                count=1,
                dtype="uint8",
                crs="EPSG:5186",
                transform=from_origin(200000, 600000, 1, 1),
            ) as dst:
                dst.write(np.ones((1, 8, 8), dtype=np.uint8))
            with rasterio.open(raster_path) as src:
                raster_ok = src.read(1).shape == (8, 8)
            vector_ok = bool(gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:5186").geometry.is_valid.all())
        checks["geospatial"] = {"ok": raster_ok and vector_ok}
        if not (raster_ok and vector_ok):
            failures.append("geospatial")
    except Exception as exc:
        checks["geospatial"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        failures.append("geospatial")

    weights = Path(os.environ.get("BCENET_WEIGHTS", "/home/work/models/BCE-Net/checkpoint-best-whu.pth"))
    checks["whu_weights"] = {"ok": weights.is_file(), "path": str(weights)}
    if not weights.is_file():
        failures.append("whu_weights")

    report["status"] = "pass" if not failures else "failed"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
