#!/usr/bin/env python3
"""Inspect geospatial Python package and GDAL/PROJ/GEOS runtime state."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = (REPO_ROOT / "../results").resolve()
DEFAULT_OUT_JSON = RESULTS_DIR / "geospatial_env_inspection.json"

PACKAGES = [
    "numpy",
    "pandas",
    "shapely",
    "fiona",
    "pyproj",
    "rasterio",
    "geopandas",
    "pyogrio",
    "scipy",
    "skimage",
    "cv2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    return parser.parse_args()


def run_command(command: list[str]) -> dict[str, Any]:
    record: dict[str, Any] = {"command": command, "available": bool(shutil.which(command[0]))}
    if not record["available"]:
        return record
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        record.update(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    return record


def package_record(name: str) -> dict[str, Any]:
    record: dict[str, Any] = {"name": name, "ok": False}
    try:
        module = importlib.import_module(name)
        record.update(
            {
                "ok": True,
                "version": getattr(module, "__version__", None),
                "file": getattr(module, "__file__", None),
            }
        )
        if name == "fiona":
            record["gdal_version"] = getattr(module, "__gdal_version__", None)
        elif name == "rasterio":
            record["gdal_version"] = getattr(module, "__gdal_version__", None)
            record["proj_version"] = getattr(module, "__proj_version__", None)
            record["geos_version"] = getattr(module, "__geos_version__", None)
        elif name == "pyogrio":
            record["gdal_version"] = getattr(module, "__gdal_version__", None)
            try:
                record["gdal_version_runtime"] = module.__gdal_version__
            except Exception:
                pass
        elif name == "pyproj":
            record["proj_version"] = getattr(module, "proj_version_str", None)
        elif name == "shapely":
            record["geos_version"] = getattr(module, "geos_version_string", None)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    return record


def main() -> int:
    args = parse_args()
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "sys_path": sys.path,
        "pip": run_command([sys.executable, "-m", "pip", "--version"]),
        "packages": {name: package_record(name) for name in PACKAGES},
        "commands": {
            "gdalinfo": run_command(["gdalinfo", "--version"]),
            "ogrinfo": run_command(["ogrinfo", "--version"]),
        },
    }

    failures = [
        name for name, item in report["packages"].items() if not item.get("ok")
    ]
    report["status"] = "pass" if not failures else "partial"
    report["failed_imports"] = failures

    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote geospatial env inspection JSON: {out_json}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
