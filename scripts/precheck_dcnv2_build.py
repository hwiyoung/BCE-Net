#!/usr/bin/env python3
"""Capture pre-build diagnostics for the BCE-Net DCNv2 extension."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DCNV2_DIR = REPO_ROOT / "DCNv2"
RESULTS_DIR = (REPO_ROOT / "../results").resolve()
OUT_JSON = RESULTS_DIR / "dcnv2_build_precheck.json"


def run(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return {
            "cmd": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"cmd": args, "error": repr(exc)}


def version(name: str, args: list[str]) -> dict[str, Any]:
    path = shutil.which(name)
    return {"path": path, "version": run([path, *args]) if path else None}


def torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        info["version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_version"] = torch.version.cuda
        info["CUDA_HOME_from_torch"] = CUDA_HOME
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        try:
            info["arch_list"] = list(torch.cuda.get_arch_list())
        except Exception as exc:
            info["arch_list_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def listing(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for item in sorted(path.iterdir(), key=lambda p: p.name):
        records.append(
            {
                "name": item.name,
                "path": str(item),
                "is_dir": item.is_dir(),
                "is_file": item.is_file(),
                "size_bytes": item.stat().st_size if item.is_file() else None,
            }
        )
    return records


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    so_files = [str(path) for path in DCNV2_DIR.glob("*.so")]
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"executable": sys.executable, "version": sys.version},
        "torch": torch_info(),
        "environment": {
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        },
        "tools": {
            "nvcc": version("nvcc", ["--version"]),
            "gcc": version("gcc", ["--version"]),
            "g++": version("g++", ["--version"]),
            "make": version("make", ["--version"]),
        },
        "dcnv2_dir": str(DCNV2_DIR),
        "dcnv2_listing": listing(DCNV2_DIR),
        "existing_so_files": so_files,
        "build_dir_exists": (DCNV2_DIR / "build").exists(),
        "egg_info_dirs": [str(path) for path in DCNV2_DIR.glob("*.egg-info")],
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote DCNv2 build precheck JSON: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
