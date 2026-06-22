#!/usr/bin/env python3
"""Inspect the managed BCE-Net development container environment."""

from __future__ import annotations

import getpass
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = (REPO_ROOT / "../results").resolve()
OUT_JSON = RESULTS_DIR / "managed_container_env_inspection.json"


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
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
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        return {"cmd": args, "error": repr(exc)}


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        return f"<error: {exc!r}>"


def tool_info(name: str, version_args: list[str]) -> dict[str, Any]:
    path = shutil.which(name)
    info: dict[str, Any] = {"path": path}
    if path:
        info["version"] = run_command([path, *version_args])
    return info


def import_status(module_name: str) -> dict[str, Any]:
    info: dict[str, Any] = {"ok": False}
    try:
        module = importlib.import_module(module_name)
        info["ok"] = True
        info["version"] = getattr(module, "__version__", None)
        info["file"] = getattr(module, "__file__", None)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import torch

        info["version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_version"] = torch.version.cuda
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


def path_exists(relative_path: str) -> dict[str, Any]:
    path = (REPO_ROOT / relative_path).resolve()
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "is_file": path.is_file(),
    }


def results_write_check() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="write_check_",
            suffix=".tmp",
            dir=RESULTS_DIR,
            delete=False,
        ) as handle:
            handle.write("ok\n")
            temp_path = Path(handle.name)
        temp_path.unlink(missing_ok=True)
        return {"ok": True, "results_dir": str(RESULTS_DIR)}
    except Exception as exc:
        return {"ok": False, "results_dir": str(RESULTS_DIR), "error": repr(exc)}


def main() -> int:
    os_release = read_text(Path("/etc/os-release"))
    proc1_comm = read_text(Path("/proc/1/comm"))
    package_modules = [
        "numpy",
        "pandas",
        "cv2",
        "PIL",
        "torch",
        "rasterio",
        "geopandas",
        "shapely",
        "fiona",
        "pyproj",
        "pyogrio",
        "tqdm",
        "skimage",
        "scipy",
    ]

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": os.getcwd(),
        "repo_root": str(REPO_ROOT),
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "os_release": os_release,
        "container": {
            "dockerenv_exists": Path("/.dockerenv").exists(),
            "proc1_comm": proc1_comm,
        },
        "git": {
            "commit": run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
            "status_short": run_command(["git", "status", "--short"], cwd=REPO_ROOT),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
        },
        "torch": torch_info(),
        "tools": {
            "nvcc": tool_info("nvcc", ["--version"]),
            "gcc": tool_info("gcc", ["--version"]),
            "g++": tool_info("g++", ["--version"]),
            "make": tool_info("make", ["--version"]),
        },
        "packages": {name: import_status(name) for name in package_modules},
        "bcenet_paths": {
            "DCNv2/": path_exists("DCNv2"),
            "DCNv2/setup.py": path_exists("DCNv2/setup.py"),
            "DCNv2/dcn_v2.py": path_exists("DCNv2/dcn_v2.py"),
            "Testmodel/CDResWHU.py": path_exists("Testmodel/CDResWHU.py"),
            "Testmodel/CDResSIBU.py": path_exists("Testmodel/CDResSIBU.py"),
            "dataset/cd_dataload_512.py": path_exists("dataset/cd_dataload_512.py"),
            "test_model_whu.py": path_exists("test_model_whu.py"),
            "test_model_sibu.py": path_exists("test_model_sibu.py"),
        },
        "models_dir": {
            "path": str((REPO_ROOT / "../models/BCE-Net").resolve()),
            "exists": (REPO_ROOT / "../models/BCE-Net").resolve().exists(),
        },
        "results_write_check": results_write_check(),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote environment inspection JSON: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
