#!/usr/bin/env python3
"""Check generated DCNv2 build artifacts and Python extension ABI suffix."""

from __future__ import annotations

import json
import argparse
import sysconfig
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DCNV2_DIR = REPO_ROOT / "DCNv2"
RESULTS_DIR = (REPO_ROOT / "../results").resolve()
OUT_JSON = RESULTS_DIR / "dcnv2_build_outputs.json"


def file_record(path: Path, ext_suffix: str | None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "size_bytes": stat.st_size,
        "matches_python_ext_suffix": bool(ext_suffix and path.name.endswith(ext_suffix)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-json",
        default=str(OUT_JSON),
        help="Path to write the build output inspection JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so_files = sorted(DCNV2_DIR.rglob("*.so"))
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dcnv2_dir": str(DCNV2_DIR),
        "python_ext_suffix": ext_suffix,
        "so_files": [file_record(path, ext_suffix) for path in so_files],
        "build_dir_exists": (DCNV2_DIR / "build").exists(),
        "build_dir": str((DCNV2_DIR / "build").resolve()),
        "egg_info_dirs": [str(path.resolve()) for path in DCNV2_DIR.glob("*.egg-info")],
    }
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote DCNv2 build outputs JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
