#!/usr/bin/env python3
"""Scan DCNv2 source for legacy PyTorch C++/CUDA API usage."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = (REPO_ROOT / "../results").resolve()
JSON_OUT = RESULTS_DIR / "dcnv2_legacy_api_analysis.json"
MD_OUT = RESULTS_DIR / "dcnv2_legacy_api_analysis.md"

TARGETS = [
    REPO_ROOT / "DCNv2/setup.py",
    REPO_ROOT / "DCNv2/dcn_v2.py",
]
TARGETS.extend(sorted((REPO_ROOT / "DCNv2/src").rglob("*")))
TARGET_SUFFIXES = {".cpp", ".cu", ".h", ".cuh"}

PATTERN_CATEGORIES = {
    "TH/TH.h": "removed_legacy_torch_header",
    "THC/THC.h": "removed_legacy_cuda_header",
    "THCState": "removed_legacy_cuda_state",
    "THCuda": "removed_legacy_cuda_macro",
    "AT_CHECK": "deprecated_check_macro",
    "CHECK_CUDA": "custom_check_macro",
    "CHECK_CONTIGUOUS": "custom_check_macro",
    "CHECK_INPUT": "custom_check_macro",
    ".data<": "deprecated_tensor_pointer_api",
    ".data(": "deprecated_tensor_data_api",
    "data<": "deprecated_tensor_pointer_api",
    "data_ptr": "modern_tensor_pointer_api",
    "getCurrentCUDAStream": "cuda_stream_api",
    "cudaStream_t": "cuda_stream_type",
    "extern THCState": "removed_legacy_cuda_state",
    "_ext": "extension_module_name",
    "PYBIND11_MODULE": "extension_binding",
    "torch/extension.h": "modern_extension_header",
    "ATen": "aten_api",
    "c10": "c10_api",
}


def iter_source_files() -> list[Path]:
    files = []
    for target in TARGETS:
        if target.is_file() and (target.name in {"setup.py", "dcn_v2.py"} or target.suffix in TARGET_SUFFIXES):
            files.append(target)
    return sorted(set(files))


def scan_file(path: Path) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern, category in PATTERN_CATEGORIES.items():
            if pattern in line:
                matches.append(
                    {
                        "file": str(path.relative_to(REPO_ROOT)),
                        "line_number": line_no,
                        "pattern": pattern,
                        "risk_category": category,
                        "line_text": line.strip(),
                        "is_comment_only": bool(re.match(r"^\s*(//|#)", line)),
                    }
                )
    return matches


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# DCNv2 Legacy API Analysis",
        "",
        f"Created at UTC: `{report['created_at_utc']}`",
        "",
        "## Summary",
        "",
        f"- Files scanned: `{report['files_scanned_count']}`",
        f"- Matches: `{report['match_count']}`",
        "",
        "## Category Counts",
        "",
        "| Category | Count |",
        "| --- | ---: |",
    ]
    for category, count in sorted(report["category_counts"].items()):
        lines.append(f"| `{category}` | {count} |")
    lines.extend(["", "## Matches", "", "| File | Line | Pattern | Category | Text |", "| --- | ---: | --- | --- | --- |"])
    for match in report["matches"]:
        text = match["line_text"].replace("|", "\\|")
        lines.append(
            f"| `{match['file']}` | {match['line_number']} | `{match['pattern']}` | `{match['risk_category']}` | `{text}` |"
        )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = iter_source_files()
    matches: list[dict[str, Any]] = []
    for path in files:
        matches.extend(scan_file(path))
    category_counts: dict[str, int] = {}
    for match in matches:
        category_counts[match["risk_category"]] = category_counts.get(match["risk_category"], 0) + 1
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "files_scanned": [str(path.relative_to(REPO_ROOT)) for path in files],
        "files_scanned_count": len(files),
        "patterns": PATTERN_CATEGORIES,
        "matches": matches,
        "match_count": len(matches),
        "category_counts": category_counts,
    }
    JSON_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote legacy API analysis JSON: {JSON_OUT}")
    print(f"Wrote legacy API analysis MD: {MD_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
