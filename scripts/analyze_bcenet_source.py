#!/usr/bin/env python3
"""Analyze BCE-Net source usage relevant to DCNv2 and inference smoke tests."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = (REPO_ROOT / "../results").resolve()
JSON_OUT = RESULTS_DIR / "bcenet_source_analysis.json"
MD_OUT = RESULTS_DIR / "bcenet_source_analysis.md"

TARGET_FILES = [
    "test_model_whu.py",
    "test_model_sibu.py",
    "Testmodel/CDResWHU.py",
    "Testmodel/CDResSIBU.py",
    "dataset/cd_dataload_512.py",
    "DCNv2/dcn_v2.py",
    "DCNv2/setup.py",
]

PATTERNS = [
    "Baseline",
    "Baseline34",
    "DataParallel",
    "state_dict",
    "labels_o",
    "predicts_b",
    "predicts_mov",
    "predicts_new",
    "DCN",
    "dcn",
    "DCNv2",
    "_ext",
    "deform",
]


def scan_file(path: Path) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    pattern_map = {pattern: re.compile(re.escape(pattern)) for pattern in PATTERNS}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return [
            {
                "file": str(path.relative_to(REPO_ROOT)),
                "line_number": None,
                "pattern": "<read_error>",
                "line_text": repr(exc),
            }
        ]

    for idx, line in enumerate(lines, start=1):
        for pattern, regex in pattern_map.items():
            if regex.search(line):
                matches.append(
                    {
                        "file": str(path.relative_to(REPO_ROOT)),
                        "line_number": idx,
                        "pattern": pattern,
                        "line_text": line.strip(),
                    }
                )
    return matches


def write_markdown(report: dict[str, Any]) -> None:
    lines: list[str] = [
        "# BCE-Net Source Analysis",
        "",
        f"Created at UTC: `{report['created_at_utc']}`",
        "",
        "## File Existence",
        "",
        "| File | Exists |",
        "| --- | --- |",
    ]
    for item in report["files"]:
        lines.append(f"| `{item['path']}` | `{item['exists']}` |")

    lines.extend(["", "## Pattern Matches", "", "| File | Line | Pattern | Text |", "| --- | ---: | --- | --- |"])
    for match in report["matches"]:
        text = str(match["line_text"]).replace("|", "\\|")
        lines.append(
            f"| `{match['file']}` | {match['line_number']} | `{match['pattern']}` | `{text}` |"
        )

    lines.extend(
        [
            "",
            "## DCNv2 Import Interpretation",
            "",
            "- BCE-Net model files import `DCN` from `DCNv2.dcn_v2`.",
            "- `DCNv2/dcn_v2.py` imports the compiled extension as top-level `_ext`.",
            "- Smoke tests should try `from DCNv2.dcn_v2 import DCN, DCNv2` first, then `from dcn_v2 import DCN, DCNv2`, and then direct `_ext` imports.",
        ]
    )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    matches: list[dict[str, Any]] = []
    for rel in TARGET_FILES:
        path = REPO_ROOT / rel
        files.append({"path": rel, "exists": path.exists(), "is_file": path.is_file()})
        if path.exists() and path.is_file():
            matches.extend(scan_file(path))

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "patterns": PATTERNS,
        "files": files,
        "matches": matches,
        "match_count": len(matches),
        "import_interpretation": {
            "repo_model_import": "from DCNv2.dcn_v2 import DCN",
            "dcn_v2_backend_import": "import _ext as _backend",
            "preferred_smoke_import": "from DCNv2.dcn_v2 import DCN, DCNv2",
        },
    }
    JSON_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote source analysis JSON: {JSON_OUT}")
    print(f"Wrote source analysis MD: {MD_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
