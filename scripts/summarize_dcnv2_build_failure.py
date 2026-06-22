#!/usr/bin/env python3
"""Summarize DCNv2 build failure logs with small context windows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = (REPO_ROOT / "../results").resolve()
JSON_OUT = RESULTS_DIR / "dcnv2_build_failure_summary.json"
MD_OUT = RESULTS_DIR / "dcnv2_build_failure_summary.md"

KEYWORDS = [
    "fatal error",
    "error:",
    "TH/TH.h",
    "THC/THC.h",
    "AT_CHECK",
    "THCuda",
    "undefined",
    "not declared",
    "No such file or directory",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="../results/logs/dcnv2_build_ext_inplace.log")
    parser.add_argument("--out-prefix", default="dcnv2_build_failure_summary")
    return parser.parse_args()


def classify(lines: list[str]) -> list[str]:
    text = "\n".join(lines)
    classes: list[str] = []
    if "TH/TH.h" in text or "THC/THC.h" in text:
        classes.append("removed TH/THC headers")
    if "AT_CHECK" in text:
        classes.append("AT_CHECK/TORCH_CHECK issue")
    if "THCuda" in text:
        classes.append("THCuda macro issue")
    if "not declared" in text:
        classes.append("symbol not declared")
    if "undefined" in text:
        classes.append("undefined symbol/link issue")
    if not classes:
        classes.append("unclassified build failure")
    return classes


def main() -> int:
    args = parse_args()
    log_path = Path(args.log).resolve()
    json_out = RESULTS_DIR / f"{args.out_prefix}.json"
    md_out = RESULTS_DIR / f"{args.out_prefix}.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.exists() else []
    interesting_indices = [
        idx for idx, line in enumerate(lines) if any(keyword in line for keyword in KEYWORDS)
    ]
    blocks: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for idx in interesting_indices:
        start = max(0, idx - 2)
        end = min(len(lines), idx + 3)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        blocks.append(
            {
                "start_line": start + 1,
                "end_line": end,
                "lines": lines[start:end],
            }
        )

    last_error_block = blocks[-1] if blocks else None
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path),
        "log_exists": log_path.exists(),
        "keywords": KEYWORDS,
        "match_count": len(interesting_indices),
        "context_blocks": blocks,
        "last_error_block": last_error_block,
        "failure_classification_candidates": classify(lines),
    }
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# DCNv2 Build Failure Summary",
        "",
        f"Log: `{log_path}`",
        "",
        "## Failure Classification Candidates",
        "",
    ]
    for item in report["failure_classification_candidates"]:
        md_lines.append(f"- {item}")
    md_lines.extend(["", "## Context Blocks", ""])
    for block in blocks:
        md_lines.append(f"### Lines {block['start_line']}-{block['end_line']}")
        md_lines.append("")
        md_lines.append("```text")
        md_lines.extend(block["lines"])
        md_lines.append("```")
        md_lines.append("")
    md_out.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote build failure summary JSON: {json_out}")
    print(f"Wrote build failure summary MD: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
