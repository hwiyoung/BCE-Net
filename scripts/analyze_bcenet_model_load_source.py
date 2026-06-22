#!/usr/bin/env python3
"""Analyze BCE-Net WHU model construction and checkpoint loading source paths."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = (REPO_ROOT / "../results").resolve()
OUT_JSON = RESULTS_DIR / "bcenet_model_load_source_analysis.json"
OUT_MD = RESULTS_DIR / "bcenet_model_load_source_analysis.md"

TARGET_FILES = [
    "test_model_whu.py",
    "Testmodel/CDResWHU.py",
    "Testmodel/CDResSIBU.py",
    "DCNv2/dcn_v2.py",
]

PATTERNS = [
    "from Testmodel",
    "import",
    "Baseline",
    "Baseline34",
    "pretrained",
    "DataParallel",
    "load_state_dict",
    "state_dict",
    "checkpoint",
    "strict",
    "forward",
    "labels_o",
    "predicts_b",
    "predicts_mov",
    "predicts_new",
    "DCN",
]


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def find_matches(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    matches: list[dict[str, Any]] = []
    for lineno, line in enumerate(read_lines(path), start=1):
        for pattern in PATTERNS:
            if pattern in line:
                matches.append(
                    {
                        "file": str(path.relative_to(REPO_ROOT)),
                        "line": lineno,
                        "pattern": pattern,
                        "text": line.rstrip(),
                    }
                )
    return matches


def extract_class_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    candidates: list[dict[str, Any]] = []
    lines = read_lines(path)
    for idx, line in enumerate(lines):
        class_match = re.match(r"^class\s+(\w*Baseline\w*)\s*\(", line)
        if not class_match:
            continue
        cls_name = class_match.group(1)
        init_line = None
        for j in range(idx + 1, min(idx + 25, len(lines))):
            if re.match(r"^\S", lines[j]) and not lines[j].startswith("class "):
                break
            if "def __init__" in lines[j]:
                init_line = {"line": j + 1, "text": lines[j].strip()}
                break
        candidates.append(
            {
                "file": str(path.relative_to(REPO_ROOT)),
                "line": idx + 1,
                "class": cls_name,
                "init": init_line,
            }
        )
    return candidates


def summarize_test_script(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model_creation": [],
        "checkpoint_load": [],
        "data_parallel": [],
        "forward_calls": [],
    }
    if not path.exists():
        return summary
    for lineno, line in enumerate(read_lines(path), start=1):
        stripped = line.strip()
        if "Baseline34(" in stripped:
            summary["model_creation"].append({"line": lineno, "text": stripped})
        if "torch.nn.DataParallel" in stripped or "DataParallel(" in stripped:
            summary["data_parallel"].append({"line": lineno, "text": stripped})
        if "load_state_dict" in stripped or "torch.load" in stripped or "state_dict" in stripped:
            summary["checkpoint_load"].append({"line": lineno, "text": stripped})
        if ".forward(" in stripped or "net(" in stripped:
            summary["forward_calls"].append({"line": lineno, "text": stripped})
    return summary


def make_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BCE-Net WHU Model Load Source Analysis",
        "",
        f"- Created at UTC: `{report['created_at_utc']}`",
        f"- Repo root: `{report['repo_root']}`",
        "",
        "## File Existence",
        "",
    ]
    for item in report["files"]:
        status = "exists" if item["exists"] else "missing"
        lines.append(f"- `{item['path']}`: {status}")
    lines.extend(["", "## Original WHU Test Script", ""])
    test_summary = report["original_test_script"]
    for key, title in [
        ("model_creation", "Model Creation"),
        ("data_parallel", "DataParallel"),
        ("checkpoint_load", "Checkpoint Load"),
        ("forward_calls", "Forward Calls Recorded Only"),
    ]:
        lines.append(f"### {title}")
        if not test_summary[key]:
            lines.append("- none")
        for item in test_summary[key]:
            lines.append(f"- `{item['line']}`: `{item['text']}`")
        lines.append("")
    lines.extend(["## Model Class Candidates", ""])
    if not report["model_class_candidates"]:
        lines.append("- none")
    for item in report["model_class_candidates"]:
        init_text = item["init"]["text"] if item.get("init") else "init not found nearby"
        lines.append(f"- `{item['file']}:{item['line']}` `{item['class']}` / `{init_text}`")
    lines.extend(["", "## Pattern Matches", ""])
    for item in report["matches"]:
        lines.append(f"- `{item['file']}:{item['line']}` [{item['pattern']}]: `{item['text']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    matches: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for rel in TARGET_FILES:
        path = REPO_ROOT / rel
        files.append({"path": rel, "exists": path.exists()})
        matches.extend(find_matches(path))
        candidates.extend(extract_class_candidates(path))

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "files": files,
        "patterns": PATTERNS,
        "matches": matches,
        "model_class_candidates": candidates,
        "original_test_script": summarize_test_script(REPO_ROOT / "test_model_whu.py"),
        "inferred_model_class": "Testmodel.CDResWHU.Baseline34",
        "inferred_original_constructor": "Baseline34(pretrained=True).cuda()",
        "inferred_original_checkpoint_load": "torch.nn.DataParallel(net).load_state_dict(torch.load(trained_model)['state_dict'])",
        "notes": [
            "The original WHU test script wraps the model in torch.nn.DataParallel before loading state_dict.",
            "The active Baseline34 constructor accepts pretrained=False but internally calls resnet34(pretrained=True).",
            "Stage 5M must avoid external pretrained weight downloads by forcing the resnet34 call to pretrained=False during smoke construction.",
            "Forward calls are recorded for Stage 6M planning only and are not executed in Stage 5M.",
        ],
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(make_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote source analysis JSON: {OUT_JSON}")
    print(f"Wrote source analysis Markdown: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
