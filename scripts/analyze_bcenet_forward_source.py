#!/usr/bin/env python3
"""Analyze BCE-Net forward call source for Stage 6M dummy forward smoke test."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = (REPO_ROOT / "../results").resolve()
OUT_JSON = RESULTS_DIR / "bcenet_forward_source_analysis.json"
OUT_MD = RESULTS_DIR / "bcenet_forward_source_analysis.md"

TARGET_FILES = [
    "test_model_whu.py",
    "test_model_sibu.py",
    "Testmodel/CDResWHU.py",
    "Testmodel/CDResSIBU.py",
    "dataset/cd_dataload_512.py",
]

PATTERNS = [
    "forward",
    "def forward",
    "net.forward",
    "model(",
    "inputs",
    "labels_o",
    "labelso",
    "predicts_b",
    "predicts_mov",
    "predicts_new",
    "sigmoid",
    "torch.sigmoid",
    ".cuda",
    "Variable",
    "DataParallel",
    "Baseline34",
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


def collect_key_lines(path: Path, needles: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for lineno, line in enumerate(read_lines(path), start=1):
        if any(needle in line for needle in needles):
            records.append({"line": lineno, "text": line.strip()})
    return records


def make_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BCE-Net Forward Source Analysis",
        "",
        f"- Created at UTC: `{report['created_at_utc']}`",
        f"- Repo root: `{report['repo_root']}`",
        "",
        "## Inferred Forward Contract",
        "",
    ]
    for key, value in report["inference"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## WHU Test Script Key Lines", ""])
    for item in report["whu_test_key_lines"]:
        lines.append(f"- `{item['line']}`: `{item['text']}`")
    lines.extend(["", "## WHU Model Key Lines", ""])
    for item in report["whu_model_key_lines"]:
        lines.append(f"- `{item['line']}`: `{item['text']}`")
    lines.extend(["", "## Pattern Matches", ""])
    for item in report["matches"]:
        lines.append(f"- `{item['file']}:{item['line']}` [{item['pattern']}]: `{item['text']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    matches: list[dict[str, Any]] = []
    for rel in TARGET_FILES:
        path = REPO_ROOT / rel
        files.append({"path": rel, "exists": path.exists()})
        matches.extend(find_matches(path))

    whu_test = REPO_ROOT / "test_model_whu.py"
    whu_model = REPO_ROOT / "Testmodel/CDResWHU.py"
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "files": files,
        "patterns": PATTERNS,
        "matches": matches,
        "whu_test_key_lines": collect_key_lines(
            whu_test,
            [
                "inputs, labels_o",
                "labels_o =",
                "predicts_b, predicts_mov, predicts_new",
                "torch.sigmoid",
                "predictsn[",
                "predictsb[",
            ],
        ),
        "whu_model_key_lines": collect_key_lines(
            whu_model,
            [
                "def forward(self, inputs, labelso)",
                "def forward(self, inputs,labelso)",
                "torch.unsqueeze(labelso",
                "return out, mov_out, new_out",
                "feat_all",
                "feat_mov",
            ],
        ),
        "inference": {
            "original_forward_call": "predicts_b, predicts_mov, predicts_new, _, _ = net.forward(inputs, labels_o)",
            "input_tensor": "inputs, expected [B, 3, H, W] float tensor on CUDA",
            "labels_o_role": "historical/old building footprint mask used as labelso",
            "labels_o_shape_evidence": "test script indexes labels_o[index] as 2D and model calls torch.unsqueeze(labelso, dim=1), so [B, H, W] is the likely runtime shape",
            "requested_first_label_shape_candidate": "[B, 1, H, W]",
            "likely_label_shape_candidate": "[B, H, W]",
            "output_unpacking": "outputs[0]=predicts_b, outputs[1]=predicts_mov, outputs[2]=predicts_new, outputs[3]=feat_all, outputs[4]=feat_mov",
            "sigmoid_location": "test_model_whu.py applies torch.sigmoid to predicts_new, predicts_mov, predicts_b",
            "threshold_location": "test_model_whu.py thresholds predicts_new and predicts_b at 0.5; thresholding is recorded only, not executed in Stage 6M",
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(make_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote forward source analysis JSON: {OUT_JSON}")
    print(f"Wrote forward source analysis Markdown: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
