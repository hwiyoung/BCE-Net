#!/usr/bin/env python3
"""Propose a minimal DCNv2 compatibility patch plan from analysis artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = (REPO_ROOT / "../results").resolve()
LEGACY_JSON = RESULTS_DIR / "dcnv2_legacy_api_analysis.json"
FAILURE_JSON = RESULTS_DIR / "dcnv2_build_failure_summary.json"
JSON_OUT = RESULTS_DIR / "dcnv2_patch_plan.json"
MD_OUT = RESULTS_DIR / "dcnv2_patch_plan.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    legacy = load_json(LEGACY_JSON)
    failure = load_json(FAILURE_JSON)
    matches = legacy.get("matches", [])

    active_files = sorted(
        {
            match["file"]
            for match in matches
            if match["risk_category"]
            in {
                "removed_legacy_torch_header",
                "removed_legacy_cuda_header",
                "removed_legacy_cuda_state",
                "removed_legacy_cuda_macro",
                "deprecated_tensor_pointer_api",
            }
            and not match.get("is_comment_only")
        }
    )

    modifications = [
        {
            "file": "DCNv2/src/cpu/dcn_v2_cpu.cpp",
            "purpose": "Remove legacy TH include, replace THArgCheck, and update active .data<T>() calls to .data_ptr<T>().",
        },
        {
            "file": "DCNv2/src/cpu/dcn_v2_im2col_cpu.cpp",
            "purpose": "Remove legacy TH include.",
        },
        {
            "file": "DCNv2/src/cpu/dcn_v2_psroi_pooling_cpu.cpp",
            "purpose": "Remove legacy TH include and update active .data<T>() calls to .data_ptr<T>().",
        },
        {
            "file": "DCNv2/src/cuda/dcn_v2_cuda.cu",
            "purpose": "Remove legacy THC include/state usage and replace THArgCheck.",
        },
        {
            "file": "DCNv2/src/cuda/dcn_v2_im2col_cuda.cu",
            "purpose": "Remove legacy THC include if no THC symbols are used.",
        },
        {
            "file": "DCNv2/src/cuda/dcn_v2_psroi_pooling_cuda.cu",
            "purpose": "Remove legacy THC include and replace THCudaCheck/THCCeilDiv.",
        },
    ]

    next_errors = [
        "AT_ASSERTM or other assertion macro compatibility issue",
        "AT_DISPATCH_FLOATING_TYPES signature issue",
        "CUDA kernel helper macro issue",
        "link or undefined symbol issue",
        "runtime failure in generated _ext forward path",
    ]

    patch_size = "medium" if len(modifications) <= 6 else "large"
    recommendation = "apply_minimal_patch" if patch_size in {"small", "medium"} else "inspect_more"

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_analysis": str(LEGACY_JSON),
        "failure_summary": str(FAILURE_JSON),
        "active_legacy_files": active_files,
        "modifications": modifications,
        "expected_next_errors": next_errors,
        "patch_size": patch_size,
        "recommendation": recommendation,
        "failure_classification_candidates": failure.get("failure_classification_candidates", []),
    }
    JSON_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# DCNv2 Patch Plan",
        "",
        f"Patch size: `{patch_size}`",
        f"Recommendation: `{recommendation}`",
        "",
        "## Files to Modify",
        "",
    ]
    for item in modifications:
        lines.append(f"- `{item['file']}`: {item['purpose']}")
    lines.extend(["", "## Expected Next Errors", ""])
    for item in next_errors:
        lines.append(f"- {item}")
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote patch plan JSON: {JSON_OUT}")
    print(f"Wrote patch plan MD: {MD_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
