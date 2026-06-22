#!/usr/bin/env python3
"""Smoke test DCNv2 imports and a minimal CUDA forward pass."""

from __future__ import annotations

import importlib
import json
import argparse
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DCNV2_DIR = REPO_ROOT / "DCNv2"
RESULTS_DIR = (REPO_ROOT / "../results").resolve()
OUT_JSON = RESULTS_DIR / "dcnv2_smoke_test.json"

for path in (REPO_ROOT, DCNV2_DIR):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def attempt(label: str, func) -> dict[str, Any]:
    record: dict[str, Any] = {"label": label, "ok": False}
    try:
        value = func()
        record["ok"] = True
        record["value"] = value
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    return record


def torch_runtime() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "python": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    return info


def import_candidates() -> tuple[list[dict[str, Any]], type | None, type | None]:
    records: list[dict[str, Any]] = []
    selected_dcn = None
    selected_dcnv2 = None

    def import_pkg_path():
        from DCNv2.dcn_v2 import DCN, DCNv2

        return {"DCN": repr(DCN), "DCNv2": repr(DCNv2)}

    record = attempt("from DCNv2.dcn_v2 import DCN, DCNv2", import_pkg_path)
    records.append(record)
    if record["ok"]:
        from DCNv2.dcn_v2 import DCN, DCNv2

        selected_dcn = DCN
        selected_dcnv2 = DCNv2

    def import_top_level():
        from dcn_v2 import DCN, DCNv2

        return {"DCN": repr(DCN), "DCNv2": repr(DCNv2)}

    record = attempt("from dcn_v2 import DCN, DCNv2", import_top_level)
    records.append(record)
    if selected_dcn is None and record["ok"]:
        from dcn_v2 import DCN, DCNv2

        selected_dcn = DCN
        selected_dcnv2 = DCNv2

    def import_package():
        module = importlib.import_module("DCNv2")
        return {"module": repr(module), "file": getattr(module, "__file__", None)}

    records.append(attempt("import DCNv2", import_package))

    def import_ext():
        module = importlib.import_module("_ext")
        return {"module": repr(module), "file": getattr(module, "__file__", None)}

    records.append(attempt("import _ext", import_ext))
    return records, selected_dcn, selected_dcnv2


def forward_smoke(DCN) -> dict[str, Any]:
    import torch

    record: dict[str, Any] = {"attempted": False, "ok": False}
    if DCN is None:
        record["status"] = "not_attempted_no_dcn_class"
        return record
    if not torch.cuda.is_available():
        record["status"] = "not_attempted_cuda_unavailable"
        return record

    record["attempted"] = True
    try:
        device = torch.device("cuda:0")
        model = DCN(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1,
        ).to(device)
        model.eval()
        x = torch.randn(1, 3, 16, 16, device=device)
        with torch.no_grad():
            y = model(x)
        torch.cuda.synchronize()
        record.update(
            {
                "ok": True,
                "status": "forward_passed",
                "input_shape": list(x.shape),
                "output_shape": list(y.shape),
                "output_dtype": str(y.dtype),
            }
        )
    except Exception as exc:
        record.update(
            {
                "ok": False,
                "status": "import_pass_forward_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-json",
        default=str(OUT_JSON),
        help="Path to write the DCNv2 smoke test JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "dcnv2_dir": str(DCNV2_DIR),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "sys_path_head": sys.path[:8],
        "runtime": {},
        "import_attempts": [],
        "forward_smoke": {},
        "status": "failed",
        "failure_classification": [],
    }

    try:
        report["runtime"] = torch_runtime()
    except Exception as exc:
        report["runtime_error"] = f"{type(exc).__name__}: {exc}"
        report["runtime_traceback"] = traceback.format_exc()

    import_records, dcn_cls, _dcnv2_cls = import_candidates()
    report["import_attempts"] = import_records
    report["forward_smoke"] = forward_smoke(dcn_cls)

    import_ok = dcn_cls is not None
    forward_ok = bool(report["forward_smoke"].get("ok"))
    if import_ok and forward_ok:
        report["status"] = "pass"
    elif import_ok:
        report["status"] = report["forward_smoke"].get("status", "import_pass_forward_not_tested")
        report["failure_classification"].append("DCNv2 runtime/forward issue")
    else:
        report["status"] = "import_failed"
        report["failure_classification"].append("DCNv2 build/import issue")

    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote DCNv2 smoke test JSON: {out_json}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
