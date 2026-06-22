#!/usr/bin/env python3
"""Smoke test BCE-Net WHU model import, strict checkpoint load, and CUDA move."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import socket
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = (REPO_ROOT / "../results").resolve()
DEFAULT_WEIGHTS = Path("/home/work/models/BCE-Net/checkpoint-best-whu.pth")
DEFAULT_OUT_JSON = RESULTS_DIR / "bcenet_model_load_smoke.json"

for path in (REPO_ROOT, REPO_ROOT / "DCNv2"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-source", default="whu", choices=["whu"])
    parser.add_argument("--allow-strict-false", action="store_true")
    return parser.parse_args()


def runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "pythonpath": os.environ.get("PYTHONPATH"),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    return info


def exception_record(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def import_dcnv2() -> dict[str, Any]:
    record: dict[str, Any] = {"ok": False}
    try:
        from DCNv2.dcn_v2 import DCN, DCNv2

        record.update({"ok": True, "DCN": repr(DCN), "DCNv2": repr(DCNv2)})
    except Exception as exc:
        record["error"] = exception_record(exc)
    return record


def import_whu_module() -> tuple[dict[str, Any], Any | None, Any | None]:
    record: dict[str, Any] = {"ok": False}
    try:
        import Testmodel.CDResWHU as whu_module
        from Testmodel.CDResWHU import Baseline34

        record.update(
            {
                "ok": True,
                "module": "Testmodel.CDResWHU",
                "class": "Baseline34",
                "class_repr": repr(Baseline34),
                "signature": str(inspect.signature(Baseline34)),
            }
        )
        return record, whu_module, Baseline34
    except Exception as exc:
        record["error"] = exception_record(exc)
        return record, None, None


def install_download_blocker(whu_module: Any) -> dict[str, Any]:
    guard: dict[str, Any] = {"download_blocked_calls": [], "resnet34_calls": []}

    def blocked_load_state_dict_from_url(*args: Any, **kwargs: Any) -> Any:
        guard["download_blocked_calls"].append({"args": [str(a) for a in args], "kwargs": kwargs})
        raise RuntimeError("External pretrained weight download is blocked in Stage 5M.")

    original_resnet34 = whu_module.resnet34

    def safe_resnet34(*args: Any, **kwargs: Any) -> Any:
        guard["resnet34_calls"].append(
            {
                "original_args": [repr(a) for a in args],
                "original_kwargs": {key: repr(value) for key, value in kwargs.items()},
                "forced_pretrained": False,
            }
        )
        return original_resnet34(pretrained=False, progress=kwargs.get("progress", True))

    whu_module.load_state_dict_from_url = blocked_load_state_dict_from_url
    whu_module.resnet34 = safe_resnet34
    return guard


def construct_model(Baseline34: Any, constructor_kind: str) -> tuple[Any | None, dict[str, Any]]:
    record: dict[str, Any] = {"kind": constructor_kind, "ok": False}
    try:
        if constructor_kind == "Baseline34(pretrained=False)":
            model = Baseline34(pretrained=False)
        elif constructor_kind == "Baseline34()":
            model = Baseline34()
        else:
            raise ValueError(f"unsupported constructor kind: {constructor_kind}")
        record["ok"] = True
        return model, record
    except Exception as exc:
        record["error"] = exception_record(exc)
        return None, record


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "load_method": None,
        "warnings": [],
    }
    if not path.exists():
        record["error"] = "checkpoint file does not exist"
        return record, None

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        record["load_method"] = "torch.load(weights_only=True)"
    except Exception as exc:
        record["warnings"].append(
            {
                "message": "weights_only=True failed; falling back to regular torch.load",
                "error": exception_record(exc),
            }
        )
        try:
            checkpoint = torch.load(path, map_location="cpu")
            record["load_method"] = "torch.load(weights_only=False fallback)"
        except Exception as fallback_exc:
            record["error"] = exception_record(fallback_exc)
            return record, None

    record["checkpoint_type"] = type(checkpoint).__name__
    if isinstance(checkpoint, dict):
        record["top_level_keys"] = list(checkpoint.keys())
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
            record["structure"] = "dict['state_dict']"
        else:
            state_dict = checkpoint
            record["structure"] = "raw dict state_dict"
    else:
        record["structure"] = "unsupported"
        record["error"] = f"unsupported checkpoint object: {type(checkpoint).__name__}"
        return record, None

    record["state_dict_key_count"] = len(state_dict)
    record["first_20_keys"] = list(state_dict.keys())[:20]
    record["tensor_shape_sample"] = [
        {"key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in state_dict.items()
        if torch.is_tensor(value)
    ][:20]
    return record, state_dict


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def state_dict_summary(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    keys = list(state_dict.keys())
    module_prefix_count = sum(1 for key in keys if key.startswith("module."))
    return {
        "key_count": len(keys),
        "first_20_keys": keys[:20],
        "module_prefix_count": module_prefix_count,
        "all_keys_have_module_prefix": bool(keys) and module_prefix_count == len(keys),
        "tensor_shape_sample": [
            {"key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in state_dict.items()
            if torch.is_tensor(value)
        ][:20],
    }


def compare_state_dicts(model_state: dict[str, Any], candidate: dict[str, torch.Tensor]) -> dict[str, Any]:
    model_keys = set(model_state.keys())
    candidate_keys = set(candidate.keys())
    common = sorted(model_keys & candidate_keys)
    shape_mismatch = []
    for key in common:
        model_value = model_state[key]
        candidate_value = candidate[key]
        if torch.is_tensor(model_value) and torch.is_tensor(candidate_value):
            if tuple(model_value.shape) != tuple(candidate_value.shape):
                shape_mismatch.append(
                    {
                        "key": key,
                        "model_shape": list(model_value.shape),
                        "checkpoint_shape": list(candidate_value.shape),
                    }
                )
    return {
        "missing_keys": sorted(model_keys - candidate_keys),
        "unexpected_keys": sorted(candidate_keys - model_keys),
        "shape_mismatch_keys": shape_mismatch,
    }


def attempt_load(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], label: str, strict: bool) -> dict[str, Any]:
    comparison = compare_state_dicts(model.state_dict(), state_dict)
    record: dict[str, Any] = {
        "label": label,
        "strict": strict,
        "ok": False,
        "missing_key_count_precheck": len(comparison["missing_keys"]),
        "unexpected_key_count_precheck": len(comparison["unexpected_keys"]),
        "shape_mismatch_count_precheck": len(comparison["shape_mismatch_keys"]),
        "missing_keys_precheck_sample": comparison["missing_keys"][:30],
        "unexpected_keys_precheck_sample": comparison["unexpected_keys"][:30],
        "shape_mismatch_sample": comparison["shape_mismatch_keys"][:30],
    }
    try:
        result = model.load_state_dict(state_dict, strict=strict)
        record.update(
            {
                "ok": True,
                "missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys),
            }
        )
    except Exception as exc:
        record["error"] = exception_record(exc)
    return record


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"parameter_count": int(total), "trainable_parameter_count": int(trainable)}


def count_dcn_modules(model: torch.nn.Module) -> dict[str, Any]:
    from DCNv2.dcn_v2 import DCN, DCNv2

    modules = []
    for name, module in model.named_modules():
        if isinstance(module, (DCN, DCNv2)):
            modules.append({"name": name, "type": module.__class__.__name__})
    return {"dcn_module_count": len(modules), "dcn_modules": modules}


def classify_failure(report: dict[str, Any]) -> list[str]:
    classes = []
    if not report["dcnv2_import"].get("ok"):
        classes.append("DCNv2 import regression")
    if not report["model_import"].get("ok"):
        classes.append("Testmodel.CDResWHU import failure")
    if report.get("constructor", {}).get("download_guard", {}).get("download_blocked_calls"):
        classes.append("external pretrained download attempt risk")
    if report["checkpoint"].get("error"):
        classes.append("checkpoint file load issue")
    if report["checkpoint"].get("structure") == "unsupported":
        classes.append("checkpoint structure issue")
    strict = report.get("strict_load_result", {})
    if strict.get("status") == "failed":
        classes.append("state_dict key mismatch")
        for attempt in report.get("load_attempts", []):
            if attempt.get("shape_mismatch_count_precheck"):
                classes.append("tensor shape mismatch")
                break
        if report.get("state_dict_summary", {}).get("module_prefix_count"):
            classes.append("DataParallel prefix mismatch")
    if report.get("cuda_move", {}).get("ok") is False:
        classes.append("CUDA move failure")
    return sorted(set(classes))


def main() -> int:
    args = parse_args()
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    weights_path = Path(args.weights).resolve()

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "runtime": runtime_info(),
        "dcnv2_import": {},
        "model_import": {},
        "constructor": {},
        "checkpoint": {},
        "state_dict_summary": {},
        "stripped_state_dict_summary": {},
        "load_attempts": [],
        "strict_load_result": {"status": "not_attempted"},
        "cuda_move": {"attempted": False, "ok": False},
        "parameter_count": None,
        "trainable_parameter_count": None,
        "dcn_module_count": None,
        "dcn_modules": [],
        "forward_executed": False,
        "status": "failed",
        "failure_classification": [],
        "warnings": [],
    }

    report["dcnv2_import"] = import_dcnv2()
    if not report["dcnv2_import"].get("ok"):
        report["failure_classification"] = classify_failure(report)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    model_import, whu_module, Baseline34 = import_whu_module()
    report["model_import"] = model_import
    if not model_import.get("ok") or whu_module is None or Baseline34 is None:
        report["failure_classification"] = classify_failure(report)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    download_guard = install_download_blocker(whu_module)
    report["constructor"]["download_guard"] = download_guard
    report["constructor"]["strategy"] = (
        "Call Baseline34(pretrained=False) while monkeypatching Testmodel.CDResWHU.resnet34 "
        "to force pretrained=False, because active Baseline34 internally calls resnet34(pretrained=True)."
    )

    checkpoint_record, state_dict = load_checkpoint(weights_path)
    report["checkpoint"] = checkpoint_record
    if state_dict is None:
        report["failure_classification"] = classify_failure(report)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    stripped_state_dict = strip_module_prefix(state_dict)
    report["state_dict_summary"] = state_dict_summary(state_dict)
    report["stripped_state_dict_summary"] = state_dict_summary(stripped_state_dict)

    selected_model = None
    selected_label = None
    data_parallel_needed = False
    prefix_handling = "none"

    model, constructor_record = construct_model(Baseline34, "Baseline34(pretrained=False)")
    report["constructor"]["attempts"] = [constructor_record]
    if model is None:
        model, constructor_record_2 = construct_model(Baseline34, "Baseline34()")
        report["constructor"]["attempts"].append(constructor_record_2)

    if model is None:
        report["failure_classification"] = ["model constructor issue"]
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    report["constructor"]["used"] = next(
        item["kind"] for item in report["constructor"]["attempts"] if item.get("ok")
    )

    attempt_model, _ = construct_model(Baseline34, report["constructor"]["used"])
    assert attempt_model is not None
    attempt = attempt_load(attempt_model, state_dict, "model original state_dict strict=True", strict=True)
    report["load_attempts"].append(attempt)
    if attempt.get("ok"):
        selected_model = attempt_model
        selected_label = attempt["label"]

    if selected_model is None:
        attempt_model, _ = construct_model(Baseline34, report["constructor"]["used"])
        assert attempt_model is not None
        attempt = attempt_load(
            attempt_model,
            stripped_state_dict,
            "model module-prefix-stripped state_dict strict=True",
            strict=True,
        )
        report["load_attempts"].append(attempt)
        if attempt.get("ok"):
            selected_model = attempt_model
            selected_label = attempt["label"]
            prefix_handling = "removed module. prefix"

    if selected_model is None:
        attempt_model, _ = construct_model(Baseline34, report["constructor"]["used"])
        assert attempt_model is not None
        dp_model = torch.nn.DataParallel(attempt_model)
        attempt = attempt_load(dp_model, state_dict, "DataParallel(model) original state_dict strict=True", strict=True)
        report["load_attempts"].append(attempt)
        if attempt.get("ok"):
            selected_model = dp_model
            selected_label = attempt["label"]
            data_parallel_needed = True

    if selected_model is None and args.allow_strict_false:
        attempt_model, _ = construct_model(Baseline34, report["constructor"]["used"])
        assert attempt_model is not None
        attempt = attempt_load(
            attempt_model,
            stripped_state_dict,
            "diagnostic only: model stripped state_dict strict=False",
            strict=False,
        )
        report["load_attempts"].append(attempt)
        if attempt.get("ok"):
            report["warnings"].append("strict=False load succeeded only as diagnostic and is not a pass condition.")

    if selected_model is None:
        report["strict_load_result"] = {"status": "failed"}
        report["status"] = "blocked"
        report["failure_classification"] = classify_failure(report)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    report["strict_load_result"] = {
        "status": "pass",
        "selected_attempt": selected_label,
        "data_parallel_needed": data_parallel_needed,
        "prefix_handling": prefix_handling,
    }

    params = count_parameters(selected_model)
    dcn_info = count_dcn_modules(selected_model)
    report.update(params)
    report.update(dcn_info)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        report["cuda_move"] = {"attempted": False, "ok": False, "reason": "CUDA unavailable"}
        report["status"] = "partial"
    else:
        report["cuda_move"]["attempted"] = True
        try:
            selected_model.to(torch.device(args.device))
            selected_model.eval()
            first_param = next(selected_model.parameters())
            report["cuda_move"].update(
                {
                    "ok": True,
                    "device": str(first_param.device),
                    "model_eval": not selected_model.training,
                }
            )
            report["status"] = "pass"
        except Exception as exc:
            report["cuda_move"].update({"ok": False, "error": exception_record(exc)})
            report["status"] = "blocked"

    report["failure_classification"] = classify_failure(report)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote BCE-Net model load smoke JSON: {out_json}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
