#!/usr/bin/env python3
"""Run BCE-Net WHU dummy tensor forward smoke test without real data inference."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = (REPO_ROOT / "../results").resolve()
DEFAULT_WEIGHTS = Path("/home/work/models/BCE-Net/checkpoint-best-whu.pth")
DEFAULT_OUT_JSON = RESULTS_DIR / "bcenet_forward_smoke.json"

for path in (REPO_ROOT, REPO_ROOT / "DCNv2"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-mode", default="random", choices=["random", "zeros", "ones"])
    parser.add_argument("--old-mask-mode", default="synthetic_rect", choices=["synthetic_rect", "zeros", "ones"])
    parser.add_argument("--try-fallback-sizes", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def exception_record(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def cuda_memory() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


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
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    return info


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

        record.update({"ok": True, "module": "Testmodel.CDResWHU", "class": "Baseline34"})
        return record, whu_module, Baseline34
    except Exception as exc:
        record["error"] = exception_record(exc)
        return record, None, None


def install_download_blocker(whu_module: Any) -> dict[str, Any]:
    guard: dict[str, Any] = {"download_blocked_calls": [], "resnet34_calls": []}

    def blocked_load_state_dict_from_url(*args: Any, **kwargs: Any) -> Any:
        guard["download_blocked_calls"].append({"args": [str(a) for a in args], "kwargs": kwargs})
        raise RuntimeError("External pretrained weight download is blocked in Stage 6M.")

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


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_model(weights: Path, device: torch.device) -> tuple[torch.nn.Module | None, dict[str, Any]]:
    report: dict[str, Any] = {
        "model_class": "Testmodel.CDResWHU.Baseline34",
        "constructor": "Baseline34(pretrained=False)",
        "checkpoint_path": str(weights),
        "checkpoint_structure": None,
        "strict_load": {"ok": False},
        "download_guard": {},
    }
    model_import, whu_module, Baseline34 = import_whu_module()
    report["model_import"] = model_import
    if not model_import.get("ok") or whu_module is None or Baseline34 is None:
        return None, report

    report["download_guard"] = install_download_blocker(whu_module)
    try:
        model = Baseline34(pretrained=False)
    except Exception as exc:
        report["constructor_error"] = exception_record(exc)
        return None, report

    try:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        report["checkpoint_load_method"] = "torch.load(weights_only=True)"
    except Exception as exc:
        report["checkpoint_load_warning"] = exception_record(exc)
        try:
            checkpoint = torch.load(weights, map_location="cpu")
            report["checkpoint_load_method"] = "torch.load(weights_only=False fallback)"
        except Exception as fallback_exc:
            report["checkpoint_error"] = exception_record(fallback_exc)
            return None, report

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]
        report["checkpoint_structure"] = "dict['state_dict']"
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        report["checkpoint_structure"] = "raw dict state_dict"
    else:
        report["checkpoint_error"] = f"unsupported checkpoint type: {type(checkpoint).__name__}"
        return None, report

    stripped = strip_module_prefix(state_dict)
    report["state_dict_key_count"] = len(state_dict)
    report["module_prefix_count"] = sum(1 for key in state_dict if key.startswith("module."))
    try:
        result = model.load_state_dict(stripped, strict=True)
        report["strict_load"] = {
            "ok": True,
            "prefix_handling": "removed module. prefix",
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }
    except Exception as exc:
        report["strict_load"] = {"ok": False, "error": exception_record(exc)}
        return None, report

    try:
        model.to(device)
        model.eval()
        first_param = next(model.parameters())
        report["cuda_move"] = {"ok": True, "device": str(first_param.device), "model_eval": not model.training}
    except Exception as exc:
        report["cuda_move"] = {"ok": False, "error": exception_record(exc)}
        return None, report
    return model, report


def make_inputs(
    batch_size: int,
    height: int,
    width: int,
    input_mode: str,
    label_shape_kind: str,
    old_mask_mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if input_mode == "random":
        inputs = torch.rand(batch_size, 3, height, width, dtype=torch.float32, device=device)
    elif input_mode == "zeros":
        inputs = torch.zeros(batch_size, 3, height, width, dtype=torch.float32, device=device)
    else:
        inputs = torch.ones(batch_size, 3, height, width, dtype=torch.float32, device=device)

    if label_shape_kind == "B1HW":
        labels = torch.zeros(batch_size, 1, height, width, dtype=torch.float32, device=device)
        fill = labels[:, :, height // 4 : height * 3 // 4, width // 4 : width * 3 // 4]
    elif label_shape_kind == "BHW":
        labels = torch.zeros(batch_size, height, width, dtype=torch.float32, device=device)
        fill = labels[:, height // 4 : height * 3 // 4, width // 4 : width * 3 // 4]
    elif label_shape_kind == "B3HW":
        labels = torch.zeros(batch_size, 3, height, width, dtype=torch.float32, device=device)
        fill = labels[:, :, height // 4 : height * 3 // 4, width // 4 : width * 3 // 4]
    else:
        raise ValueError(f"unknown label shape kind: {label_shape_kind}")

    if old_mask_mode == "synthetic_rect":
        fill.fill_(1.0)
        if height >= 128 and width >= 128:
            if label_shape_kind == "BHW":
                labels[:, height // 8 : height // 4, width // 8 : width // 3] = 1.0
            else:
                labels[:, :, height // 8 : height // 4, width // 8 : width // 3] = 1.0
    elif old_mask_mode == "ones":
        labels.fill_(1.0)

    meta = {
        "inputs_shape": list(inputs.shape),
        "labels_o_shape": list(labels.shape),
        "input_mode": input_mode,
        "old_mask_mode": old_mask_mode,
        "label_shape_kind": label_shape_kind,
    }
    return inputs, labels, meta


def tensor_summary(tensor: torch.Tensor, name: str, index: int | None = None) -> dict[str, Any]:
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    stats_tensor = detached.float()
    return {
        "name": name,
        "index": index,
        "is_tensor": True,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "min": float(stats_tensor.min().item()),
        "max": float(stats_tensor.max().item()),
        "mean": float(stats_tensor.mean().item()),
        "std": float(stats_tensor.std(unbiased=False).item()),
        "has_nan": bool(torch.isnan(detached).any().item()),
        "has_inf": bool(torch.isinf(detached).any().item()),
        "all_finite": bool(finite.all().item()),
        "is_bchw": detached.ndim == 4,
    }


def summarize_outputs(outputs: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if isinstance(outputs, tuple):
        sequence = list(outputs)
        output_type = "tuple"
    elif isinstance(outputs, list):
        sequence = outputs
        output_type = "list"
    else:
        sequence = [outputs]
        output_type = type(outputs).__name__

    output_names = ["predicts_b", "predicts_mov", "predicts_new", "feat_all", "feat_mov"]
    summaries: list[dict[str, Any]] = []
    sigmoid_summaries: list[dict[str, Any]] = []
    for idx, item in enumerate(sequence):
        name = output_names[idx] if idx < len(output_names) else f"output_{idx}"
        if torch.is_tensor(item):
            summaries.append(tensor_summary(item, name, idx))
            if idx < 3:
                sigmoid_summaries.append(tensor_summary(torch.sigmoid(item), f"sigmoid_{name}", idx))
        else:
            summaries.append(
                {
                    "name": name,
                    "index": idx,
                    "is_tensor": False,
                    "type": type(item).__name__,
                    "repr": repr(item),
                }
            )
    meta = {
        "output_type": output_type,
        "output_count": len(sequence),
        "mapping": {
            "predicts_b": "outputs[0]",
            "predicts_mov": "outputs[1]",
            "predicts_new": "outputs[2]",
        },
    }
    return summaries, sigmoid_summaries, meta


def run_forward_attempt(
    model: torch.nn.Module,
    args: argparse.Namespace,
    height: int,
    width: int,
    label_shape_kind: str,
    call_kind: str,
    device: torch.device,
) -> dict[str, Any]:
    attempt: dict[str, Any] = {
        "height": height,
        "width": width,
        "label_shape_kind": label_shape_kind,
        "forward_call": call_kind,
        "ok": False,
    }
    try:
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        inputs, labels_o, input_meta = make_inputs(
            args.batch_size,
            height,
            width,
            args.input_mode,
            label_shape_kind,
            args.old_mask_mode,
            device,
        )
        attempt["input_meta"] = input_meta
        attempt["cuda_memory_before"] = cuda_memory()
        start = time.perf_counter()
        with torch.no_grad():
            if call_kind == "model.forward(inputs, labels_o)":
                outputs = model.forward(inputs, labels_o)
            elif call_kind == "model(inputs, labels_o)":
                outputs = model(inputs, labels_o)
            else:
                raise ValueError(f"unsupported call kind: {call_kind}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        output_summaries, sigmoid_summaries, output_meta = summarize_outputs(outputs)
        attempt.update(
            {
                "ok": True,
                "timing_ms": elapsed_ms,
                "cuda_memory_after": cuda_memory(),
                "output_meta": output_meta,
                "output_summaries": output_summaries,
                "sigmoid_summaries": sigmoid_summaries,
            }
        )
    except Exception as exc:
        attempt["error"] = exception_record(exc)
        attempt["cuda_memory_after_error"] = cuda_memory()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return attempt


def failure_classes(report: dict[str, Any]) -> list[str]:
    classes = []
    if not report["dcnv2_import"].get("ok"):
        classes.append("DCNv2 runtime regression")
    if not report["model_load"].get("strict_load", {}).get("ok"):
        classes.append("model load regression")
    if not report["forward"].get("selected_attempt"):
        messages = " ".join(
            attempt.get("error", {}).get("message", "") for attempt in report["forward"].get("attempts", [])
        ).lower()
        if "positional" in messages or "argument" in messages:
            classes.append("forward signature mismatch")
        if "size" in messages or "shape" in messages or "dimension" in messages:
            classes.append("labels_o shape mismatch")
        if "cuda" in messages:
            classes.append("CUDA runtime error")
        if "illegal memory access" in messages:
            classes.append("CUDA illegal memory access")
        if not classes:
            classes.append("forward runtime failure")
    selected = report["forward"].get("selected_attempt")
    if selected:
        summaries = selected.get("output_summaries", [])
        if selected.get("output_meta", {}).get("output_count", 0) < 3:
            classes.append("output tuple mismatch")
        if any(item.get("has_nan") or item.get("has_inf") for item in summaries if item.get("is_tensor")):
            classes.append("NaN/Inf output issue")
    return sorted(set(classes))


def main() -> int:
    args = parse_args()
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    torch.manual_seed(1024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1024)

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "environment": runtime_info(),
        "dcnv2_import": import_dcnv2(),
        "model_load": {},
        "forward": {
            "requested_size": [args.height, args.width],
            "attempts": [],
            "selected_attempt": None,
            "fallback_used": False,
        },
        "status": "failed",
        "failure_classification": [],
        "real_inference_executed": False,
        "geospatial_processing_executed": False,
    }

    if not report["dcnv2_import"].get("ok"):
        report["failure_classification"] = failure_classes(report)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    model, model_report = load_model(Path(args.weights).resolve(), device)
    report["model_load"] = model_report
    if model is None:
        report["failure_classification"] = failure_classes(report)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    size_candidates = [(args.height, args.width)]
    if args.try_fallback_sizes:
        for size in (256, 128):
            if (size, size) not in size_candidates:
                size_candidates.append((size, size))
    label_candidates = ["B1HW", "BHW", "B3HW"]
    call_candidates = ["model.forward(inputs, labels_o)", "model(inputs, labels_o)"]

    for height, width in size_candidates:
        for label_shape_kind in label_candidates:
            for call_kind in call_candidates:
                attempt = run_forward_attempt(model, args, height, width, label_shape_kind, call_kind, device)
                report["forward"]["attempts"].append(attempt)
                if attempt.get("ok"):
                    report["forward"]["selected_attempt"] = attempt
                    report["forward"]["fallback_used"] = [height, width] != [args.height, args.width]
                    break
            if report["forward"]["selected_attempt"]:
                break
        if report["forward"]["selected_attempt"]:
            break

    selected = report["forward"]["selected_attempt"]
    if selected is None:
        report["status"] = "blocked"
    else:
        output_count = selected.get("output_meta", {}).get("output_count", 0)
        summaries = selected.get("output_summaries", [])
        sigmoid_count = len(selected.get("sigmoid_summaries", []))
        has_bad = any(item.get("has_nan") or item.get("has_inf") for item in summaries if item.get("is_tensor"))
        if (
            not report["forward"]["fallback_used"]
            and output_count >= 3
            and sigmoid_count >= 3
            and not has_bad
        ):
            report["status"] = "pass"
        else:
            report["status"] = "partial"

    report["failure_classification"] = failure_classes(report)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote BCE-Net forward smoke JSON: {out_json}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
