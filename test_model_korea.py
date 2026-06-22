#!/usr/bin/env python3
"""Run BCE-Net Korea synthetic inference smoke test.

Outputs are reviewer-facing probability/mask smoke artifacts, not confirmed
building errors and not accuracy evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, REPO_ROOT / "DCNv2"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from dataset.cd_dataload_korea_512 import Mydataset


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="../results/dev_synthetic/korea_poc/dataset/test_korea.csv")
    parser.add_argument("--weights", default="/home/work/models/BCE-Net/checkpoint-best-whu.pth")
    parser.add_argument("--out-dir", default="../results/dev_synthetic/res-korea")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold-new", type=float, default=0.5)
    parser.add_argument("--threshold-removed", type=float, default=0.5)
    parser.add_argument("--threshold-building", type=float, default=0.5)
    parser.add_argument("--save-prob", type=parse_bool, default=True)
    parser.add_argument("--save-mask", type=parse_bool, default=True)
    parser.add_argument("--has-gt", type=parse_bool, default=False)
    return parser.parse_args()


def exception_record(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {(key[len("module.") :] if key.startswith("module.") else key): value for key, value in state_dict.items()}


def load_model(weights: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    import Testmodel.CDResWHU as whu_module
    from Testmodel.CDResWHU import Baseline34
    from DCNv2.dcn_v2 import DCN, DCNv2

    original_resnet34 = whu_module.resnet34

    def safe_resnet34(*args: Any, **kwargs: Any) -> Any:
        return original_resnet34(pretrained=False, progress=kwargs.get("progress", True))

    whu_module.resnet34 = safe_resnet34
    model = Baseline34(pretrained=False)
    try:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        load_method = "torch.load(weights_only=True)"
    except Exception:
        checkpoint = torch.load(weights, map_location="cpu")
        load_method = "torch.load(weights_only=False fallback)"
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a state_dict mapping")
    stripped = strip_module_prefix(state_dict)
    result = model.load_state_dict(stripped, strict=True)
    model.to(device)
    model.eval()
    return model, {
        "class": "Testmodel.CDResWHU.Baseline34",
        "constructor": "Baseline34(pretrained=False)",
        "checkpoint_path": str(weights),
        "checkpoint_structure": "dict['state_dict']" if "state_dict" in checkpoint else "raw state_dict",
        "checkpoint_load_method": load_method,
        "state_dict_keys": len(state_dict),
        "module_prefix_count": sum(1 for key in state_dict if key.startswith("module.")),
        "strict_load": True,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "dcnv2_import": {"ok": True, "DCN": repr(DCN), "DCNv2": repr(DCNv2)},
    }


def write_geotiff(path: Path, array: np.ndarray, image_path: Path, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype=dtype, compress="deflate", nodata=None)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def colorize(prob: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    clipped = np.clip(prob, 0, 1).astype(np.float32)
    out = np.zeros((*clipped.shape, 3), dtype=np.uint8)
    for channel, value in enumerate(color):
        out[..., channel] = (clipped * value).astype(np.uint8)
    return out


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    mask_bool = mask > 0
    overlay = np.array(color, dtype=np.float32)
    out[mask_bool] = out[mask_bool] * (1 - alpha) + overlay * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_preview(path: Path, image_path: Path, old_path: Path, new_prob: np.ndarray, removed_prob: np.ndarray, building_prob: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(image_path) as src:
        rgb = np.moveaxis(src.read(indexes=[1, 2, 3]), 0, -1).astype(np.uint8)
    with rasterio.open(old_path) as src:
        old = src.read(1)
    panel_old = overlay_mask(rgb, old, (0, 220, 255))
    panel_new = np.maximum((rgb * 0.45).astype(np.uint8), colorize(new_prob, (255, 72, 32)))
    panel_removed = np.maximum((rgb * 0.45).astype(np.uint8), colorize(removed_prob, (210, 60, 255)))
    panel_building = np.maximum((rgb * 0.45).astype(np.uint8), colorize(building_prob, (255, 220, 40)))
    preview = np.concatenate([rgb, panel_old, panel_new, panel_removed, panel_building], axis=1)
    Image.fromarray(preview).save(path)


def tensor_stats(array: np.ndarray) -> dict[str, Any]:
    return {
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array)),
        "has_nan": bool(np.isnan(array).any()),
        "has_inf": bool(np.isinf(array).any()),
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


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    for subdir in (
        "prob/building",
        "prob/removed",
        "prob/new",
        "mask/building",
        "mask/removed_raw",
        "mask/new",
        "preview",
    ):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    csv_path = Path(args.csv).resolve()
    weights = Path(args.weights).resolve()
    metadata = pd.read_csv(csv_path).set_index("tile_id")
    dataset = Mydataset(str(csv_path), augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked",
        "mode": "Managed Container Mode",
        "venv": sys.prefix,
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "csv": str(csv_path),
        "out_dir": str(out_dir),
        "thresholds": {
            "new": args.threshold_new,
            "removed": args.threshold_removed,
            "building": args.threshold_building,
        },
        "has_gt": args.has_gt,
        "metrics_skipped": not args.has_gt,
        "model": {},
        "tiles": [],
        "tile_count": len(dataset),
        "success_tile_count": 0,
        "failed_tile_count": 0,
        "output_count": 0,
        "cuda_memory_before": cuda_memory(),
        "cuda_memory_after": None,
        "timing": {},
        "failure_classification": [],
        "real_data_inference_executed": False,
        "candidate_vectorization_executed": False,
    }

    try:
        model, model_report = load_model(weights, device)
        report["model"] = model_report
    except Exception as exc:
        report["model_error"] = exception_record(exc)
        report["failure_classification"] = ["model load regression"]
        (out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    start_all = time.perf_counter()
    prob_stats: dict[str, list[dict[str, Any]]] = {"building": [], "removed": [], "new": []}

    with torch.no_grad():
        for batch in loader:
            inputs, labels_o, labels_n, labels_m, labels_b, labels, tile_ids = batch
            batch_record: dict[str, Any] = {
                "tile_ids": list(tile_ids),
                "input_shape": list(inputs.shape),
                "labels_o_shape": list(labels_o.shape),
                "ok": False,
            }
            if list(labels_o.shape)[1:] != [int(metadata.loc[tile_ids[0], "height"]), int(metadata.loc[tile_ids[0], "width"])]:
                batch_record["error"] = "labels_o batch shape is not [B,H,W]"
                report["tiles"].append(batch_record)
                report["failed_tile_count"] += len(tile_ids)
                continue
            try:
                inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=False)
                labels_o = labels_o.to(device=device, dtype=torch.float32, non_blocking=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = model.forward(inputs, labels_o)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_record["forward_ms"] = (time.perf_counter() - start) * 1000.0
                if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
                    raise RuntimeError(f"Expected at least 3 model outputs, got {type(outputs).__name__}")
                predicts_b, predicts_mov, predicts_new = outputs[0], outputs[1], outputs[2]
                probs = {
                    "building": torch.sigmoid(predicts_b).detach().cpu().numpy(),
                    "removed": torch.sigmoid(predicts_mov).detach().cpu().numpy(),
                    "new": torch.sigmoid(predicts_new).detach().cpu().numpy(),
                }
                batch_record["output_count"] = len(outputs)
                for sample_index, tile_id in enumerate(tile_ids):
                    tile_id = str(tile_id)
                    row = metadata.loc[tile_id]
                    image_path = Path(row["image_path"])
                    old_path = Path(row["old_footprint_path"])
                    sample_record = {
                        "tile_id": tile_id,
                        "labels_o_item_shape": list(labels_o[sample_index].shape),
                        "outputs": {},
                    }
                    arrays = {
                        "building": probs["building"][sample_index, 0].astype(np.float32),
                        "removed": probs["removed"][sample_index, 0].astype(np.float32),
                        "new": probs["new"][sample_index, 0].astype(np.float32),
                    }
                    if args.save_prob:
                        write_geotiff(out_dir / "prob" / "building" / f"{tile_id}.tif", arrays["building"], image_path, "float32")
                        write_geotiff(out_dir / "prob" / "removed" / f"{tile_id}.tif", arrays["removed"], image_path, "float32")
                        write_geotiff(out_dir / "prob" / "new" / f"{tile_id}.tif", arrays["new"], image_path, "float32")
                    if args.save_mask:
                        masks = {
                            "building": (arrays["building"] >= args.threshold_building).astype(np.uint8) * 255,
                            "removed_raw": (arrays["removed"] >= args.threshold_removed).astype(np.uint8) * 255,
                            "new": (arrays["new"] >= args.threshold_new).astype(np.uint8) * 255,
                        }
                        write_geotiff(out_dir / "mask" / "building" / f"{tile_id}.tif", masks["building"], image_path, "uint8")
                        write_geotiff(out_dir / "mask" / "removed_raw" / f"{tile_id}.tif", masks["removed_raw"], image_path, "uint8")
                        write_geotiff(out_dir / "mask" / "new" / f"{tile_id}.tif", masks["new"], image_path, "uint8")
                    make_preview(
                        out_dir / "preview" / f"{tile_id}_quicklook.png",
                        image_path,
                        old_path,
                        arrays["new"],
                        arrays["removed"],
                        arrays["building"],
                    )
                    for name, arr in arrays.items():
                        stats = tensor_stats(arr)
                        sample_record["outputs"][name] = stats
                        prob_stats[name].append(stats)
                    report["tiles"].append(sample_record)
                    report["success_tile_count"] += 1
                batch_record["ok"] = True
                report["output_count"] += len(tile_ids)
            except Exception as exc:
                batch_record["error"] = exception_record(exc)
                report["failed_tile_count"] += len(tile_ids)
                report["tiles"].append(batch_record)

    report["timing"]["total_ms"] = (time.perf_counter() - start_all) * 1000.0
    report["cuda_memory_after"] = cuda_memory()
    report["probability_stats"] = {
        name: {
            "min": float(min(item["min"] for item in values)) if values else None,
            "max": float(max(item["max"] for item in values)) if values else None,
            "mean_of_means": float(np.mean([item["mean"] for item in values])) if values else None,
        }
        for name, values in prob_stats.items()
    }
    if report["success_tile_count"] == report["tile_count"] and report["tile_count"] > 0:
        report["status"] = "pass"
    elif report["success_tile_count"] > 0:
        report["status"] = "partial"
        report["failure_classification"] = ["partial tile inference issue"]
    else:
        report["status"] = "blocked"
        report["failure_classification"] = ["BCE-Net forward issue"]

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote Korea synthetic inference summary: {summary_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
