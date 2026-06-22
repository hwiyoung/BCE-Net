#!/usr/bin/env python3
"""Inspect BCE-Net pretrained weight files without running inference."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "size_bytes": stat.st_size,
        "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def priority(path: Path) -> tuple[int, float, str]:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name == "checkpoint-best-sibu.pth":
        rank = 0
    elif "sibu" in name:
        rank = 1
    elif name == "checkpoint-best-whu.pth":
        rank = 2
    elif "whu" in name:
        rank = 3
    elif suffix == ".pth":
        rank = 4
    else:
        rank = 5
    return (rank, -path.stat().st_mtime, name)


def find_weights(weights_dir: Path) -> list[Path]:
    patterns = ("*.pth", "*.pt", "*.ckpt")
    found: list[Path] = []
    for pattern in patterns:
        found.extend(weights_dir.rglob(pattern))
    return sorted(set(found), key=priority)


def extract_state_dict(checkpoint: Any) -> tuple[Any | None, str, list[str]]:
    if isinstance(checkpoint, dict):
        top_level_keys = [str(key) for key in checkpoint.keys()]
        for key in ("state_dict", "model_state_dict", "net", "model", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value, f"dict[{key!r}]", top_level_keys
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint, "raw_state_dict", top_level_keys
        return None, "dict_without_detected_state_dict", top_level_keys
    return None, type(checkpoint).__name__, []


def tensor_shapes(state_dict: Any, limit: int = 20) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not isinstance(state_dict, dict):
        return records
    for key, value in list(state_dict.items())[:limit]:
        record: dict[str, Any] = {"key": str(key), "type": type(value).__name__}
        if hasattr(value, "shape"):
            record["shape"] = list(value.shape)
            record["dtype"] = str(getattr(value, "dtype", ""))
        records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", default="../models/BCE-Net")
    parser.add_argument("--out-json", default="../results/bcenet_weight_inspection.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights_dir = Path(args.weights_dir).resolve()
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights_dir": str(weights_dir),
        "weights_dir_exists": weights_dir.exists(),
        "candidate_files": [],
        "selected_weight_path": None,
        "warnings": [],
    }

    if not weights_dir.exists():
        report["error"] = "weights_dir does not exist"
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2

    candidates = find_weights(weights_dir)
    report["candidate_files"] = [file_record(path) for path in candidates]

    has_sibu = any("sibu" in path.name.lower() for path in candidates)
    if not has_sibu:
        report["warnings"].append(
            "Only WHU pretrained weight is available. Domain gap risk exists for Korea PoC."
        )

    if not candidates:
        report["error"] = "no .pth, .pt, or .ckpt files found"
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 3

    selected = candidates[0]
    report["selected_weight_path"] = str(selected.resolve())
    report["selected_weight"] = file_record(selected)

    try:
        checkpoint = torch.load(str(selected), map_location="cpu")
    except Exception as exc:
        report["load_error"] = f"{type(exc).__name__}: {exc}"
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 4

    state_dict, checkpoint_type, top_level_keys = extract_state_dict(checkpoint)
    state_dict_keys = [str(key) for key in list(state_dict.keys())[:20]] if isinstance(state_dict, dict) else []
    report.update(
        {
            "checkpoint_type": checkpoint_type,
            "top_level_keys": top_level_keys,
            "has_state_dict": isinstance(state_dict, dict),
            "state_dict_key_count": len(state_dict) if isinstance(state_dict, dict) else None,
            "state_dict_keys_20": state_dict_keys,
            "tensor_shapes_20": tensor_shapes(state_dict, limit=20),
        }
    )

    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote weight inspection JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
