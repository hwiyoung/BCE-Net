#!/usr/bin/env python3
"""Evaluate a frozen BCE-Net checkpoint on file-disjoint map-ortho test crops."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


REPO_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, REPO_ROOT / "DCNv2"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from dataset.bcenet_map_ortho import MapOrthoBCENetDataset, seed_worker
from utils.bcenet_visualization import save_qualitative_grid


PROTOCOL = "map-ortho-test-center-crop-512-v1"
SPLIT = "test"
CROP_SIZE = 512
THRESHOLD = 0.5
EXPECTED_SAMPLES = 100


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    total: int = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().bool()
        target = target.detach().bool()
        self.tp += int((prediction & target).sum().item())
        self.fp += int((prediction & ~target).sum().item())
        self.fn += int((~prediction & target).sum().item())
        self.total += int(target.numel())

    def compute(self) -> dict[str, float | int]:
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        iou = self.tp / max(1, self.tp + self.fp + self.fn)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "prediction_rate": (self.tp + self.fp) / max(1, self.total),
            "target_rate": (self.tp + self.fn) / max(1, self.total),
            "total_pixels": self.total,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="dataset/map_ortho_manifest.csv",
        help="Frozen manifest containing the test split.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Frozen checkpoint-best.pth selected using validation only.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="A new, empty output directory. Existing contents are rejected.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--fixed-samples", type=int, default=4)
    parser.add_argument("--ranked-samples", type=int, default=4)
    parser.add_argument("--panel-size", type=int, default=256)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def prepare_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch_load(checkpoint_path, "cpu")
    checkpoint_args = checkpoint.get("args", {})
    model_variant = checkpoint_args.get("model_variant", "whu")
    if model_variant != "whu":
        raise ValueError(
            f"This evaluation protocol expects the WHU architecture, got {model_variant!r}"
        )
    from Testmodel.CDResWHU import Baseline34

    model = Baseline34(pretrained=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    result = model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    metadata = {
        "epoch_zero_based": checkpoint.get("epoch"),
        "epoch_one_based": (
            int(checkpoint["epoch"]) + 1 if checkpoint.get("epoch") is not None else None
        ),
        "validation_selection_score": checkpoint.get("best_score"),
        "training_threshold": checkpoint_args.get("threshold"),
        "training_crop_size": checkpoint_args.get("crop_size"),
        "model_variant": model_variant,
        "imagenet_normalize": bool(
            checkpoint_args.get("imagenet_normalize", False)
        ),
    }
    if metadata["training_threshold"] != THRESHOLD:
        raise ValueError(
            "Checkpoint training threshold differs from fixed evaluation threshold: "
            f"{metadata['training_threshold']} != {THRESHOLD}"
        )
    if metadata["training_crop_size"] != CROP_SIZE:
        raise ValueError(
            "Checkpoint crop size differs from fixed evaluation crop: "
            f"{metadata['training_crop_size']} != {CROP_SIZE}"
        )
    del checkpoint, state_dict
    gc.collect()
    return model, metadata


def move_batch(
    batch: dict[str, torch.Tensor | list[str] | list[int]], device: torch.device
) -> dict[str, torch.Tensor | list[str] | list[int]]:
    return {
        key: value.to(device=device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def metrics_for_sample(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float | int]:
    counts = Counts()
    counts.update(prediction, target)
    return counts.compute()


def select_fixed_indices(
    dataset: MapOrthoBCENetDataset, count: int
) -> list[int]:
    if count <= 0:
        return []
    by_class = {
        value: [
            index
            for index, sample in enumerate(dataset.samples)
            if sample.center_class_value == value
        ]
        for value in (2, 3)
    }
    selected: list[int] = []
    offset = 0
    while len(selected) < min(count, len(dataset)):
        added = False
        for value in (2, 3):
            if offset < len(by_class[value]) and len(selected) < count:
                selected.append(by_class[value][offset])
                added = True
        if not added:
            break
        offset += 1
    return selected


def evaluate(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, dict[str, float | int]], list[dict[str, Any]], float]:
    totals = {name: Counts() for name in ("omission", "excess", "combined")}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            image = batch["image"]
            reference = batch["reference_mask"]
            assert isinstance(image, torch.Tensor)
            assert isinstance(reference, torch.Tensor)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                _, removed_logit, new_logit, _, _ = model(image, reference)

            omission_prediction = torch.sigmoid(new_logit) >= THRESHOLD
            excess_prediction = torch.sigmoid(removed_logit) >= THRESHOLD
            omission_target = batch["target_new_head"]
            excess_target = batch["target_removed_head"]
            assert isinstance(omission_target, torch.Tensor)
            assert isinstance(excess_target, torch.Tensor)
            omission_target = omission_target > 0.5
            excess_target = excess_target > 0.5
            predictions = {
                "omission": omission_prediction,
                "excess": excess_prediction,
                "combined": omission_prediction | excess_prediction,
            }
            targets = {
                "omission": omission_target,
                "excess": excess_target,
                "combined": omission_target | excess_target,
            }
            for name in totals:
                totals[name].update(predictions[name], targets[name])

            sample_ids = list(raw_batch["sample_id"])
            crop_ys = raw_batch["crop_y"].tolist()
            crop_xs = raw_batch["crop_x"].tolist()
            for item_index, sample_id in enumerate(sample_ids):
                row: dict[str, Any] = {
                    "sample_id": sample_id,
                    "crop_y": int(crop_ys[item_index]),
                    "crop_x": int(crop_xs[item_index]),
                }
                for name in totals:
                    values = metrics_for_sample(
                        predictions[name][item_index], targets[name][item_index]
                    )
                    for metric_name, value in values.items():
                        row[f"{name}_{metric_name}"] = value
                rows.append(row)
    return (
        {name: counts.compute() for name, counts in totals.items()},
        rows,
        time.perf_counter() - started,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No per-sample rows were produced")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranked_selections(
    *,
    rows: list[dict[str, Any]],
    sample_to_index: dict[str, int],
    count: int,
) -> dict[str, list[int]]:
    selections: dict[str, list[int]] = {}
    for class_name in ("omission", "excess", "combined"):
        for error_name in ("fp", "fn"):
            key = f"{class_name}_{error_name}"
            ranked = sorted(
                rows,
                key=lambda row: (-int(row[key]), str(row["sample_id"])),
            )[: max(0, count)]
            selections[f"top_{key}"] = [
                sample_to_index[str(row["sample_id"])] for row in ranked
            ]
    return selections


def infer_qualitative_record(
    *,
    model: torch.nn.Module,
    dataset: MapOrthoBCENetDataset,
    index: int,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    raw_batch = default_collate([dataset[index]])
    batch = move_batch(raw_batch, device)
    image = batch["image"]
    reference = batch["reference_mask"]
    assert isinstance(image, torch.Tensor)
    assert isinstance(reference, torch.Tensor)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp and device.type == "cuda",
    ):
        existing, removed, new, _, _ = model(image, reference)
    return {
        "sample_id": raw_batch["sample_id"][0],
        "image": raw_batch["image"][0],
        "reference": raw_batch["reference_mask"][0],
        "target_existing": raw_batch["target_existing"][0],
        "target_omission": raw_batch["target_new_head"][0],
        "target_excess": raw_batch["target_removed_head"][0],
        "existing_probability": torch.sigmoid(existing[0]).cpu(),
        "omission_probability": torch.sigmoid(new[0]).cpu(),
        "excess_probability": torch.sigmoid(removed[0]).cpu(),
    }


def write_qualitative_outputs(
    *,
    model: torch.nn.Module,
    dataset: MapOrthoBCENetDataset,
    selections: dict[str, list[int]],
    rows: list[dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    amp: bool,
    panel_size: int,
    imagenet_normalized: bool,
) -> dict[str, list[dict[str, Any]]]:
    qualitative_dir = output_dir / "qualitative"
    samples_dir = qualitative_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    unique_indices = sorted({index for indices in selections.values() for index in indices})
    records = {
        index: infer_qualitative_record(
            model=model,
            dataset=dataset,
            index=index,
            device=device,
            amp=amp,
        )
        for index in unique_indices
    }
    row_by_id = {str(row["sample_id"]): row for row in rows}
    selection_manifest: dict[str, list[dict[str, Any]]] = {}
    for name, indices in selections.items():
        selected_records = [records[index] for index in indices]
        if selected_records:
            save_qualitative_grid(
                selected_records,
                qualitative_dir / f"{name}.png",
                threshold=THRESHOLD,
                imagenet_normalized=imagenet_normalized,
                panel_size=panel_size,
            )
        selection_manifest[name] = [
            row_by_id[str(record["sample_id"])] for record in selected_records
        ]
    for index in unique_indices:
        record = records[index]
        save_qualitative_grid(
            [record],
            samples_dir / f"{record['sample_id']}.png",
            threshold=THRESHOLD,
            imagenet_normalized=imagenet_normalized,
            panel_size=panel_size,
        )
    return selection_manifest


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if checkpoint_path.name != "checkpoint-best.pth":
        raise ValueError(
            "Frozen test evaluation accepts checkpoint-best.pth only; got "
            f"{checkpoint_path.name}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.fixed_samples < 0 or args.ranked_samples < 0:
        raise ValueError("qualitative sample counts must be non-negative")
    if args.panel_size <= 0:
        raise ValueError("panel-size must be positive")

    prepare_output_dir(output_dir)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    checkpoint_hash_before = sha256(checkpoint_path)
    manifest_hash_before = sha256(manifest)
    model, checkpoint_metadata = build_model(checkpoint_path)
    model = model.to(device)

    dataset = MapOrthoBCENetDataset(
        manifest,
        split=SPLIT,
        crop_size=CROP_SIZE,
        train_jitter=0,
        augment=False,
        imagenet_normalize=checkpoint_metadata["imagenet_normalize"],
    )
    if len(dataset) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} untouched test samples, found {len(dataset)}"
        )
    unexpected_shapes = sorted(
        {
            (sample.height, sample.width)
            for sample in dataset.samples
            if (sample.height, sample.width) != (1024, 1024)
        }
    )
    if unexpected_shapes:
        raise ValueError(
            "The fixed center-crop protocol expects 1024x1024 sources, found "
            f"{unexpected_shapes}"
        )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    started_at = datetime.now(timezone.utc)
    metrics, rows, inference_seconds = evaluate(
        model=model,
        loader=loader,
        device=device,
        amp=args.amp,
    )
    macro_f1 = (float(metrics["omission"]["f1"]) + float(metrics["excess"]["f1"])) / 2.0

    sample_to_index = {
        sample.sample_id: index for index, sample in enumerate(dataset.samples)
    }
    selections = {
        "fixed": select_fixed_indices(dataset, args.fixed_samples),
        **ranked_selections(
            rows=rows,
            sample_to_index=sample_to_index,
            count=args.ranked_samples,
        ),
    }
    qualitative_manifest = write_qualitative_outputs(
        model=model,
        dataset=dataset,
        selections=selections,
        rows=rows,
        output_dir=output_dir,
        device=device,
        amp=args.amp,
        panel_size=args.panel_size,
        imagenet_normalized=checkpoint_metadata["imagenet_normalize"],
    )

    checkpoint_hash_after = sha256(checkpoint_path)
    manifest_hash_after = sha256(manifest)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("Checkpoint changed during evaluation")
    if manifest_hash_after != manifest_hash_before:
        raise RuntimeError("Manifest changed during evaluation")

    finished_at = datetime.now(timezone.utc)
    result = {
        "protocol": PROTOCOL,
        "scope": {
            "split": SPLIT,
            "sample_count": len(dataset),
            "source_shape": [1024, 1024],
            "evaluated_crop_shape": [CROP_SIZE, CROP_SIZE],
            "crop_origin": [256, 256],
            "pixels_per_sample": CROP_SIZE * CROP_SIZE,
            "total_evaluated_pixels": len(dataset) * CROP_SIZE * CROP_SIZE,
            "sliding_window": False,
            "directly_comparable_to_validation_center_crop": True,
        },
        "selection_policy": {
            "checkpoint_selected_by": "validation macro omission/excess F1",
            "test_used_for_model_selection": False,
            "test_used_for_threshold_selection": False,
            "threshold": THRESHOLD,
        },
        "model_description": "BCE-Net-based noisy-label robust baseline",
        "paper_exact_reproduction": False,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash_before,
            **checkpoint_metadata,
        },
        "manifest": {
            "path": str(manifest),
            "sha256": manifest_hash_before,
        },
        "runtime": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "inference_seconds": inference_seconds,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "amp": bool(args.amp and device.type == "cuda"),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "metrics": metrics,
        "macro_omission_excess_f1": macro_f1,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "per_sample_metrics.csv", rows)
    (output_dir / "qualitative" / "selections.json").write_text(
        json.dumps(qualitative_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Frozen test evaluation\n\n"
        "This directory contains the frozen `checkpoint-best.pth` evaluation on "
        "the 100-sample map-ortho test split. The threshold is fixed at 0.5.\n\n"
        "The protocol evaluates only the central 512×512 crop of each 1024×1024 "
        "source, matching validation. It is not a 1024 sliding-window/full-tile "
        "evaluation, and the two scopes must not be mixed.\n\n"
        "The evaluated model is a BCE-Net-based noisy-label robust baseline, not "
        "an exact reproduction of the paper. Test results were not used to select "
        "the model or threshold. File disjointness does not by itself prove "
        "spatial disjointness; run the split-overlap audit before interpreting "
        "this as spatial generalization.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
