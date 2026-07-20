#!/usr/bin/env python3
"""Train BCE-Net on before-ortho / after-map PNG patches."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate


REPO_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, REPO_ROOT / "DCNv2"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from dataset.bcenet_map_ortho import MapOrthoBCENetDataset, seed_worker
from utils.bcenet_loss import BCENetCriterion, LossConfig
from utils.bcenet_visualization import (
    save_qualitative_grid,
    save_training_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="dataset/map_ortho_manifest.csv"
    )
    parser.add_argument(
        "--output-dir",
        default="/home/work/models/BCE-Net/custom-map-ortho-robust",
    )
    parser.add_argument("--model-variant", choices=("whu", "sibu"), default="whu")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet-pretrained ResNet34 when no init checkpoint is supplied.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--train-jitter", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--pixel-loss", choices=("bce", "gce"), default="gce")
    parser.add_argument("--gce-q", type=float, default=0.7)
    parser.add_argument("--pixel-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-min-area", type=int, default=16)
    parser.add_argument("--positive-weight-existing", type=float, default=1.0)
    parser.add_argument("--positive-weight-new", type=float, default=4.0)
    parser.add_argument("--positive-weight-removed", type=float, default=4.0)
    parser.add_argument("--boundary-width", type=int, default=2)
    parser.add_argument("--boundary-weight", type=float, default=0.25)
    parser.add_argument("--secondary-change-weight", type=float, default=0.5)
    parser.add_argument(
        "--imagenet-normalize", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--best-metric",
        choices=("macro_change_f1", "combined_change_f1"),
        default="macro_change_f1",
    )
    parser.add_argument(
        "--visualize-every",
        type=int,
        default=5,
        help="Write fixed validation prediction panels every N epochs; 0 disables.",
    )
    parser.add_argument("--visualize-samples", type=int, default=4)
    parser.add_argument(
        "--collapse-patience",
        type=int,
        default=3,
        help="Stop after this many epochs with a zero-rate change head; 0 disables.",
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def torch_load(path: str | Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model_variant == "whu":
        from Testmodel.CDResWHU import Baseline34
    else:
        from Testmodel.CDResSIBU import Baseline34
    model = Baseline34(pretrained=args.pretrained and not args.init_checkpoint)
    if args.init_checkpoint:
        checkpoint = torch_load(args.init_checkpoint, "cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        result = model.load_state_dict(strip_module_prefix(state_dict), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
    return model


class BinaryMetrics:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.total = 0

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> None:
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
        }


def move_batch(
    batch: dict[str, torch.Tensor | str | int], device: torch.device
) -> dict[str, torch.Tensor | list[str] | list[int]]:
    result: dict[str, torch.Tensor | list[str] | list[int]] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device, non_blocking=True)
        else:
            result[key] = value
    return result


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: BCENetCriterion,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp: bool,
    threshold: float,
    log_every: int,
    max_batches: int | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    loss_sums: defaultdict[str, float] = defaultdict(float)
    batches = 0
    metrics = {
        "existing": BinaryMetrics(),
        "omission": BinaryMetrics(),
        "excess": BinaryMetrics(),
        "change": BinaryMetrics(),
    }
    started = time.perf_counter()

    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        image = batch["image"]
        reference = batch["reference_mask"]
        assert isinstance(image, torch.Tensor)
        assert isinstance(reference, torch.Tensor)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                outputs = model(image, reference)
                loss, details = criterion(outputs, batch)  # type: ignore[arg-type]
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batches += 1
        for key, value in details.items():
            loss_sums[key] += float(value.detach().item())

        existing_logit, removed_logit, new_logit = outputs[:3]
        existing_prediction = torch.sigmoid(existing_logit) >= threshold
        omission_prediction = torch.sigmoid(new_logit) >= threshold
        excess_prediction = torch.sigmoid(removed_logit) >= threshold
        target_existing = batch["target_existing"]
        target_omission = batch["target_new_head"]
        target_excess = batch["target_removed_head"]
        assert isinstance(target_existing, torch.Tensor)
        assert isinstance(target_omission, torch.Tensor)
        assert isinstance(target_excess, torch.Tensor)
        metrics["existing"].update(existing_prediction, target_existing > 0.5)
        metrics["omission"].update(omission_prediction, target_omission > 0.5)
        metrics["excess"].update(excess_prediction, target_excess > 0.5)
        metrics["change"].update(
            omission_prediction | excess_prediction,
            (target_omission > 0.5) | (target_excess > 0.5),
        )
        if log_every > 0 and (batch_index + 1) % log_every == 0:
            mode = "train" if training else "val"
            print(
                f"{mode} batch={batch_index + 1}/{len(loader)} "
                f"loss={loss_sums['loss'] / batches:.6f}",
                flush=True,
            )

    if batches == 0:
        raise RuntimeError("No batches were processed")
    result: dict[str, Any] = {
        "batches": batches,
        "seconds": time.perf_counter() - started,
        "losses": {key: value / batches for key, value in loss_sums.items()},
        "metrics": {key: value.compute() for key, value in metrics.items()},
    }
    return result


def make_loader(
    args: argparse.Namespace,
    *,
    split: str,
    generator: torch.Generator,
) -> DataLoader:
    dataset: torch.utils.data.Dataset = MapOrthoBCENetDataset(
        args.manifest,
        split=split,
        crop_size=args.crop_size,
        train_jitter=args.train_jitter,
        augment=split == "train",
        boundary_width=args.boundary_width,
        boundary_weight=args.boundary_weight,
        secondary_change_weight=args.secondary_change_weight,
        imagenet_normalize=args.imagenet_normalize,
    )
    limit = args.limit_train_samples if split == "train" else args.limit_val_samples
    if limit is not None:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=split == "train",
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def make_visual_dataset(args: argparse.Namespace) -> MapOrthoBCENetDataset:
    return MapOrthoBCENetDataset(
        args.manifest,
        split="val",
        crop_size=args.crop_size,
        train_jitter=0,
        augment=False,
        boundary_width=args.boundary_width,
        boundary_weight=args.boundary_weight,
        secondary_change_weight=args.secondary_change_weight,
        imagenet_normalize=args.imagenet_normalize,
    )


def select_visual_indices(
    dataset: MapOrthoBCENetDataset, count: int
) -> list[int]:
    if count <= 0:
        return []
    by_class = {
        class_value: [
            index
            for index, sample in enumerate(dataset.samples)
            if sample.center_class_value == class_value
        ]
        for class_value in (2, 3)
    }
    selected: list[int] = []
    offset = 0
    while len(selected) < min(count, len(dataset)):
        added = False
        for class_value in (2, 3):
            candidates = by_class[class_value]
            if offset < len(candidates) and len(selected) < count:
                selected.append(candidates[offset])
                added = True
        if not added:
            break
        offset += 1
    return selected


def write_qualitative_results(
    *,
    model: torch.nn.Module,
    dataset: MapOrthoBCENetDataset,
    indices: list[int],
    device: torch.device,
    amp: bool,
    threshold: float,
    imagenet_normalized: bool,
    output_path: Path,
) -> None:
    if not indices:
        return
    was_training = model.training
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in indices:
            raw_batch = default_collate([dataset[index]])
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
                existing, removed, new, _, _ = model(image, reference)
            sample_ids = raw_batch["sample_id"]
            records.append(
                {
                    "sample_id": sample_ids[0],
                    "image": raw_batch["image"][0],
                    "reference": raw_batch["reference_mask"][0],
                    "target_existing": raw_batch["target_existing"][0],
                    "target_omission": raw_batch["target_new_head"][0],
                    "target_excess": raw_batch["target_removed_head"][0],
                    "existing_probability": torch.sigmoid(existing[0]).cpu(),
                    "omission_probability": torch.sigmoid(new[0]).cpu(),
                    "excess_probability": torch.sigmoid(removed[0]).cpu(),
                }
            )
    save_qualitative_grid(
        records,
        output_path,
        threshold=threshold,
        imagenet_normalized=imagenet_normalized,
    )
    model.train(was_training)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_score: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            # Retained for compatibility with checkpoints from the first run.
            "best_f1": best_score,
            "args": vars(args),
        },
        path,
    )


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(args, split="train", generator=generator)
    val_loader = make_loader(args, split="val", generator=generator)
    visual_dataset = make_visual_dataset(args)
    visual_indices = select_visual_indices(
        visual_dataset, args.visualize_samples
    )
    model = build_model(args).to(device)
    criterion = BCENetCriterion(
        LossConfig(
            pixel_kind=args.pixel_loss,
            pixel_weight=args.pixel_loss_weight,
            dice_weight=args.dice_weight,
            contrastive_weight=args.contrastive_weight,
            gce_q=args.gce_q,
            contrastive_min_area=args.contrastive_min_area,
            positive_weight_existing=args.positive_weight_existing,
            positive_weight_new=args.positive_weight_new,
            positive_weight_removed=args.positive_weight_removed,
        )
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )

    start_epoch = 0
    best_score = -1.0
    if args.resume:
        checkpoint = torch_load(args.resume, device)
        model.load_state_dict(strip_module_prefix(checkpoint["state_dict"]), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler and checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(
            checkpoint.get("best_score", checkpoint.get("best_f1", -1.0))
        )

    print(
        json.dumps(
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )

    metrics_path = output_dir / "metrics.jsonl"
    if args.visualize_every > 0 and visual_indices:
        write_qualitative_results(
            model=model,
            dataset=visual_dataset,
            indices=visual_indices,
            device=device,
            amp=args.amp,
            threshold=args.threshold,
            imagenet_normalized=args.imagenet_normalize,
            output_path=(
                output_dir
                / "qualitative"
                / f"initial-epoch-{start_epoch:04d}.png"
            ),
        )

    collapse_streak = {"omission": 0, "excess": 0}
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        train_result = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp=args.amp,
            threshold=args.threshold,
            log_every=args.log_every,
            max_batches=args.max_train_batches,
        )
        with torch.no_grad():
            val_result = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=scaler,
                amp=args.amp,
                threshold=args.threshold,
                log_every=args.log_every,
                max_batches=args.max_val_batches,
            )
        if scheduler:
            scheduler.step()
        val_change_f1 = float(val_result["metrics"]["change"]["f1"])
        val_omission_f1 = float(val_result["metrics"]["omission"]["f1"])
        val_excess_f1 = float(val_result["metrics"]["excess"]["f1"])
        val_macro_f1 = (val_omission_f1 + val_excess_f1) / 2.0
        selection_score = (
            val_macro_f1
            if args.best_metric == "macro_change_f1"
            else val_change_f1
        )
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_started,
            "selection_metric": args.best_metric,
            "selection_score": selection_score,
            "val_macro_change_f1": val_macro_f1,
            "train": train_result,
            "val": val_result,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        save_checkpoint(
            output_dir / "checkpoint-last.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_score=max(best_score, selection_score),
            args=args,
        )
        if selection_score > best_score:
            best_score = selection_score
            save_checkpoint(
                output_dir / "checkpoint-best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                args=args,
            )
        save_training_curves(
            metrics_path, output_dir / "training-curves.png"
        )
        if (
            args.visualize_every > 0
            and visual_indices
            and (
                epoch == start_epoch
                or (epoch + 1) % args.visualize_every == 0
                or epoch + 1 == args.epochs
            )
        ):
            write_qualitative_results(
                model=model,
                dataset=visual_dataset,
                indices=visual_indices,
                device=device,
                amp=args.amp,
                threshold=args.threshold,
                imagenet_normalized=args.imagenet_normalize,
                output_path=(
                    output_dir
                    / "qualitative"
                    / f"epoch-{epoch + 1:04d}.png"
                ),
            )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_result['losses']['loss']:.6f} "
            f"val_loss={val_result['losses']['loss']:.6f} "
            f"val_change_f1={val_change_f1:.6f} "
            f"val_macro_f1={val_macro_f1:.6f} "
            f"omission_f1={val_omission_f1:.6f} "
            f"excess_f1={val_excess_f1:.6f} "
            f"best={best_score:.6f}",
            flush=True,
        )
        if args.collapse_patience > 0:
            for class_name in ("omission", "excess"):
                prediction_rate = float(
                    val_result["metrics"][class_name]["prediction_rate"]
                )
                if prediction_rate <= 1e-5:
                    collapse_streak[class_name] += 1
                    print(
                        f"WARNING: {class_name} prediction rate is "
                        f"{prediction_rate:.8f} "
                        f"({collapse_streak[class_name]}/"
                        f"{args.collapse_patience})",
                        flush=True,
                    )
                else:
                    collapse_streak[class_name] = 0
            collapsed = [
                name
                for name, streak in collapse_streak.items()
                if streak >= args.collapse_patience
            ]
            if collapsed:
                print(
                    "EARLY STOP: zero-rate class head detected: "
                    + ", ".join(collapsed),
                    flush=True,
                )
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
