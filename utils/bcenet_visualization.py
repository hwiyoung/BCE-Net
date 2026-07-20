"""Qualitative panels and training curves for BCE-Net monitoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


STATE_COLORS = {
    "unchanged": np.array([55, 190, 90], dtype=np.uint8),
    "omission": np.array([255, 155, 35], dtype=np.uint8),
    "excess": np.array([220, 55, 210], dtype=np.uint8),
}


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _image_from_tensor(
    tensor: torch.Tensor, *, imagenet_normalized: bool
) -> np.ndarray:
    image = _as_numpy(tensor).transpose(1, 2, 0)
    if imagenet_normalized:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = image * std + mean
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _mask(value: torch.Tensor) -> np.ndarray:
    array = _as_numpy(value)
    return np.squeeze(array) > 0.5


def _probability(value: torch.Tensor) -> np.ndarray:
    return np.clip(np.squeeze(_as_numpy(value)), 0.0, 1.0)


def _blend_mask(
    image: np.ndarray,
    masks_and_colors: list[tuple[np.ndarray, np.ndarray]],
    *,
    alpha: float = 0.48,
) -> np.ndarray:
    overlay = image.copy()
    occupied = np.zeros(image.shape[:2], dtype=bool)
    for mask, color in masks_and_colors:
        overlay[mask] = color
        occupied |= mask
    result = image.copy()
    if occupied.any():
        result[occupied] = np.clip(
            (1.0 - alpha) * image[occupied] + alpha * overlay[occupied],
            0,
            255,
        ).astype(np.uint8)
    return result


def _probability_panel(image: np.ndarray, probability: np.ndarray) -> np.ndarray:
    heatmap_bgr = cv2.applyColorMap(
        np.round(probability * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 0.35, heatmap, 0.65, 0)


def _error_panel(
    image: np.ndarray,
    target_omission: np.ndarray,
    target_excess: np.ndarray,
    predicted_omission: np.ndarray,
    predicted_excess: np.ndarray,
) -> np.ndarray:
    target_change = target_omission | target_excess
    predicted_change = predicted_omission | predicted_excess
    correct_class = (
        (target_omission & predicted_omission)
        | (target_excess & predicted_excess)
    )
    wrong_class = target_change & predicted_change & ~correct_class
    false_positive = predicted_change & ~target_change
    false_negative = target_change & ~predicted_change
    return _blend_mask(
        image,
        [
            (correct_class, np.array([45, 210, 70], dtype=np.uint8)),
            (false_positive, np.array([240, 45, 45], dtype=np.uint8)),
            (false_negative, np.array([45, 100, 245], dtype=np.uint8)),
            (wrong_class, np.array([255, 220, 30], dtype=np.uint8)),
        ],
        alpha=0.65,
    )


def _titled(panel: np.ndarray, title: str, size: int) -> np.ndarray:
    resized = cv2.resize(panel, (size, size), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size + 30, size, 3), dtype=np.uint8)
    canvas[30:] = resized
    cv2.putText(
        canvas,
        title,
        (7, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return canvas


def qualitative_row(
    record: dict[str, Any],
    *,
    threshold: float,
    imagenet_normalized: bool,
    panel_size: int = 256,
) -> np.ndarray:
    image = _image_from_tensor(
        record["image"], imagenet_normalized=imagenet_normalized
    )
    reference = _mask(record["reference"])
    target_existing = _mask(record["target_existing"])
    target_omission = _mask(record["target_omission"])
    target_excess = _mask(record["target_excess"])
    existing_probability = _probability(record["existing_probability"])
    omission_probability = _probability(record["omission_probability"])
    excess_probability = _probability(record["excess_probability"])

    predicted_existing = existing_probability >= threshold
    predicted_omission = omission_probability >= threshold
    predicted_excess = excess_probability >= threshold
    target_unchanged = target_existing & reference & ~target_excess
    predicted_unchanged = predicted_existing & reference & ~predicted_excess

    reference_panel = _blend_mask(
        image,
        [(reference, np.array([35, 205, 235], dtype=np.uint8))],
    )
    target_panel = _blend_mask(
        image,
        [
            (target_unchanged, STATE_COLORS["unchanged"]),
            (target_omission, STATE_COLORS["omission"]),
            (target_excess, STATE_COLORS["excess"]),
        ],
    )
    predicted_panel = _blend_mask(
        image,
        [
            (predicted_unchanged, STATE_COLORS["unchanged"]),
            (predicted_omission, STATE_COLORS["omission"]),
            (predicted_excess, STATE_COLORS["excess"]),
        ],
    )
    error_panel = _error_panel(
        image,
        target_omission,
        target_excess,
        predicted_omission,
        predicted_excess,
    )

    sample_id = str(record["sample_id"])
    panels = [
        _titled(image, f"Before: {sample_id}", panel_size),
        _titled(reference_panel, "After footprint (cyan)", panel_size),
        _titled(target_panel, "GT: green/same orange/omit", panel_size),
        _titled(predicted_panel, f"Prediction @ {threshold:.2f}", panel_size),
        _titled(
            _probability_panel(image, omission_probability),
            "Omission probability",
            panel_size,
        ),
        _titled(
            _probability_panel(image, excess_probability),
            "Excess probability",
            panel_size,
        ),
        _titled(
            error_panel,
            "Error: TP/green FP/red FN/blue",
            panel_size,
        ),
    ]
    return np.concatenate(panels, axis=1)


def save_qualitative_grid(
    records: list[dict[str, Any]],
    path: str | Path,
    *,
    threshold: float,
    imagenet_normalized: bool,
    panel_size: int = 256,
) -> None:
    if not records:
        raise ValueError("At least one qualitative record is required")
    rows = [
        qualitative_row(
            record,
            threshold=threshold,
            imagenet_normalized=imagenet_normalized,
            panel_size=panel_size,
        )
        for record in records
    ]
    grid = np.concatenate(rows, axis=0)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write qualitative image to {destination}")


def save_training_curves(
    metrics_path: str | Path, output_path: str | Path
) -> None:
    metrics_file = Path(metrics_path)
    if not metrics_file.exists():
        return
    records = [
        json.loads(line)
        for line in metrics_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(record["epoch"]) + 1 for record in records]
    train_loss = [record["train"]["losses"]["loss"] for record in records]
    val_loss = [record["val"]["losses"]["loss"] for record in records]

    def metric(class_name: str, name: str) -> list[float]:
        return [
            float(record["val"]["metrics"][class_name][name])
            for record in records
        ]

    omission_f1 = metric("omission", "f1")
    excess_f1 = metric("excess", "f1")
    change_f1 = metric("change", "f1")
    macro_f1 = [
        (omission + excess) / 2.0
        for omission, excess in zip(omission_f1, excess_f1)
    ]

    figure, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    axes[0, 0].plot(epochs, train_loss, marker="o", label="train")
    axes[0, 0].plot(epochs, val_loss, marker="o", label="validation")
    axes[0, 0].set_title("Total loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, change_f1, marker="o", label="combined change")
    axes[0, 1].plot(epochs, macro_f1, marker="o", label="macro")
    axes[0, 1].plot(epochs, omission_f1, marker="o", label="omission")
    axes[0, 1].plot(epochs, excess_f1, marker="o", label="excess")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_title("Validation F1")
    axes[0, 1].legend()

    axes[1, 0].plot(
        epochs, metric("omission", "precision"), marker="o", label="precision"
    )
    axes[1, 0].plot(
        epochs, metric("omission", "recall"), marker="o", label="recall"
    )
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_title("Omission")
    axes[1, 0].legend()

    axes[1, 1].plot(
        epochs, metric("excess", "precision"), marker="o", label="precision"
    )
    axes[1, 1].plot(
        epochs, metric("excess", "recall"), marker="o", label="recall"
    )
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title("Excess")
    axes[1, 1].legend()

    for column, class_name in enumerate(("omission", "excess")):
        prediction_rate = metric(class_name, "prediction_rate")
        target_rate = metric(class_name, "target_rate")
        axes[2, column].plot(
            epochs, prediction_rate, marker="o", label="prediction rate"
        )
        axes[2, column].plot(
            epochs, target_rate, linestyle="--", label="target rate"
        )
        axes[2, column].set_ylim(bottom=0.0)
        axes[2, column].set_title(f"{class_name.title()} positive-pixel rate")
        axes[2, column].legend()
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140)
    plt.close(figure)
