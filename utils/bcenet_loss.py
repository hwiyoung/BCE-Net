"""Paper-style and noisy-label-aware losses for BCE-Net."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_shapes(
    logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> None:
    if logits.shape != target.shape or target.shape != weight.shape:
        raise ValueError(
            "logits/target/weight shapes must match: "
            f"{logits.shape}, {target.shape}, {weight.shape}"
        )


def weighted_pixel_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    kind: str,
    gce_q: float,
    positive_weight: float,
) -> torch.Tensor:
    _validate_shapes(logits, target, weight)
    if not bool((weight > 0).any()):
        return logits.sum() * 0.0
    if kind == "bce":
        raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    elif kind == "gce":
        if not 0.0 < gce_q <= 1.0:
            raise ValueError("gce_q must be in (0, 1]")
        probability = torch.sigmoid(logits)
        target_probability = (
            target * probability + (1.0 - target) * (1.0 - probability)
        )
        raw = (1.0 - target_probability.clamp_min(1e-7).pow(gce_q)) / gce_q
    else:
        raise ValueError(f"Unknown pixel loss: {kind}")
    if positive_weight <= 0.0:
        raise ValueError("positive_weight must be greater than zero")
    class_weight = 1.0 + target * (positive_weight - 1.0)
    effective_weight = weight * class_weight
    return (raw * effective_weight).sum() / effective_weight.sum().clamp_min(1.0)


def weighted_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    _validate_shapes(logits, target, weight)
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, logits.ndim))
    valid = weight.sum(dim=dims) > 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    intersection = (probability * target * weight).sum(dim=dims)
    denominator = (probability * weight).sum(dim=dims) + (target * weight).sum(
        dim=dims
    )
    score = (2.0 * intersection + smooth) / (denominator + smooth)
    return (1.0 - score[valid]).mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    pixel_kind: str,
    pixel_weight: float,
    dice_weight: float,
    gce_q: float,
    positive_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pixel = weighted_pixel_loss(
        logits,
        target,
        weight,
        kind=pixel_kind,
        gce_q=gce_q,
        positive_weight=positive_weight,
    )
    dice = weighted_dice_loss(logits, target, weight)
    return pixel_weight * pixel + dice_weight * dice, pixel, dice


def _instance_terms(
    feature_a: torch.Tensor,
    feature_b: torch.Tensor,
    target: torch.Tensor,
    quality: torch.Tensor,
    *,
    similar: bool,
    min_area: int,
    trusted_fraction: float,
) -> list[torch.Tensor]:
    mask = (target.detach().cpu().numpy() > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    terms: list[torch.Tensor] = []
    for component_id in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[component_id])
        if area < min_area:
            continue
        component_np = labels[y : y + height, x : x + width] == component_id
        quality_np = quality[y : y + height, x : x + width].detach().cpu().numpy()
        if float(quality_np[component_np].mean()) < trusted_fraction:
            continue
        component = torch.from_numpy(component_np).to(
            device=feature_a.device, dtype=torch.bool
        )
        a = torch.sigmoid(feature_a[:, y : y + height, x : x + width])
        b = torch.sigmoid(feature_b[:, y : y + height, x : x + width])
        selector = component.unsqueeze(0).expand_as(a)
        a_vector = a[selector].reshape(1, -1)
        b_vector = b[selector].reshape(1, -1)
        if a_vector.numel() == 0:
            continue
        similarity = F.cosine_similarity(a_vector, b_vector, dim=1).squeeze(0)
        terms.append(
            0.5 * ((1.0 - similarity) if similar else (1.0 + similarity))
        )
    return terms


def instance_contrastive_loss(
    feature_all: torch.Tensor,
    feature_split: torch.Tensor,
    target_new: torch.Tensor,
    target_removed: torch.Tensor,
    weight_new: torch.Tensor,
    weight_removed: torch.Tensor,
    *,
    min_area: int,
    trusted_fraction: float,
) -> torch.Tensor:
    """Instance-constrained contrastive term from the BCE-Net paper.

    Omission/new instances are pulled toward the all-building representation.
    Excess/removed instances are pushed away.
    """

    if feature_all.shape != feature_split.shape:
        raise ValueError(
            f"Contrastive feature shapes differ: "
            f"{feature_all.shape} != {feature_split.shape}"
        )
    terms: list[torch.Tensor] = []
    for index in range(feature_all.shape[0]):
        terms.extend(
            _instance_terms(
                feature_all[index],
                feature_split[index],
                target_new[index, 0],
                weight_new[index, 0],
                similar=True,
                min_area=min_area,
                trusted_fraction=trusted_fraction,
            )
        )
        terms.extend(
            _instance_terms(
                feature_all[index],
                feature_split[index],
                target_removed[index, 0],
                weight_removed[index, 0],
                similar=False,
                min_area=min_area,
                trusted_fraction=trusted_fraction,
            )
        )
    if not terms:
        return feature_all.sum() * 0.0
    return torch.stack(terms).mean()


@dataclass(frozen=True)
class LossConfig:
    pixel_kind: str = "bce"
    pixel_weight: float = 1.0
    dice_weight: float = 1.0
    contrastive_weight: float = 1.0
    gce_q: float = 0.7
    contrastive_min_area: int = 16
    contrastive_trusted_fraction: float = 0.5
    positive_weight_existing: float = 1.0
    positive_weight_new: float = 4.0
    positive_weight_removed: float = 4.0


class BCENetCriterion(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(outputs) != 5:
            raise ValueError(f"Expected five BCE-Net outputs, got {len(outputs)}")
        existing, removed, new, feature_all, feature_split = outputs
        cfg = self.config
        existing_total, existing_pixel, existing_dice = segmentation_loss(
            existing,
            batch["target_existing"],
            batch["weight_existing"],
            pixel_kind=cfg.pixel_kind,
            pixel_weight=cfg.pixel_weight,
            dice_weight=cfg.dice_weight,
            gce_q=cfg.gce_q,
            positive_weight=cfg.positive_weight_existing,
        )
        removed_total, removed_pixel, removed_dice = segmentation_loss(
            removed,
            batch["target_removed_head"],
            batch["weight_removed_head"],
            pixel_kind=cfg.pixel_kind,
            pixel_weight=cfg.pixel_weight,
            dice_weight=cfg.dice_weight,
            gce_q=cfg.gce_q,
            positive_weight=cfg.positive_weight_removed,
        )
        new_total, new_pixel, new_dice = segmentation_loss(
            new,
            batch["target_new_head"],
            batch["weight_new_head"],
            pixel_kind=cfg.pixel_kind,
            pixel_weight=cfg.pixel_weight,
            dice_weight=cfg.dice_weight,
            gce_q=cfg.gce_q,
            positive_weight=cfg.positive_weight_new,
        )
        contrastive = instance_contrastive_loss(
            feature_all,
            feature_split,
            batch["target_new_head"],
            batch["target_removed_head"],
            batch["weight_new_head"],
            batch["weight_removed_head"],
            min_area=cfg.contrastive_min_area,
            trusted_fraction=cfg.contrastive_trusted_fraction,
        )
        total = (
            existing_total
            + removed_total
            + new_total
            + cfg.contrastive_weight * contrastive
        )
        return total, {
            "loss": total,
            "existing": existing_total,
            "removed": removed_total,
            "new": new_total,
            "contrastive": contrastive,
            "existing_pixel": existing_pixel,
            "existing_dice": existing_dice,
            "removed_pixel": removed_pixel,
            "removed_dice": removed_dice,
            "new_pixel": new_pixel,
            "new_dice": new_dice,
        }
