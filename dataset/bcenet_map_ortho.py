"""Map/ortho training dataset for the custom BCE-Net task.

Each sample contains a 2020 (before) ortho image, a 2022 (after) map mask and
the following state label:

* 0: background
* 1: no change (building in both image and map)
* 2: omission (building in the image but absent from the map)
* 3: excess (building in the map but absent from the image)

This is already the state convention expected by BCE-Net.  The map input is
classes 1+3, the auxiliary existing-building target is classes 1+2, the new
head learns class 2 and the removed head learns class 3.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MapOrthoSample:
    sample_id: str
    image_path: Path
    map_mask_path: Path
    label_mask_path: Path
    split: str
    center_class_value: int
    width: int
    height: int


def read_manifest(path: str | Path, split: str) -> list[MapOrthoSample]:
    manifest = Path(path).resolve()
    samples: list[MapOrthoSample] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"].strip().lower() != split.lower():
                continue
            samples.append(
                MapOrthoSample(
                    sample_id=row["sample_id"],
                    image_path=Path(row["image_path"]),
                    map_mask_path=Path(row["map_mask_path"]),
                    label_mask_path=Path(row["label_mask_path"]),
                    split=split,
                    center_class_value=int(row["center_class_value"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
            )
    if not samples:
        raise ValueError(f"No rows for split {split!r} in {manifest}")
    return samples


def _read(path: Path, flag: int) -> np.ndarray:
    array = cv2.imread(str(path), flag)
    if array is None:
        raise FileNotFoundError(f"OpenCV could not read {path}")
    return array


def _boundary(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or not bool(mask.any()):
        return np.zeros(mask.shape, dtype=bool)
    kernel = np.ones((2 * width + 1, 2 * width + 1), dtype=np.uint8)
    return (
        cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    )


def _center_component(
    target: np.ndarray, center_y: int, center_x: int
) -> np.ndarray:
    """Return the target component nearest the known patch-center candidate."""

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        target.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros(target.shape, dtype=bool)
    direct = int(labels[np.clip(center_y, 0, target.shape[0] - 1), np.clip(
        center_x, 0, target.shape[1] - 1
    )])
    if direct > 0:
        selected = direct
    else:
        candidates = np.arange(1, count)
        distances = (
            (centroids[candidates, 0] - center_x) ** 2
            + (centroids[candidates, 1] - center_y) ** 2
        )
        selected = int(candidates[int(np.argmin(distances))])
    return labels == selected


def derive_targets_and_weights(
    map_mask: np.ndarray,
    state_label: np.ndarray,
    *,
    center_class_value: int,
    center_y: int,
    center_x: int,
    boundary_width: int = 2,
    boundary_weight: float = 0.25,
    secondary_change_weight: float = 0.5,
) -> dict[str, np.ndarray]:
    if map_mask.shape != state_label.shape:
        raise ValueError(
            f"map/label shape mismatch: {map_mask.shape} != {state_label.shape}"
        )
    values = set(int(value) for value in np.unique(state_label))
    if not values.issubset({0, 1, 2, 3}):
        raise ValueError(f"Expected state values 0/1/2/3, found {sorted(values)}")
    if center_class_value not in {2, 3}:
        raise ValueError(f"Center class must be 2 or 3, got {center_class_value}")
    if not 0.0 <= boundary_weight <= 1.0:
        raise ValueError("boundary_weight must be in [0, 1]")
    if not 0.0 <= secondary_change_weight <= 1.0:
        raise ValueError("secondary_change_weight must be in [0, 1]")

    reference = map_mask > 0
    no_change = state_label == 1
    omission = state_label == 2
    excess = state_label == 3
    existing = no_change | omission

    # These relations are required by the BCE-Net feature split.
    invalid = (omission & reference) | (excess & ~reference)
    invalid |= reference != (no_change | excess)

    common_boundary = (
        _boundary(reference, boundary_width)
        | _boundary(existing, boundary_width)
        | _boundary(omission, boundary_width)
        | _boundary(excess, boundary_width)
    )

    def base_weight() -> np.ndarray:
        weight = np.ones(reference.shape, dtype=np.float32)
        weight[common_boundary] = boundary_weight
        weight[invalid] = 0.0
        return weight

    weight_existing = base_weight()
    weight_new = base_weight()
    weight_removed = base_weight()

    center_target = state_label == center_class_value
    center_component = _center_component(center_target, center_y, center_x)
    all_changes = omission | excess
    secondary_changes = all_changes & ~center_component
    # The central candidate is the sampling anchor. Other change polygons remain
    # useful but are less trusted because the patch was not selected for them.
    weight_new[secondary_changes] = np.minimum(
        weight_new[secondary_changes], secondary_change_weight
    )
    weight_removed[secondary_changes] = np.minimum(
        weight_removed[secondary_changes], secondary_change_weight
    )

    return {
        "reference_mask": reference.astype(np.float32),
        "target_existing": existing.astype(np.float32),
        "target_new_head": omission.astype(np.float32),
        "target_removed_head": excess.astype(np.float32),
        "weight_existing": weight_existing,
        "weight_new_head": weight_new,
        "weight_removed_head": weight_removed,
        "invalid_mask": invalid.astype(np.float32),
        "center_component": center_component.astype(np.float32),
    }


class MapOrthoBCENetDataset(Dataset):
    """Central-crop dataset retaining each patch's sampled change candidate."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        split: str,
        crop_size: int = 512,
        train_jitter: int = 128,
        augment: bool = False,
        boundary_width: int = 2,
        boundary_weight: float = 0.25,
        secondary_change_weight: float = 0.5,
        imagenet_normalize: bool = False,
    ) -> None:
        self.samples = read_manifest(manifest, split)
        self.crop_size = crop_size
        self.train_jitter = train_jitter if split == "train" else 0
        self.augment = augment
        self.boundary_width = boundary_width
        self.boundary_weight = boundary_weight
        self.secondary_change_weight = secondary_change_weight
        self.imagenet_normalize = imagenet_normalize

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_origin(self, height: int, width: int) -> tuple[int, int]:
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(
                f"Input {height}x{width} is smaller than crop {self.crop_size}"
            )
        base_y = (height - self.crop_size) // 2
        base_x = (width - self.crop_size) // 2
        if self.train_jitter:
            base_y += random.randint(-self.train_jitter, self.train_jitter)
            base_x += random.randint(-self.train_jitter, self.train_jitter)
        return (
            int(np.clip(base_y, 0, height - self.crop_size)),
            int(np.clip(base_x, 0, width - self.crop_size)),
        )

    @staticmethod
    def _augment(
        image: np.ndarray,
        map_mask: np.ndarray,
        label: np.ndarray,
        center_y: int,
        center_x: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        height, width = label.shape
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            map_mask = np.flip(map_mask, axis=1)
            label = np.flip(label, axis=1)
            center_x = width - 1 - center_x
        if random.random() < 0.5:
            image = np.flip(image, axis=0)
            map_mask = np.flip(map_mask, axis=0)
            label = np.flip(label, axis=0)
            center_y = height - 1 - center_y
        rotations = random.randint(0, 3)
        for _ in range(rotations):
            image = np.rot90(image, 1)
            map_mask = np.rot90(map_mask, 1)
            label = np.rot90(label, 1)
            center_y, center_x = width - 1 - center_x, center_y
            height, width = width, height
        if random.random() < 0.5:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-18.0, 18.0)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(
                np.uint8
            )
        return (
            np.ascontiguousarray(image),
            np.ascontiguousarray(map_mask),
            np.ascontiguousarray(label),
            center_y,
            center_x,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.samples[index]
        image = _read(sample.image_path, cv2.IMREAD_COLOR)
        map_mask = _read(sample.map_mask_path, cv2.IMREAD_GRAYSCALE)
        label = _read(sample.label_mask_path, cv2.IMREAD_GRAYSCALE)
        if image.shape[:2] != map_mask.shape or map_mask.shape != label.shape:
            raise ValueError(
                f"{sample.sample_id}: image/map/label shapes differ: "
                f"{image.shape[:2]}, {map_mask.shape}, {label.shape}"
            )

        height, width = label.shape
        y, x = self._crop_origin(height, width)
        size = self.crop_size
        image = image[y : y + size, x : x + size]
        map_mask = map_mask[y : y + size, x : x + size]
        label = label[y : y + size, x : x + size]
        center_y, center_x = height // 2 - y, width // 2 - x

        # OpenCV uses BGR; the ImageNet-pretrained encoder expects RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augment:
            image, map_mask, label, center_y, center_x = self._augment(
                image, map_mask, label, center_y, center_x
            )

        arrays = derive_targets_and_weights(
            map_mask,
            label,
            center_class_value=sample.center_class_value,
            center_y=center_y,
            center_x=center_x,
            boundary_width=self.boundary_width,
            boundary_weight=self.boundary_weight,
            secondary_change_weight=self.secondary_change_weight,
        )
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        ).float() / 255.0
        if self.imagenet_normalize:
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            image_tensor = (image_tensor - mean) / std

        result: dict[str, torch.Tensor | str | int] = {
            "image": image_tensor,
            "sample_id": sample.sample_id,
            "crop_y": y,
            "crop_x": x,
        }
        for name, array in arrays.items():
            tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
            if name != "reference_mask":
                tensor = tensor.unsqueeze(0)
            result[name] = tensor
        return result


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
