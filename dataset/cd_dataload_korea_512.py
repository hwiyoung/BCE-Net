"""Korea synthetic BCE-Net dataloader.

The dummy labels in this dataset are smoke-test placeholders, not ground truth for
metrics. BCE-Net expects item-level ``labels_o`` as [H, W], so DataLoader batches
produce [B, H, W].
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


class Mydataset(Dataset):
    def __init__(self, path: str, transform: Any | None = None, augment: bool = False, target_transform: Any | None = None):
        self.csv_path = Path(path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Korea manifest CSV does not exist: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        self.transform = transform
        self.target_transform = target_transform
        self.aug = augment
        required = [
            "tile_id",
            "image_path",
            "old_footprint_path",
            "dummy_new_path",
            "dummy_removed_path",
            "dummy_building_path",
            "dummy_change_path",
            "height",
            "width",
        ]
        missing = [column for column in required if column not in self.data.columns]
        if missing:
            raise ValueError(f"Korea manifest missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _check_path(path_value: str, role: str) -> Path:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Missing {role} file: {path}")
        return path

    @staticmethod
    def _read_image(path: Path, expected_h: int, expected_w: int) -> torch.Tensor:
        with rasterio.open(path) as src:
            if src.count < 3:
                raise ValueError(f"Image tile must have at least 3 bands, got {src.count}: {path}")
            arr = src.read(indexes=[1, 2, 3])
        if arr.shape != (3, expected_h, expected_w):
            raise ValueError(f"Image shape mismatch for {path}: got {arr.shape}, expected {(3, expected_h, expected_w)}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return torch.from_numpy(arr.astype(np.float32) / 255.0)

    @staticmethod
    def _read_mask(path: Path, expected_h: int, expected_w: int, role: str) -> torch.Tensor:
        with rasterio.open(path) as src:
            arr = src.read(1)
        if arr.shape != (expected_h, expected_w):
            raise ValueError(f"{role} shape mismatch for {path}: got {arr.shape}, expected {(expected_h, expected_w)}")
        arr = (arr > 0).astype(np.float32)
        return torch.from_numpy(arr)

    def __getitem__(self, item: int):
        row = self.data.iloc[item]
        tile_id = str(row["tile_id"])
        expected_h = int(row["height"])
        expected_w = int(row["width"])

        image = self._read_image(self._check_path(row["image_path"], "image"), expected_h, expected_w)
        labels_o = self._read_mask(self._check_path(row["old_footprint_path"], "old footprint"), expected_h, expected_w, "old footprint")
        labels_n = self._read_mask(self._check_path(row["dummy_new_path"], "dummy new label"), expected_h, expected_w, "dummy new")
        labels_m = self._read_mask(self._check_path(row["dummy_removed_path"], "dummy removed label"), expected_h, expected_w, "dummy removed")
        labels_b = self._read_mask(self._check_path(row["dummy_building_path"], "dummy building label"), expected_h, expected_w, "dummy building")
        labels = self._read_mask(self._check_path(row["dummy_change_path"], "dummy change label"), expected_h, expected_w, "dummy change")

        if image.dtype != torch.float32 or labels_o.dtype != torch.float32:
            raise TypeError(f"Unexpected tensor dtype for {tile_id}: image={image.dtype}, labels_o={labels_o.dtype}")
        if tuple(labels_o.shape) != (expected_h, expected_w):
            raise ValueError(f"labels_o must be item-level [H,W], got {tuple(labels_o.shape)} for {tile_id}")

        return image, labels_o, labels_n, labels_m, labels_b, labels, tile_id
