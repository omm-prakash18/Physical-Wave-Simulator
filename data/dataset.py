"""
PyTorch Dataset for loading pre-generated wave simulation sequences.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WaveDataset(Dataset):
    """
    Loads pre-generated wave simulation sequences from .npz files.

    Each sample returns:
        context_frames: (T_in, C, H, W) — input context
        target_frames:  (T_out, C, H, W) — ground-truth future frames
        metadata:       dict with simulation parameters
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        t_in: int = 10,
        t_out: int = 10,
        augment: bool = True,
    ):
        self.data_dir = data_dir
        self.split = split
        self.t_in = t_in
        self.t_out = t_out
        self.augment = augment and (split == "train")

        # Load the file list for this split
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.files = sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".npz")
        ])

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files found in {split_dir}")

        # Load metadata if available
        meta_path = os.path.join(data_dir, "metadata.json")
        self.metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = np.load(self.files[idx], allow_pickle=True)
        frames = data["frames"]  # (T, C, H, W)
        params = data["params"].item() if "params" in data else {}

        # Scale frames to [-1, 1] if they are currently in [0, 1]
        if frames.min() >= 0.0 and frames.max() <= 1.0:
            frames = frames * 2.0 - 1.0

        # Split into context and target
        assert frames.shape[0] >= self.t_in + self.t_out, (
            f"Sequence length {frames.shape[0]} < t_in({self.t_in}) + t_out({self.t_out})"
        )
        context = frames[: self.t_in]
        target = frames[self.t_in : self.t_in + self.t_out]

        # Data augmentation (physics-invariant transforms)
        if self.augment:
            if np.random.random() > 0.5:
                context = context[:, :, ::-1, :].copy()  # vertical flip
                target = target[:, :, ::-1, :].copy()
            if np.random.random() > 0.5:
                context = context[:, :, :, ::-1].copy()  # horizontal flip
                target = target[:, :, :, ::-1].copy()

        context = torch.from_numpy(context).float()
        target = torch.from_numpy(target).float()

        return {
            "context": context,
            "target": target,
            "params": params,
            "index": idx,
        }


def create_dataloaders(
    data_dir: str,
    t_in: int = 10,
    t_out: int = 10,
    batch_size: int = 16,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to DataLoaders.
    """
    loaders = {}

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue

        dataset = WaveDataset(
            data_dir=data_dir,
            split=split,
            t_in=t_in,
            t_out=t_out,
            augment=(split == "train"),
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
            persistent_workers=num_workers > 0,
        )

    return loaders
