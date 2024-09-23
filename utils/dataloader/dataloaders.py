import torch
import numpy as np
from .transforms import processer
from torch.utils.data import DataLoader
from .dataset import BaseDataset


def build_dataloaders(cfgs):
    train_dataset = BaseDataset(cfgs, "train", transform=processer)
    val_dataset = BaseDataset(cfgs, "val", transform=processer)
    test_dataset = BaseDataset(cfgs, "test", transform=processer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfgs["dataset"]["batch_size"],
        num_workers=cfgs["dataset"]["num_workers"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfgs["dataset"]["batch_size"],
        num_workers=cfgs["dataset"]["num_workers"],
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfgs["dataset"]["batch_size"],
        num_workers=cfgs["dataset"]["num_workers"],
        shuffle=False,
    )


    return train_loader, val_loader, test_loader