import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .dataset import BaseDataset


def build_dataloaders(cfgs, a2id, tokenizer):
    train_dataset = BaseDataset(cfgs, "train")
    val_dataset = BaseDataset(cfgs, "val")
    test_dataset = BaseDataset(cfgs, "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfgs["dataset"]["batch_size"],
        num_workers=cfgs["dataset"]["num_thread_reader"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfgs["dataset"]["batch_size"],
        num_workers=cfgs["dataset"]["num_thread_reader"],
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfgs["dataset"]["batch_size"],
        num_workers=cfgs["dataset"]["num_thread_reader"],
        shuffle=False,
    )


    return train_loader, val_loader, test_loader