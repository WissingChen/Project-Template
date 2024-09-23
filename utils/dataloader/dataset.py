import os.path as osp
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

class BaseDataset(Dataset):
    def __init__(self, cfgs, split, transform=None):
        self.data = MNIST('./data',train=True if split == "train" else False, download=False)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, label = self.data[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
