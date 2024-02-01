import os.path as osp
import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random


class BaseDataset(Dataset):
    def __init__(self, cfgs, split, transform=None):
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self):
        pass
