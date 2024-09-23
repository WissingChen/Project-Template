import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def compute_ce_loss(pred, target, mask=None):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    loss = criterion(pred, target)
    return loss

LOSS_FNS = {"ce": compute_ce_loss}

class BuildLossFunc():
    def __init__(self, cfgs):
        self.loss_fns = cfgs["optim"]["loss"]

    def __call__(self, pred, target):
        loss = {}
        total_loss = 0
        for loss_fn in self.loss_fns.keys():
            _loss = LOSS_FNS[loss_fn](pred, target)
            total_loss = total_loss + _loss * self.loss_fns[loss_fn]
            loss[f"{loss_fn}_loss"] = _loss
        loss["total_loss"] = total_loss
        return loss