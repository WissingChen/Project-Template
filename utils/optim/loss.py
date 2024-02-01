import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, pred, target, mask):
        # truncate to the same size
        target = target[:, :pred.size(1)]
        mask = mask[:, :pred.size(1)]
        output = -pred.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def compute_lm_loss(pred, target, mask):
    if isinstance(pred, list):
        pred = pred[0]
    criterion = LanguageModelCriterion()
    loss = criterion(pred, target[:, 1:], mask[:, 1:]).mean()
    return loss

def compute_ce_loss(pred, target, mask=None):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    loss = criterion(pred, target)
    return loss
