import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, cfgs):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(28**2, 14**2),
            nn.BatchNorm1d(14**2),
            nn.ReLU(),
            nn.Linear(14**2, 7**2),
            nn.BatchNorm1d(7**2),
            nn.ReLU(),
            nn.Linear(7**2, 10),
        )
    
    def forward(self, x):
        B = x.size(0)
        x = x.reshape([B, -1])
        logits = self.mlp(x)
        prob = logits.softmax(dim=-1)
        pred = prob.argmax(dim=-1)
        return {"logits": logits, "prob": prob, "pred": pred}