
from typing import Any
import numpy as np


class Metric(object):
    def __init__(self, cfgs) -> None:
        pass

    def __call__(self, pred, target):
        score = {
            "Acc":  0,
        }
        return score
