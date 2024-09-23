
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import label_binarize


class Metric(object):
    # TODO 根据cfgs选择
    def __init__(self, cfgs) -> None:
        self.scorer = {
            "Acc.":  accuracy_score,
            "AUC":  roc_auc_score,
            "Rec.":  recall_score,
            "Pre.":  precision_score,
            "F1":  f1_score
            }
    
    def get_score(self, target, pred, metric_name):
        target, pred, prob = self._check_type(target, pred)
        if metric_name == 'AUC':
            target = label_binarize(target, classes=[i for i in range(10)])
            score = self.scorer[metric_name](target, prob, multi_class='ovr', average='macro')
        else:
            try: 
                score = self.scorer[metric_name](target, pred)
            except:
                score = self.scorer[metric_name](target, pred, average='macro')
        return score

    def __call__(self, target, pred):
        # target, pred, prob = self._check_type(target, pred)
        score = {
            "Acc.": self.get_score(target, pred, "Acc."),
            "AUC":  self.get_score(target, pred, "AUC"),
            "Rec.": self.get_score(target, pred, "Rec."),
            "Pre.": self.get_score(target, pred, "Pre."),
            "F1":  self.get_score(target, pred, "F1"),
        }
        return score
    
    def _check_type(self, target, pred):
        prob = pred['prob']
        pred = pred['pred']
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(prob, torch.Tensor):
            prob = prob.detach().cpu().numpy()
        return target, pred, prob

