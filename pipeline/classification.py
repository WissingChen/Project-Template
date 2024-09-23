from .base import BasePipeline
import torch
from torch import nn
from tabulate import tabulate
import time
from utils.misc import get_remain_time
import numpy as np

class ClassificationPipeline(BasePipeline):
    def __init__(self, cfgs):
        super().__init__(cfgs)
      
    def get_loss(self, batch):
        
        image, label = (
            batch[0].cuda(),
            batch[1].cuda(),
        )

        output = self.model(image)
        loss = self.criterion(output["logits"], label)

        monitor_metric = self.metric.get_score(label, output, self.cfgs['stat']['monitor']['metric'])

        return loss, monitor_metric

    def _train_epoch(self, epoch):
        
        self.monitor.log_info(f"\n{'#'*20} Epoch {epoch}/{self.cfgs['optim']['epochs']} ... {'#'*20}\n")
        #########################################
        # Train Stage
        #########################################
        print("Training Stage")
        self.model.train()
        # reset monitor
        self.monitor.reset_kv(f"running_{self.cfgs['stat']['monitor']['metric']}")
        self.monitor.reset_kv(f"running_total_loss")
        for key in self.cfgs["optim"]["loss"].keys():
            self.monitor.reset_kv(f"running_{key}_loss")
        # train iter
        for idx, batch in enumerate(self.train_dataloader):
            start_time = time.time()
            loss, monitor_metric = self.get_loss(batch)
            self.optimizer.zero_grad()
            loss["total_loss"].backward()
            if self.cfgs["optim"]["clip"]:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfgs["optim"]["clip"])
            self.optimizer.step()
            self.lr_scheduler.step()

            self.monitor.logkv_mean(f"running_{self.cfgs['stat']['monitor']['metric']}", monitor_metric)
            progress = float(idx + 1) / len(self.train_dataloader)

            end_time = time.time()
            ct, rbt, ret = get_remain_time(start_time, end_time, idx, len(self.train_dataloader), epoch, self.cfgs["optim"]['epochs'])
            print_info = f"Progress: {progress:3.2%} (ct-rbt-ret: {ct}-{rbt}-{ret}) || Train {self.cfgs['stat']['monitor']['metric']}: {monitor_metric:3.2%} ||"

            for _, (loss_fns, _loss) in enumerate(loss.items()):
                running_loss = _loss.detach().cpu().numpy()
                self.monitor.logkv_mean(f"running_{loss_fns}", running_loss)
                print_info += f" running_{loss_fns}: {running_loss:.4f} ||"

            # print info
            print(f'\r{print_info}', end='', flush=True)
        
        print_info = [[], []]
        for _, (name, val) in enumerate(self.monitor.name2val.items()):
            print_info[0].append(f"mean {name}")
            print_info[1].append(f"{val:.4f}")

        self.monitor.log_info(f'\n{tabulate(print_info, headers="firstrow", tablefmt="grid")}')

        #########################################
        # Eval Stage
        #########################################
        print("\nEval Stage")
        # val set
        val_score = self.eval(self.val_dataloader, epoch)
        # test set
        test_score = self.eval(self.test_dataloader, epoch)
        score = {}
        score.update({f"val_{k}": val_score[k] for k in val_score.keys()})
        score.update({f"test_{k}": test_score[k] for k in test_score.keys()})
        # visualization
        if self.cfgs['stat']['monitor']['vis']:
            self.monitor.plot_current_metrics(epoch, self.monitor.name2val)
        self.print_score(val_score, test_score)
        return score
    
    def print_score(self, val_score, test_score):
        def _print_score(_score):
            _info = [[],[]]
            for _, (name, val) in enumerate(_score.items()):
                _info[0].append(name)
                _info[1].append(f"{val:.2%}")
            self.monitor.log_info(f'\n{tabulate(_info, headers="firstrow", tablefmt="grid")}\n')

        self.monitor.log_info("\nVal Set:")
        _print_score(val_score)

        self.monitor.log_info("\nTest Set:")
        _print_score(test_score)
        
    def eval(self, data_loader, epoch=None, save=False):
        self.model.eval()

        outputs_prob = []
        outputs_pred = []
        labels = []

        score = {}
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                image, label = (
                    batch[0].cuda(),
                    batch[1].cuda(),
                )

                output = self.model(image)
                outputs_prob.extend(output['prob'].detach().cpu().numpy())
                outputs_pred.extend(output['pred'].detach().cpu().numpy())
                labels.extend(label.detach().cpu().numpy())

                progress = float(idx + 1) / len(data_loader)
                print(f"\rProgress: {progress:.2%}", end='', flush=True)


            # ---------------------------------------
            # Eval and Save the result
            # ---------------------------------------
            outputs_prob = np.array(outputs_prob)
            outputs_pred = np.array(outputs_pred)
            labels = np.array(labels)
            score = self.metric(labels, {"prob": outputs_prob, "pred": outputs_pred})
        return score
    
    def inference(self, epoch=None):
        count_param = sum(p.numel() for p in self.model.parameters())
        print(f"Model Params: {count_param}")
        print("\nEval Stage")
        # val set
        val_score = self.eval(self.val_dataloader, epoch, True if self.cfgs["dataset"]["name"] == "star" else False)
        # test set
        if self.cfgs["dataset"]["name"] == "star":
            test_score = val_score
        else:
            test_score = self.eval(self.test_dataloader, epoch, True)
        score = {}
        score.update({f"val_{k}": val_score[k] for k in val_score.keys()})
        score.update({f"test_{k}": test_score[k] for k in test_score.keys()})
        self.print_score(score)
        return score