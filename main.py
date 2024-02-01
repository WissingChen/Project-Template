import torch
import numpy as np
import random 
import argparse
import yaml
from pipeline import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # -------------------------------
    # load hyper-param
    # -------------------------------
    cfgs = yaml.load(open(args.cfg), Loader=yaml.FullLoader)
    if args.cuda != "":
        cfgs["misc"]["cuda"] = args.cuda
    # -------------------------------
    # fix random seeds
    # -------------------------------
    if cfgs["misc"]["seed"] == -1:
        cfgs["misc"]["seed"] = np.random.randint(0, 23333)
    setup_seed(cfgs["misc"]["seed"])
    print(cfgs)
    
    pipeline = pipeline_fns["causalgqa"](cfgs)
    pipeline.train()
    
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-cfg", type=str, default="config/causalgqa.yml")
    parse.add_argument("-cuda", type=str, default="")
    args = parse.parse_args()

    main(args)
    