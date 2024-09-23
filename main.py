import argparse
from utils.misc import setup_seed, load_file
import numpy as np
import yaml
from pipeline import *


def main(args):
    # -------------------------------
    # load hyper-param
    # -------------------------------
    cfgs = load_file(args.cfg)
    if args.cuda != "":
        cfgs["misc"]["cuda"] = args.cuda
    if args.R != "":
        cfgs["misc"]["running_name"] = args.R
    # -------------------------------
    # fix random seeds
    # -------------------------------
    if cfgs["misc"]["seed"] == -1:
        cfgs["misc"]["seed"] = np.random.randint(0, 23333)
    setup_seed(cfgs["misc"]["seed"])
    print(cfgs)
    # -------------------------------
    # Run!
    # -------------------------------
    pipeline = pipeline_fns[cfgs['optim']['pipeline']](cfgs)
    pipeline.train()
    
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    # 选用哪个配置文件
    parse.add_argument("--cfg", type=str, default="config/example.yml")
    # 会优先使用由终端指定的cuda
    parse.add_argument("--cuda", type=str, default="2")
    # 会优先使用由终端指定的运行名字，用于保存实验记录
    parse.add_argument("-R", type=str, default="")
    args = parse.parse_args()

    main(args)
    