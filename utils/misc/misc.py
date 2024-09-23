import torch, random
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_remain_time(start_time, end_time, batch_idx, batch_count, epoch_idx, epoch_count):
    cost_time = end_time - start_time
    remain_time_batch = (batch_count - batch_idx) * cost_time
    remain_time_epoch = (epoch_count - epoch_idx) * cost_time + remain_time_batch
    return f'{cost_time:.2f}s', f'{remain_time_batch/60:.2f}m', f'{remain_time_epoch/3600:.2f}h'