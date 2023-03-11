import random

import numpy as np
import torch
from torch import backends

# import pandas as pd

# def remove_duplicates():
#    dataset = ['train', 'val', 'test', 'ood']
#    for data in dataset:
#        proc = pd.read_csv(f"data/processed_{data}.csv")
#        proc_new = proc.drop_duplicates(ignore_index=True)
#        proc_new.to_csv(f"data/processed_{data}.csv", index=False)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    backends.cudnn.benchmark = True
    backends.cuda.matmul.allow_tf32 = True
    backends.cudnn.allow_tf32 = True
    backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision("medium")
