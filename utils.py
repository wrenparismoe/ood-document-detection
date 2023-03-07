import torch
import random
import numpy as np

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


def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = []
        for i in l:
            new_dict[key] += i[key]
    return new_dict
