import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
#from torch import autocast
#from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
# from torch.distributed.optim import DistributedOptimizer as DOPTIM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy # size_based_auto_wrap_policy
import torch.multiprocessing as mp
from transformers import LayoutLMConfig, LayoutLMTokenizer, LayoutLMPreTrainedModel, logging
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed
# from datasets import load_metric
from model import LayoutLMForSequenceClassification
from functools import partial
from evaluation import evaluate_ood
import wandb
import warnings
from data import load
from sklearn.metrics import accuracy_score
# warnings.filterwarnings("ignore")
import os
import sys

task_to_labels = {
    'rvl_cdip': 16,
}

def wandb_log(args, results, num_steps):
    if args.log_all:
        wandb.log(results, step=num_steps)
    else:
        if args.rank == 0:
            wandb.log(results, step=num_steps)
        else:
            return

def train(args, model: FSDP, datasets, samplers, benchmarks):
    """
    Train method for the model (also facilitates validation and testing)

    Args:
        args: The parsed argument object with additional parameters/variables
              for the current rank
        model: The model relating to the current rank of our distributed model 
               architecture
        datasets: Tuple of Datasets for training, validation, and testing (400000, 40000, and 40000 images)
        samplers: Tuple of DistributedSamplers for buidling training, validation, and testing Dataloaders
        benchmarks: Tuple of tag (string) used for logging and out-of-distribution 
                    dataset for evaluating model effectiveness
    """
    train_dataset, dev_dataset, test_dataset = datasets
    train_sampler, dev_sampler, test_sampler = samplers
    # Use persistent_workers=True is issues with Distributed data pool
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, pin_memory=True, num_workers=8, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, sampler=dev_sampler, pin_memory=True, num_workers=8, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=test_sampler, pin_memory=True, num_workers=8, drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    #scaler = ShardedGradScaler()
    

    def detect_ood():
        model.prepare_ood(dev_dataloader, args.rank)
        for tag, ood_dataset, ood_sampler in benchmarks:
            results = evaluate_ood(args, model, test_dataset, ood_dataset, ood_sampler, tag=tag)
            wandb_log(args, results, num_steps)

    num_steps = 0
    # args.init_start_event.record()
    print(f"Beginning training for model {args.rank}")
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        train_dataloader.sampler.set_epoch(epoch)
        if args.rank == 0:
            inner_pbar = tqdm(range(len(train_loader)), colour="blue", desc="r0 Training Epoch")
        # for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch} (train):")):
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = {key: value.to(args.rank) for key, value in batch.items()}
            outputs = model(**batch)
            loss, cos_loss = outputs[0], outputs[1]
            #scaler.scale(loss).backward()
            loss.backward()
            num_steps += 1
            # scaler.step(optimizer)
            optimizer.step()
            scheduler.step()
            # scaler.update()
            model.zero_grad()
            wandb.log({'loss': loss.item()}, step=num_steps)
            wandb.log({'cos_loss': cos_loss.item()}, step=num_steps)
            if rank == 0:
                inner_pbar.update(1)

        results = evaluate(args, model, dev_dataloader, epoch, tag="dev")
        wandb_log(args, results, num_steps)
        results = evaluate(args, model, test_dataloader, epoch, tag="test")
        wandb_log(args, results, num_steps)
        detect_ood()
        
        if rank == 0:
            inner_pbar.close()
            print(f"Train Epoch: {epoch}, Accuracy: {results['accuracy']}")
    # args.init_end_event.record()


def evaluate(args, model, eval_dataloader, epoch, tag="train"):

    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = {}
        acc_score = accuracy_score(y_true=preds, y_pred=labels)
        # if len(result) > 1:
        #    result["score"] = np.mean(list(result.values())).item()
        result["accuracy"] = acc_score
        return result
    

    label_list, logit_list = [], []
    eval_dataloader.sampler.set_epoch(epoch)
    for step, batch in enumerate(tqdm(eval_dataloader, desc=f"Epoch {epoch} ({tag}):")):
        model.eval()
        # Keep on gpu instead of moving to cpu. "label" is an integer
        labels = batch["label"].detach() #cpu().numpy()
        batch = {key: value.to(args.rank) for key, value in batch.items()}
        batch["label"] = None
        outputs = model(**batch)
        logits = outputs[0].detach() #.cpu().numpy()
        label_list.append(labels)
        logit_list.append(logits)
    preds = torch.concatenate(logit_list, dim=0).cpu().numpy()
    labels = torch.concatenate(label_list, dim=0).cpu().numpy()

    results = compute_metrics(preds, labels)
    results = {"{}_{}".format(tag, key): value for key, value in results.items()}
    return results

def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'g3033.hyak.local'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if args.log_all:
        log = wandb.init(project=args.project_name, group="DDP", job_type="distributed log",
            name=args.task_name + '-' + str(args.alpha) + "_" + args.loss
        )
    else:
        if rank == 0:
            log = wandb.init(project=args.project_name, job_type="cumulative log",
                name=args.task_name + '-' + str(args.alpha) + "_" + args.loss
            )
        else:
            log = None
    return log

def cleanup():
    dist.destroy_process_group()


def fsdp_main(rank, world_size, args):
    args.log = setup(rank, world_size, args)

    args.rank = rank
    torch.cuda.set_device(rank)

    # args.init_start_event = torch.cuda.Event(enable_timing=True)
    # args.init_end_event = torch.cuda.Event(enable_timing=True)

    num_labels = task_to_labels[args.task_name]
    config = LayoutLMConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    config.gradient_checkpointing = True
    config.alpha = args.alpha
    config.loss = args.loss
    config.rank = rank

    print(f"Initializing DDP model {rank}")
    ddp_model = DDP(LayoutLMForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config,
    ).cuda(), device_ids=[rank], output_device=0)

    auto_wrap_policy = partial(transformer_auto_wrap_policy, 
        transformer_layer_cls={
            LayoutLMPreTrainedModel,
        },
    )

    # Shard model across GPUs
    print(f"Initializing FSDP model {rank}")
    fsdp_model = FSDP(ddp_model, 
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True), # offload params to CPU to save GPU memory and not break optimizer
        mixed_precision=True
     )
    

    if rank == 0:
        print(f"CUDA event elapsed time: {args.init_start_event.elapsed_time(args.init_end_event) / 1000}sec")
        print(f"{fsdp_model}")

    cleanup()

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

import ray
from dask.distributed import Client
from joblib import Parallel, delayed
from mpire import WorkerPool
from concurrent.futures import as_completed

from dask.distributed import progres

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="microsoft/layoutlm-base-uncased", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--task_name", default="rvl_cdip", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default="ood")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--loss", type=str, default="margin")
    parser.add_argument("--log_all", default=False, type=bool)
    args = parser.parse_args()

    datasets = ['rvl_cdip', 'ood']
    benchmarks = ()

    os.chdir("/mmfs1/gscratch/amath/wpm/")
    sys.path.append("/mmfs1/gscratch/amath/wpm/")

    tokenizer = LayoutLMTokenizer.from_pretrained(args.model_name_or_path)

    for dataset in datasets:
        if dataset == args.task_name:
            train_dataset, dev_dataset, test_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length, is_id=True)
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=66)
            dev_sampler = DistributedSampler(dev_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=66)
            test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=66)
        else:
            _, _, ood_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length)
            ood_sampler = DistributedSampler(ood_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=66)
            benchmarks = (('rvl_cdip_' + dataset, ood_dataset, ood_sampler),) + benchmarks
    train(args, ddp_model, (train_dataset, dev_dataset, test_dataset),
        (train_sampler, dev_sampler, test_sampler), benchmarks
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    #os.environ["CUDA_DEVICE_ORDER"] = "PCIE_BUS_ID"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9" # max_split_size_mb:512,
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    # logging.set_verbosity_error()

    WORLD_SIZE = torch.cuda.device_count()
    args.n_gpu = WORLD_SIZE
    set_seed(args)

    print(f"Spawning sharded model distribution (N={WORLD_SIZE})")
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
