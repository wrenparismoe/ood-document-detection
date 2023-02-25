import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch import autocast
from torch.cuda.amp import GradScaler
from transformers import LayoutLMConfig, LayoutLMTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed
from datasets import load_metric
from model import LayoutLMForSequenceClassification
from evaluation import evaluate_ood
import wandb
#wandb.init(project="ood", )
import warnings
from data import load
from sklearn.metrics import accuracy_score
import os
import gc
warnings.filterwarnings("ignore")

# 2419435459658b249d1a54abb6760d498b974b47

task_to_labels = {
    'rvl_cdip': 16,
}


def train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, num_workers=4)
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
    scaler = GradScaler()

    def detect_ood(rank=3):
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        model.cuda(device)
        model.prepare_ood(dev_dataloader, rank=rank)
        for tag, ood_features in benchmarks:
            results = evaluate_ood(args, model, test_dataset, ood_features, tag=tag, rank=rank)
            wandb.log(results, step=num_steps)

    #wandb.watch(model)

    print("Begginning training loop")
    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        model.cuda(device)
        model.zero_grad()
        #for param in model.parameters():
        #    param.grad = None
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Train (epoch {epoch})", postfix={'dataset': 'train'})):
            model.train()
            with autocast(device_type='cuda', dtype=torch.float16):
                batch = {key: value.cuda(device) for key, value in batch.items()}
                outputs = model(**batch)
            loss, cos_loss = outputs[0], outputs[1]
            scaler.scale(loss).backward()
            #loss.backward()
            num_steps += 1
            scaler.step(optimizer)
            #optimizer.step()
            scheduler.step()
            scaler.update()
            model.zero_grad()
            #for param in model.parameters():
            #    param.grad = None
            if num_steps % (args.batch_size/2):
                wandb.log({'loss': loss.item()}, step=num_steps)
                wandb.log({'cos_loss': cos_loss.item()}, step=num_steps)

        results = evaluate(args, model, dev_dataset, epoch, tag="dev", rank=1)
        wandb.log(results, step=num_steps)
        results = evaluate(args, model, test_dataset, epoch, tag="test", rank=2)
        wandb.log(results, step=num_steps)
        detect_ood(rank=3)


def evaluate(args, model, eval_dataset, epoch, tag="train", rank=0):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    model.cuda(device)

    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = {}
        acc_score = accuracy_score(y_true=preds, y_pred=labels)
        # if len(result) > 1:
        #    result["score"] = np.mean(list(result.values())).item()
        result["accuracy"] = acc_score
        return result

    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, num_workers=4)
    
    label_list, logit_list = [], []
    for step, batch in enumerate(tqdm(dataloader, desc=f"Evaluate (epoch {epoch})", postfix={'dataset': tag})):
        model.eval()
        with autocast(device_type='cuda', dtype=torch.float16):
            labels = batch["label"].detach() #.cpu().numpy()
            batch = {key: value.cuda(device) for key, value in batch.items()}
            batch["label"] = None
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs[0].detach() #.cpu().numpy()
        label_list.append(labels)
        logit_list.append(logits)
        del labels, batch, outputs, logits
    preds = torch.concatenate(logit_list, axis=0).cpu().numpy()
    labels = torch.concatenate(label_list, axis=0).cpu().numpy()

    results = compute_metrics(preds, labels)
    del preds, labels, label_list, logit_list
    gc.collect()
    torch.cuda.empty_cache()
    results = {"{}_{}".format(tag, key): value for key, value in results.items()}
    return results


def main():
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
    parser.add_argument("--create_pickles", type=bool, default=False)
    args = parser.parse_args()

    wandb.init(project=args.project_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    num_labels = task_to_labels[args.task_name]
    config = LayoutLMConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels) 
         # Config params: hidden_act="new_gelu", num_hidden_layers=10 (12), hidden_size=(768), intermediate_size=(3072)  
         # from_pretrained() params: torch_dtype, device_map, max_memory, load_in_8bit
    config.gradient_checkpointing = True
    config.alpha = args.alpha
    config.loss = args.loss
    tokenizer = LayoutLMTokenizer.from_pretrained(args.model_name_or_path)
    model = LayoutLMForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config,
    )
    model.to(device)

    datasets = ['rvl_cdip', 'ood']
    benchmarks = ()

    if args.create_pickles:
        for dataset in datasets:
            if dataset == args.task_name:
                train_dataset, dev_dataset, test_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length, is_id=True)
                torch.save(train_dataset, '/mmfs1/gscratch/amath/wpm/data/train_dataset.pt')
                torch.save(dev_dataset, '/mmfs1/gscratch/amath/wpm/data/dev_dataset.pt')
                torch.save(test_dataset, '/mmfs1/gscratch/amath/wpm/data/test_dataset.pt')
            else:
                _, _, ood_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length)
                torch.save(ood_dataset, '/mmfs1/gscratch/amath/wpm/data/ood_dataset.pt')
                benchmarks = (('rvl_cdip_' + dataset, ood_dataset),) + benchmarks
    else:
        #train_dataset = torch.load('/mmfs1/gscratch/amath/wpm/data/train_dataset.pt')
        #dev_dataset = torch.load('/mmfs1/gscratch/amath/wpm/data/dev_dataset.pt')
        #test_dataset = torch.load('/mmfs1/gscratch/amath/wpm/data/test_dataset.pt')
        #ood_dataset = torch.load('/mmfs1/gscratch/amath/wpm/data/ood_dataset.pt')
        
        for dataset in datasets:
            if dataset == args.task_name:
                train_dataset, dev_dataset, test_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length, is_id=True)
            else:
                _, _, ood_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length)
                benchmarks = (('rvl_cdip_ood', ood_dataset),) + benchmarks
                #benchmarks=None

        train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks)


if __name__ == "__main__":
    #sys.path.append('/mmfs1/gscratch/amath/wpm')
    #os.chdir('/mmfs1/gscratch/amath/wpm')
    os.chdir('/tmp/wpm')
    #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128,garbage_collection_threshold:0.66"
    
    #torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    main()


