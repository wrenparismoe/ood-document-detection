import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import LayoutLMConfig, LayoutLMTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed
from datasets import load_metric
from model import LayoutLMForSequenceClassification
from evaluation import evaluate_ood
import wandb
import warnings
from data import load
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")


task_to_labels = {
    'rvl_cdip': 16,
}


def train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, drop_last=True)
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

    def detect_ood():
        model.prepare_ood(dev_dataloader)
        for tag, ood_features in benchmarks:
            results = evaluate_ood(args, model, test_dataset, ood_features, tag=tag)
            wandb.log(results, step=num_steps)

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            batch = {key: value.to(args.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss, cos_loss = outputs[0], outputs[1]
            loss.backward()
            num_steps += 1
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            wandb.log({'loss': loss.item()}, step=num_steps)
            wandb.log({'cos_loss': cos_loss.item()}, step=num_steps)

        results = evaluate(args, model, dev_dataset, tag="dev")
        wandb.log(results, step=num_steps)
        results = evaluate(args, model, test_dataset, tag="test")
        wandb.log(results, step=num_steps)
        detect_ood()


def evaluate(args, model, eval_dataset, tag="train"):

    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = {}
        acc_score = accuracy_score(y_true=preds, y_pred=labels)
        # if len(result) > 1:
        #    result["score"] = np.mean(list(result.values())).item()
        result["accuracy"] = acc_score
        return result
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, drop_last=True)

    label_list, logit_list = [], []
    for step, batch in enumerate(tqdm(dataloader)):
        model.eval()
        labels = batch["label"].detach().cpu().numpy()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        batch["label"] = None
        outputs = model(**batch)
        logits = outputs[0].detach().cpu().numpy()
        label_list.append(labels)
        logit_list.append(logits)
    preds = np.concatenate(logit_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    results = compute_metrics(preds, labels)
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
    args = parser.parse_args()

    wandb.init(project=args.project_name, name=args.task_name + '-' + str(args.alpha) + "_" + args.loss)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    num_labels = task_to_labels[args.task_name]
    config = LayoutLMConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
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

    for dataset in datasets:
        if dataset == args.task_name:
            train_dataset, dev_dataset, test_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length, is_id=True)
        else:
            _, _, ood_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length)
            benchmarks = (('rvl_cdip_' + dataset, ood_dataset),) + benchmarks
    train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks)


if __name__ == "__main__":
    main()


