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
from model import LayoutLMForSequenceClassification
from evaluation import evaluate_ood
import wandb
import warnings
from data import load
from sklearn.metrics import accuracy_score
import os
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from control import LayoutLMModule, LayoutLMCallback
from data import DataModule


warnings.filterwarnings("ignore")


def train(args, model, train_dataloader, dev_dataset, test_dataset, benchmarks):
    total_steps = int(len(train_dataloader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    # Try torch.optim.lr_scheduler.CyclicLR() or torch.optim.lr_scheduler.OneCycleLR() lr scheduler
    scaler = GradScaler()

    def detect_ood(dev_dataloader, rank=3):
        model.prepare_ood(dev_dataloader, rank=rank)
        for tag, ood_features in benchmarks:
            results = evaluate_ood(
                args, model, test_dataset, ood_features, tag=tag, rank=rank
            )
            wandb.log(results, step=num_steps)

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad(set_to_none=True)
        for step, batch in enumerate(  # PL: train_dataloader() (#LightningDataModule)
            tqdm(
                train_dataloader,
                desc=f"Train (epoch {epoch})",
                postfix={"dataset": "train"},
                mininterval=2,
            ),
        ):
            batch = {key: value for key, value in batch.items()}
            outputs = model(**batch)
            loss, cos_loss = outputs[0], outputs[1]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            model.zero_grad(set_to_none=True)
            wandb.log({"loss": loss.item()}, step=num_steps)
            wandb.log({"cos_loss": cos_loss.item()}, step=num_steps)
            num_steps += 1

        evaluate(args, model, dev_dataset, epoch, tag="dev", rank=0)
        evaluate(args, model, test_dataset, epoch, tag="test", rank=0)


def test(model, test_dataset, dev_dataloader, benchmarks, num_steps, rank=3):
    model.prepare_ood(dev_dataloader, rank=rank)
    for tag, ood_features in benchmarks:
        results = evaluate_ood(
            args, model, test_dataset, ood_features, tag=tag, rank=rank
        )
        wandb.log(results, step=num_steps)


def evaluate(
    args, model, eval_dataset, epoch, dataloader, num_steps, tag="train", rank=0
):
    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = {}
        acc_score = accuracy_score(y_true=preds, y_pred=labels)
        result["accuracy"] = acc_score
        return result

    label_list, logit_list = [], []  # PL: val_dataloader() (#LightningDataModule)
    for step, batch in enumerate(
        tqdm(
            dataloader,
            desc=f"Evaluate (epoch {epoch})",
            postfix={"dataset": tag},
            mininterval=2,
        )
    ):
        labels = batch["label"].detach()  # .cpu().numpy()
        batch = {key: value for key, value in batch.items()}
        batch["label"] = None
        outputs = model(**batch)
        logits = outputs[0].detach()  # .cpu().numpy()
        label_list.append(labels)
        logit_list.append(logits)

    preds = torch.concatenate(logit_list, axis=0).cpu().numpy()
    labels = torch.concatenate(label_list, axis=0).cpu().numpy()

    # PL: EvalResult.log(on_epoch=True) (#LightningModule)
    results = compute_metrics(preds, labels)
    results = {"{}_{}".format(tag, key): value for key, value in results.items()}
    wandb.log(results, step=num_steps)
    # PL: EvalResult(checkpoint_on=X, early_stop_on=X) (#LightningModule)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", default="microsoft/layoutlm-base-uncased", type=str
    )
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

    set_seed(args)

    model = LayoutLMModule(args)
    data = DataModule(args)
    wandb_logger = WandbLogger(project=args.project_name, log_model="all")
    wandb_logger.watch(model, log="all")
    trainer = Trainer(logger=wandb_logger, max_epochs=args.num_train_epochs)
    trainer.fit(model, data)

    num_labels = task_to_labels[args.task_name]
    config = LayoutLMConfig.from_pretrained(
        args.model_name_or_path, num_labels=num_labels
    )
    # Config params: hidden_act="gelu_new", num_hidden_layers=10 (12), hidden_size=(768), intermediate_size=(3072)
    # from_pretrained() params: torch_dtype, device_map, max_memory, load_in_8bit
    config.gradient_checkpointing = True
    config.hidden_act = "gelu_new"  # "gelu_new"
    config.alpha = args.alpha
    config.loss = args.loss
    tokenizer = LayoutLMTokenizer.from_pretrained(args.model_name_or_path)
    model = LayoutLMForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    datasets = ["rvl_cdip", "ood"]
    benchmarks = ()

    for dataset in datasets:
        if dataset == args.task_name:
            train_dataset, dev_dataset, test_dataset = load(
                dataset, tokenizer, max_seq_length=args.max_seq_length, is_id=True
            )
            wandb.log(
                {
                    "train_size": train_dataset.num_rows,
                    "val_size": dev_dataset.num_rows,
                    "test_size": test_dataset.num_rows,
                }
            )
        else:
            _, _, ood_dataset = load(
                dataset, tokenizer, max_seq_length=args.max_seq_length
            )
            wandb.log({"ood_size": ood_dataset.num_rows})
            benchmarks = (("rvl_cdip_ood", ood_dataset),) + benchmarks

    train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks)

    ################## New Run Sequence Using PyTorch Lightning ##################


if __name__ == "__main__":
    # sys.path.append('/mmfs1/gscratch/amath/wpm')
    # os.chdir('/mmfs1/gscratch/amath/wpm')
    os.chdir("/tmp/wpm")
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512,garbage_collection_threshold:0.82"

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    main()
