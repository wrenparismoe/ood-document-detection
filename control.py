import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule, Callback
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from transformers import LayoutLMConfig
from utils import merge_keys
from model import LayoutLMForSequenceClassification
import numpy as np

task_to_labels = {
    "rvl_cdip": 16,
}


class LayoutLMModule(LightningModule):  # LightningModule
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Create model config and load pretraiend model
        config = LayoutLMConfig.from_pretrained(args.model_name_or_path)
        config.num_labels = task_to_labels[args.task_name]
        config.gradient_checkpointing = True
        config.hidden_act = "gelu_new"
        config.alpha = args.alpha
        config.loss = args.loss
        # self.log({"batch_size": args.batch_size, "learning_rate": args.learning_rate, "epochs": args.num_train_epochs,},),
        self.save_hyperparameters()
        self.model = LayoutLMForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=config
        )

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    @property
    def steps_per_epoch(self) -> int:
        return self.num_training_steps // self.trainer.max_epochs

    def configure_optimizers(self):
        warmup_steps = int(self.num_training_steps * self.args.warmup_ratio)

        no_decay = ["LayerNorm.weight", "bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        # Try torch.optim.lr_scheduler.CyclicLR() or torch.optim.lr_scheduler.OneCycleLR()

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        for param in self.model.parameters():  # More efficient model.zero_grad()
            param.grad = None

    def training_step(self, batch, batch_idx):
        batch = {key: value for key, value in batch.items()}
        outputs = self.model(**batch)
        loss, cos_loss = outputs[0], outputs[1]
        self.logger.log_metrics({"train_loss": loss}, self.global_step)
        self.logger.log_metrics({"train_cos_loss": cos_loss}, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        labels = batch["label"]  # .detach().cpu().numpy()
        batch = {key: value for key, value in batch.items()}
        batch["label"] = None
        outputs = self.model(**batch)
        logits = outputs[0]  # .detach().cpu().numpy()
        return labels, logits

    def _compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = {}
        acc_score = accuracy_score(y_true=preds, y_pred=labels)
        result["accuracy"] = acc_score
        return result

    def validation_epoch_end(self, val_outputs):
        """With multiple dataloaders, outputs will be a list of lists. The outer list contains one entry per dataloader, while the inner list contains the individual outputs of each validation step for that dataloader."""
        tags = ["dev", "test"]
        for idx, eval_output in enumerate(val_outputs):
            labels = (
                torch.cat([batch_output[0] for batch_output in eval_output])
                .cpu()
                .numpy()
            )
            preds = (
                torch.cat([batch_output[1] for batch_output in eval_output])
                .cpu()
                .numpy()
            )

            results = self._compute_metrics(preds, labels)
            results = {
                "{}_{}".format(tags[idx], key): value for key, value in results.items()
            }
            self.logger.log_metrics(results, self.global_step)

    def test_step(self, batch, batch_idx):  # prepare_ood()
        # TODO: Need to implement prepare_ood() step-by-step outside of model.py class
        self.model.prepare_ood()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):  # evaluate_ood()
        # TODO: Need to implement evaluate_ood() step-by-step outside of evaluation.py
        batch = {key: value for key, value in batch.items()}
        ood_keys = self.model.compute_ood(**batch)
        return ood_keys

    def predict_step_end(self, predict_step_outputs):
        # TODO: Not implemented correctly. Extension of predict_step() is needed
        ood_keys = torch.cat(predict_step_outputs, dim=0)
        return ood_keys


class LayoutLMCallback(Callback):  # Callback
    def __init__(self, args):
        super().__init__()
        self.args = args
