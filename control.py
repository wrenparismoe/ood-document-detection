import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from transformers import LayoutLMConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from evaluation import get_auroc, get_fpr_95
from model import LayoutLMForSequenceClassification

task_to_labels = {
    "rvl_cdip": 16,
}


class LayoutLMModule(pl.LightningModule):  # LightningModule
    def __init__(self, args, layoutlm_config=None):
        super().__init__()
        self.args = args
        # Create model config and load pretraiend model
        config = LayoutLMConfig.from_pretrained(args.model_name_or_path)
        config.num_labels = task_to_labels[args.task_name]
        #config.gradient_checkpointing = True
        config.hidden_act = "gelu_new"
        config.alpha = args.alpha
        config.loss = args.loss

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
        print(f"Training epoch {self.current_epoch + 1} of {self.trainer.max_epochs}...")
        for param in self.model.parameters():  # More efficient model.zero_grad()
            param.grad = None

    def training_step(self, batch, batch_idx):
        batch = {key: value for key, value in batch.items()}
        outputs = self.model(**batch)
        loss, cos_loss = outputs[0], outputs[1]
        self.logger.log_metrics({"train_loss": loss}, self.global_step)
        self.logger.log_metrics({"train_cos_loss": cos_loss}, self.global_step)
        return loss

    # def training_step_end(self, training_step_outputs):
    #     return training_step_outputs

    def on_validation_epoch_start(self):
        print(f"Validating epoch {self.current_epoch + 1} of {self.trainer.max_epochs}...")
        self.dev_labels = []
        self.dev_logits = []
        self.test_labels = []
        self.test_logits = []

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            print(f"Validating dataloader {dataloader_idx + 1} of 2...")
        labels = batch["label"]  # .detach().cpu().numpy()
        batch = {key: value for key, value in batch.items()}
        batch["label"] = None
        outputs = self.model(**batch)
        logits = outputs[0]  # .detach().cpu().numpy()
        if dataloader_idx == 0:
            self.dev_labels.append(labels)
            self.dev_logits.append(logits)
        elif dataloader_idx == 1:
            self.test_labels.append(labels)
            self.test_logits.append(logits)
        return labels, logits

    # def validation_step_end(self, validation_step_outputs):
    #     return validation_step_outputs

    def on_validation_epoch_end(self):
        """
        With multiple dataloaders, outputs will be a list of lists.
        The outer list contains one entry per dataloader, while the inner list
        contains the individual outputs of each validation step for that dataloader.
        """
        def _compute_metrics(preds, labels):
            preds = np.argmax(preds, axis=1)
            result = {}
            acc_score = accuracy_score(y_true=preds, y_pred=labels)
            result["accuracy"] = acc_score
            return result
        
        def _get_results(labels, logits, tag='val'):
            labels = torch.cat(labels).detach().cpu().numpy()
            preds = torch.cat(logits).detach().cpu().numpy()
            results = _compute_metrics(preds, labels)
            results = {"{}_{}".format(tag, key): value for key, value in results.items()}
            return results


        dev_results = _get_results(self.dev_labels, self.dev_logits, tag='dev')
        self.logger.log_metrics(dev_results, self.global_step)
        test_results = _get_results(self.test_labels, self.test_logits, tag='test')
        self.logger.log_metrics(test_results, self.global_step)

    def on_test_epoch_start(self):
        print(f"Testing epoch {self.current_epoch + 1} of {self.trainer.max_epochs}...")
        self.bank = None
        self.label_bank = None

    def test_step(self, batch, batch_idx):  # prepare_ood()
        # Taken from self.model.prepare_ood()
        batch = {key: value for key, value in batch.items()}
        label = batch["label"]
        outputs = self.model.layoutlm(
            input_ids=batch["input_ids"],
            bbox=batch["bbox"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )
        pooled = outputs[1]
        if self.bank is None:
            self.bank = pooled.clone().detach()
            self.label_bank = label.clone().detach()
        else:
            bank = pooled.clone().detach()
            label_bank = label.clone().detach()
            self.bank = torch.cat([bank, self.bank], dim=0)
            self.label_bank = torch.cat([label_bank, self.label_bank], dim=0)
        return self.bank, self.label_bank

    # def test_step_end(self, test_step_outputs):
    #     return test_step_outputs

    def on_test_epoch_end(self):
        # self.bank = torch.cat([batch_output[0] for batch_output in test_outputs])
        # self.label_bank = torch.cat([batch_output[1] for batch_output in test_outputs])

        self.model.norm_bank = F.normalize(self.bank, dim=-1)
        N, d = self.bank.size()
        self.model.all_classes = self.label_bank.unique()
        self.model.class_mean = torch.zeros(
            self.model.all_classes.max() + 1,
            d,
            dtype=torch.float32,
            device=self.bank.device,
        )
        for c in self.model.all_classes:
            self.model.class_mean[c] = self.bank[self.label_bank == c].mean(0)
        centered_bank = self.bank - self.model.class_mean[self.label_bank]
        self.model.class_var = torch.linalg.pinv(
            torch.cov(centered_bank.T, correction=0), hermitian=True
        )

    def on_predict_start(self):
        self.keys = ["softmax", "maha", "cosine", "energy"]

    def on_predict_epoch_start(self):
        print(f"Predicting epoch {self.current_epoch + 1} of {self.trainer.max_epochs}...")
        self.in_scores = []
        self.out_scores = []

    def predict_step(self, batch, batch_idx, dataloader_idx=None):  # evaluate_ood()
        # [DataLoader(test_dataset), DataLoader(ood_dataset)]
        batch = {key: value for key, value in batch.items()}
        ood_keys = self.model.compute_ood(**batch)
        if dataloader_idx == 0:
            self.in_scores.append(ood_keys)
        elif dataloader_idx == 1:
            self.out_scores.append(ood_keys)

    def on_predict_epoch_end(self):
        def _merge_keys(self, ood_keys, keys):
            new_dict = {}
            for key in keys:
                new_dict[key] = []
                for i in ood_keys:
                    new_dict[key] += i[key]
            return new_dict

        # Concatenate outputs from test_dataset
        in_scores = self._merge_keys(self.in_scores, keys=self.keys)
        # Concatenate outputs from ood_dataset
        out_scores = self._merge_keys(self.out_scores, keys=self.keys)
        tag = "rvl_cdip_ood"
        outputs = {}
        for key in self.keys:
            ins = np.array(in_scores[key], dtype=np.float64)
            outs = np.array(out_scores[key], dtype=np.float64)
            inl = np.ones_like(ins).astype(np.int64)
            outl = np.zeros_like(outs).astype(np.int64)
            scores = np.concatenate([ins, outs], axis=0)
            labels = np.concatenate([inl, outl], axis=0)

            auroc, fpr_95 = get_auroc(labels, scores), get_fpr_95(labels, scores)

            outputs[tag + "_" + key + "_auroc"] = auroc
            outputs[tag + "_" + key + "_fpr95"] = fpr_95

        self.logger.log_metrics(outputs, self.global_step)



class LayoutLMCallback(pl.Callback):  # Callback
    def __init__(self, args):
        super().__init__()
        self.args = args
