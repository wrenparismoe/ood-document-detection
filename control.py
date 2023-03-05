import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score
from transformers import LayoutLMConfig

import wandb
from model import LayoutLMForSequenceClassification

task_to_labels = {
    "rvl_cdip": 16,
}


class ModelControl(LightningModule):
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
        self.logger = wandb
        self.model = LayoutLMForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=config
        )

    def training_step(self, batch, batch_idx):
        batch = {key: value for key, value in batch.items()}
        outputs = model(**batch)
        loss, cos_loss = outputs[0], outputs[1]
        # NOTE: May need to use log_dict() instead of log() here
        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_cos_loss",
            cos_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return pl.TrainResult(loss, cos_loss)

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]  # .detach().cpu().numpy()
        batch = {key: value for key, value in batch.items()}
        batch["label"] = None
        outputs = model(**batch)
        logits = outputs[0]  # .detach().cpu().numpy()
        return pl.EvalResult(labels, logits)

    def validation_step_end(self, validation_step_outputs):
        # NOTE: Unsure about output from validation_step(), may need to change
        labels = torch.concatenate(validation_step_outputs[0], axis=0)  # .cpu().numpy()
        preds = torch.concatenate(validation_step_outputs[1], axis=0)

        preds = torch.argmax(preds, axis=1)
        result = {}
        acc_score = accuracy_score(y_true=preds, y_pred=labels)
        # if len(result) > 1:
        #    result["score"] = np.mean(list(result.values())).item()
        result["accuracy"] = acc_score
        # TODO: How to add 'tag' to the input parameters? Need for result logging in wandb
        results = {"{}_{}".format(tag, key): value for key, value in results.items()}
        # TODO: How to access current num_steps from here?
        self.log_dict(results, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):  # prepare_ood()
        # Need to implement this method step-by-step outside of model.py class
        self.model.prepare_ood()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):  # evaluate_ood()
        # Need to implement this method step-by-step outside of evaluation.py
        evaluate_ood()
