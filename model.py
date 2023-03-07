import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import LayoutLMPreTrainedModel, LayoutLMModel
from tqdm import tqdm
from lightning.pytorch import LightningModule


class LayoutLMForSequenceClassification(LayoutLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

        self.norm_bank = None
        self.all_classes = None
        self.class_mean = None
        self.class_var = None

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        label=None,
    ):
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = pooled = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if label is not None:
            if self.config.loss == "margin":
                dist = ((pooled.unsqueeze(1) - pooled.unsqueeze(0)) ** 2).mean(-1)
                mask = (label.unsqueeze(1) == label.unsqueeze(0)).float()
                mask = mask - torch.diag(torch.diag(mask))
                neg_mask = (label.unsqueeze(1) != label.unsqueeze(0)).float()
                max_dist = (dist * mask).max()
                cos_loss = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) + (
                    F.relu(max_dist - dist) * neg_mask
                ).sum(-1) / (neg_mask.sum(-1) + 1e-3)
                cos_loss = cos_loss.mean()
            else:
                norm_pooled = F.normalize(pooled, dim=-1)
                cosine_score = torch.exp(norm_pooled @ norm_pooled.t() / 0.3)
                mask = (label.unsqueeze(1) == label.unsqueeze(0)).float()
                cosine_score = cosine_score - torch.diag(torch.diag(cosine_score))
                mask = mask - torch.diag(torch.diag(mask))
                cos_loss = cosine_score / cosine_score.sum(dim=-1, keepdim=True)
                cos_loss = -torch.log(cos_loss + 1e-5)
                cos_loss = (mask * cos_loss).sum(-1) / (mask.sum(-1) + 1e-3)
                cos_loss = cos_loss.mean()

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
            loss = loss + self.config.alpha * cos_loss
        output = (logits,) + outputs[2:]
        output = output + (pooled,)
        return ((loss, cos_loss) + output) if loss is not None else output

    def compute_ood(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        label=None,
    ):
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = pooled = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        ood_keys = {}
        softmax_score = F.softmax(logits, dim=-1).max(-1)[0]
        ood_keys["softmax"] = softmax_score.tolist()

        maha_score = []
        for c in self.all_classes:
            centered_pooled = pooled - self.class_mean[c].unsqueeze(0)
            ms = torch.diag(centered_pooled @ self.class_var @ centered_pooled.t())
            maha_score.append(ms)
        maha_score = torch.stack(maha_score, dim=-1)
        maha_score = maha_score.min(-1)[0]
        maha_score = -maha_score
        ood_keys["maha"] = maha_score.tolist()

        norm_pooled = F.normalize(pooled, dim=-1)
        cosine_score = norm_pooled @ self.norm_bank.t()
        cosine_score = cosine_score.max(-1)[0]
        ood_keys["cosine"] = cosine_score.tolist()

        energy_score = torch.logsumexp(logits, dim=-1)
        ood_keys["energy"] = energy_score.tolist()

        return ood_keys


class LayoutLMPrepModule(LightningModule):
    def __init__(self):
        # NOTE: Initialize this sub module at begginning of each epoch
        self.bank = None
        self.label_bank = None

    def prepare_ood(self, dataloader=None, rank=3):
        self.bank = None
        self.label_bank = None
        for batch in tqdm(
            dataloader, desc="Preparing OOD", postfix={"dataset": "dev"}, mininterval=2
        ):
            self.eval()
            batch = {key: value for key, value in batch.items()}
            label = batch["label"]
            outputs = self.layoutlm(
                input_ids=batch["input_ids"],
                bbox=batch["bbox"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )
            pooled = outputs[1]
            if self.bank is None:
                self.bank = pooled.detach()  # DIFF: removed .clone()
                self.label_bank = label.detach()  # DIFF: removed .clone()
            else:
                bank = pooled.detach()  # DIFF: removed .clone()
                label_bank = label.detach()  # DIFF: removed .clone()
                self.bank = torch.cat([bank, self.bank], dim=0)
                self.label_bank = torch.cat([label_bank, self.label_bank], dim=0)

        self.norm_bank = F.normalize(self.bank, dim=-1)
        N, d = self.bank.size()
        # converted mean/cov calculations to torch to keep on GPU/avoid multiple copies
        self.all_classes = self.label_bank.unique()  # DIFF
        self.class_mean = torch.zeros(
            self.all_classes.max() + 1, d, dtype=torch.float32, device=device
        )  # DIFF
        for c in self.all_classes:
            self.class_mean[c] = self.bank[self.label_bank == c].mean(0)
        centered_bank = self.bank - self.class_mean[self.label_bank]  # DIFF
        # Fixed incorrect matrix mult dims in compute_ood() by transposing centered_bank before calculating cov
        self.class_var = torch.linalg.pinv(
            torch.cov(centered_bank.T, correction=0), hermitian=True
        )  # DIFF
