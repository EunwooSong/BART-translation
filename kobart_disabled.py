import argparse
import logging
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from dataset import KoBARTSummaryDataset
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer

parser = argparse.ArgumentParser(description="KoBART translation")

parser.add_argument("--checkpoint_path", type=str, help="checkpoint path")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgsBase:
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_file", type=str, default="data/train.tsv", help="train file"
        )

        parser.add_argument(
            "--test_file", type=str, default="data/test.tsv", help="test file"
        )

        parser.add_argument("--batch_size", type=int, default=28, help="")

        parser.add_argument("--max_len", type=int, default=512, help="max seq len")
        return parser


class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.hparams = hparams

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--batch-size",
            type=int,
            default=14,
            help="batch size for training (default: 96)",
        )

        parser.add_argument(
            "--lr", type=float, default=3e-5, help="The initial learning rate"
        )

        parser.add_argument(
            "--warmup_ratio", type=float, default=0.1, help="warmup ratio"
        )

        parser.add_argument(
            "--model_path", type=str, default=None, help="kobart model path"
        )
        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.lr, correct_bias=False
        )
        # warm up lr
        num_workers = (self.hparams.gpus if self.hparams.gpus is not None else 1) * (
            self.hparams.num_nodes if self.hparams.num_nodes is not None else 1
        )
        data_len = len(self.train_dataloader().dataset)
        logging.info(f"number of workers {num_workers}, data length {data_len}")
        num_train_steps = int(
            data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs
        )
        logging.info(f"num_train_steps : {num_train_steps}")
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f"num_warmup_steps : {num_warmup_steps}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "loss",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]


class KoBARTConditionalGeneration(Base):
    def __init__(self, hparams, **kwargs):
        super(KoBARTConditionalGeneration, self).__init__(hparams, **kwargs)
        self.model = BartForConditionalGeneration.from_pretrained(
            get_pytorch_kobart_model()
        )
        self.model.train()
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.tokenizer = get_kobart_tokenizer()

    def forward(self, inputs):
        attention_mask = inputs["input_ids"].ne(self.pad_token_id).float()
        decoder_attention_mask = (
            inputs["decoder_input_ids"].ne(self.pad_token_id).float()
        )

        return self.model(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=decoder_attention_mask,
            labels=inputs["labels"],
            return_dict=True,
        )

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs["loss"]
        return loss

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log("val_loss", torch.stack(losses).mean(), prog_bar=True)
