import argparse     # python 입력 augment 처리를 위함
import logging      # log를 출력하기 위함
import os
from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, AdamW

from transformers.optimization import AdamW as AdamL, get_cosine_schedule_with_warmup

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint 
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import BartTranslationDataset

#paser 생성
parser = argparse.ArgumentParser(description='BART Translation Trainer')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

parser.add_argument('--gradient_clip_val',
                    type=str,
                    help='gradient clip val')

parser.add_argument('--max_epochs',
                    type=str,
                    help='max epochs')

parser.add_argument('--default_root_dir',
                    type=str,
                    help='default root dir')

parser.add_argument('--gpus',
                    type=str,
                    help='default root dir')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

#토큰 로더
def get_kobart_tokenizer() :
    tok = PreTrainedTokenizerFast(
            tokenizer_file="./SKT/emji_tokenizer/model.json",
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>"
        )
    return tok


# 기본 Arg 설정
class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='data/train.tsv',
                            help='train file')

        parser.add_argument('--test_file',
                            type=str,
                            default='data/test.tsv',
                            help='test file')

        parser.add_argument('--batch_size',
                            type=int,
                            default=28,
                            help='')

        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        return parser

# DataModule: 필요한 DataLoader를 전달함
# train, test, validation dataset을 분할, 셔플, 배치로 나눌 수 있음
class KoEnCorpusDataModule(L.LightningDataModule):
    def __init__(self, train_file, test_file, tok, max_len=512, batch_size=4, num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file = train_file
        self.test_file = test_file
        self.tok = tok
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        # num_workers argments 추가
        # worker: 데이터 불러올 친구들의 수
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        return parser
    
    def setup(self, stage: str) -> None:
        # split dataset
        self.train = BartTranslationDataset(self.train_file,
                                            self.tok,
                                            self.max_len)
        self.test = BartTranslationDataset(self.train_file,
                                    self.tok,
                                    self.max_len)
        
        return super().setup(stage)

    def train_dataloader(self):
        train = DataLoader(self.train,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.test,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=False)
        return test
    

# LightningModule: pytorch 모델 정의에 사용됨
# 훈련 루프 제어, 옵티마이저 및 스케줄러 관리(configure_optimizers)
class KoBartTranslater(L.LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoBartTranslater, self).__init__()
        for attr_name, attr_value in vars(args).items():
            self.hparams[attr_name] = attr_value

        self.model = BartForConditionalGeneration.from_pretrained("./SKT/kobart_from_pretrained")
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token_id = 0
        self.tokenizer = get_kobart_tokenizer()

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=14,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',
                            type=float,
                            default=3e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='kobart model path')
        return parser
    
    def setup(self, stage: str) -> None:
        return super().setup(stage)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # optimizer: 모델 훈련 시, 모델의 weight와 bias를 조정하여 손실함수 최소화 (손실 함수의 grad 계산, 업데이트 진행)
        # lr_scheduler: 학습률(learning rate, lr) 스케줄링을 위함, LR 조절로 더 적은 Loss 도달 가능
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # torch.optim.AdamW()
        optimizer = torch.optim.AdamW(self.parameters(),
                            lr=self.hparams.lr)
        
        # warm up lr
        # num_workers = self.hparams.num_workers
        # data_len = len(self.trainer.train_dataloader())
        # logging.info(f'number of workers {num_workers}, data length {data_len}')
        # num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)

        # logging.info(f'num_train_steps : {num_train_steps}')
        # num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        # logging.info(f'num_warmup_steps : {num_warmup_steps}')
        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        # lr_scheduler = {'scheduler': scheduler, 
        #                 'monitor': 'loss', 'interval': 'step',
        #                 'frequency': 1}

        return optimizer
    
    def forward(self, inputs):
        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        
        return self.model(input_ids=inputs['input_ids'],
                            attention_mask=attention_mask,
                            decoder_input_ids=inputs['decoder_input_ids'],
                            decoder_attention_mask=decoder_attention_mask,
                            labels=inputs['labels'], return_dict=True)


    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        return (loss)

def main(hparam): 
    # 적용된 args 보여주기
    logging.info(args)

    # load model, data module
    model = KoBartTranslater(args)
    dm = KoEnCorpusDataModule(args.train_file,
                        args.test_file,
                        get_kobart_tokenizer(),
                        max_len=args.max_len,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers)
    
    #checkpoint 콜백 정의
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                            dirpath=args.default_root_dir,
                                            filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                            verbose=True,
                                            save_last=True,
                                            mode='min',
                                            save_top_k=-1)
    logger = TensorBoardLogger(save_dir=args.default_root_dir + '/logger')

    trainer = L.Trainer(accelerator="gpu",
                        gradient_clip_val=float(args.gradient_clip_val),
                        default_root_dir=args.default_root_dir,
                        max_epochs=int(args.max_epochs),
                        logger=logger,
                        log_every_n_steps=True,
                        enable_progress_bar=True,
                        enable_checkpointing=True,
                        callbacks=[checkpoint_callback],
                        )
    
    torch.set_float32_matmul_precision('heightest')
    trainer.fit(model, dm)
    pass

if __name__ == '__main__':
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KoBartTranslater.add_model_specific_args(parser)
    parser = KoEnCorpusDataModule.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)