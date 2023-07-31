import pandas as pd
import numpy as np
import torch
from transformers import BartForConditionalGeneration, AutoTokenizer, BartConfig
from dataset import BARTCustomDataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# load dataset
tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
train_set = BARTCustomDataset('data/train.tsv', tokenizer, 512)
test_set = BARTCustomDataset('data/test.tsv', tokenizer, 512)

# print(train_set[100])
# 너무 기니깐... 섞어서 1000개만 추출
# small_train_dataset = train_set.select(range(1000))
# small_test_dataset = test_set.select(range(1000))

model_info = "gogamza/kobart-base-v2"


config = BartConfig.from_pretrained(model_info)
model = BartForConditionalGeneration(config)

batch_size = 4

args = Seq2SeqTrainingArguments(
    output_dir="checkpoints-bart-baseline-2",
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=5000,
    save_strategy="steps",
    save_steps=5000,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=1e-4,
    max_steps=50_000,
    fp16=True,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=test_set,
)
trainer.train()


# model = BartForSequenceClassification.from_pretrained(get_pytorch_kobart_model(ctx='gpu'))
# configuration = model.config
# print(configuration)
