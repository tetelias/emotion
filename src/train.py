import argparse
import os

import torch
from datasets import concatenate_datasets, load_dataset
from optimum.onnxruntime import ORTModelForSequenceClassification
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, DataCollatorWithPadding

from constants_local import ID2LABEL, LABEL2ID, NUM_LABELS, PROBLEM_TYPE
from utils import binarize_labels, cudnn_speedup, remap_and_binarize_labels, stratified_split, train, validate


def main(config) -> None:

    cudnn_speedup()

    if not os.path.isdir(config.weights_dir):
        os.mkdir(config.weights_dir)

    if config.base_model == "rubert":
        base_model = "cointegrated/rubert-tiny2"
        model = BertForSequenceClassification.from_pretrained(base_model, 
                                                              num_labels=NUM_LABELS, 
                                                              id2label=ID2LABEL, 
                                                              label2id=LABEL2ID, 
                                                              problem_type=PROBLEM_TYPE)    
    elif config.base_model == "xlm":
        base_model = "xlm-roberta-base"
        model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", 
                                                                   num_labels=NUM_LABELS, 
                                                                   id2label=ID2LABEL, 
                                                                   label2id=LABEL2ID, 
                                                                   problem_type=PROBLEM_TYPE)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data_collator = DataCollatorWithPadding(tokenizer)

    hf_dataset = load_dataset("cedr")

    dataset_prepared = hf_dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True) \
        .map(lambda x: {"label": [float(y) for y in binarize_labels(x["labels"])]}, batched=False, remove_columns=["text", "labels", "source"])

    trainset, devset = stratified_split(dataset_prepared)

    if config.dataset == "combo":
        extra_dataset = load_dataset("Djacon/ru_goemotions")
        # extra_dataset = extra_dataset.filter(lambda example: any(cls in example["labels"] for cls in GOEMOTIONS_TO_CEDR_MAPPING.keys()))  
        extra_dataset = extra_dataset.filter(lambda example: any(cls in example["labels"] for cls in ["2", "4", "6"]))    
        extra_dataset = extra_dataset.map(lambda x: tokenizer(x['text'], truncation=True), batched=True) \
            .map(lambda example: {'label': remap_and_binarize_labels(example["labels"])}, remove_columns=["text", "labels"])
        trainset = concatenate_datasets((trainset, extra_dataset["train"], extra_dataset["validation"], extra_dataset["test"]))

    train_dataloader = DataLoader(
        trainset, 
        batch_size=config.batch_size, drop_last=False, shuffle=True, num_workers=0, collate_fn=data_collator
    )

    model = model.cuda()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=0.1)

    steps = config.epochs * len(train_dataloader)
    if config.scheduler == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, steps, eta_min=1e-7)
    else:
        lr_scheduler = None

    if config.train_model:
        train(config, model, train_dataloader, optimizer, lr_scheduler)

        if config.extra_info != "":
            extra_info = f"_{config.extra_info}"
        else:
            extra_info = ""
        SAVE_PATH = f'{config.weights_dir}/{config.base_model}_{config.dataset}_{config.epochs}epochs{extra_info}'
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)

    if config.evaluate_devset:
        dev_dataloader = DataLoader(
            devset, 
            batch_size=config.batch_size, drop_last=False, shuffle=True, num_workers=0, collate_fn=data_collator
        )
        validate(model, dev_dataloader)

    if config.evaluate_testset:
        test_dataloader = DataLoader(
            dataset_prepared["test"], 
            batch_size=config.batch_size, drop_last=False, shuffle=True, num_workers=0, collate_fn=data_collator
        )
        validate(model, test_dataloader)

    if config.create_onnx:
        ort_model = ORTModelForSequenceClassification.from_pretrained(SAVE_PATH, export=True, from_transformers=True)
        SAVE_PATH_ONNX = f"{SAVE_PATH}_onnx"
        ort_model.save_pretrained(SAVE_PATH_ONNX)
        tokenizer.save_pretrained(SAVE_PATH_ONNX)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument("--base-model", type=str, default="rubert")
    parser.add_argument("--batch-size", type=int, default=64) 
    parser.add_argument("--create-onnx", action="store_true")
    parser.add_argument("--dataset", type=str, default="cedr")
    parser.add_argument("--epochs", type=int, default=24) 
    parser.add_argument("--evaluate-devset", action="store_true")
    parser.add_argument("--evaluate-testset", action="store_true")
    parser.add_argument("--extra-info", type=str, default="")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--scheduler", type=str, default="")
    parser.add_argument("--train-model", action="store_true")
    parser.add_argument("--weights-dir", type=str, default="weights")
    
    config = parser.parse_args()
    main(config)
