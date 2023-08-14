from argparse import Namespace
from typing import Dict, List

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from constants_local import FOLDS, GOEMOTIONS_TO_CEDR_MAPPING, LABELS, LABEL_TRANSLATION


def binarize_labels(labels) -> List:
    return [int(len(labels)==0)] + [int(i in labels) for i in range(5)]


def cudnn_speedup() -> None:
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False


def train(config: Namespace, model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, scheduler: LRScheduler = None) -> None:
    scaler = GradScaler()
    for _ in trange(config.epochs):
        tqdm_loader = tqdm(dataloader)
        
        for _, batch in enumerate(tqdm_loader):
            batch = batch.to(model.device)
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()            

            if scheduler is not None:
                scheduler.step()


def validate(model: nn.Module, dataloader: DataLoader) -> None:

    model.eval()
    targets, preds = predict_with_model(model, dataloader)
    aucs = [round(roc_auc_score(targets[:, i], preds[:, i]), 4) for i in range(6)]
    f1 = [round(f1_score(targets[:, i], preds[:, i] > 0.5, average='macro'), 4) for i in range(6)]
    precision = [round(precision_score(targets[:, i], preds[:, i] > 0.5, average='macro'), 4) for i in range(6)]
    recall = [round(recall_score(targets[:, i], preds[:, i] > 0.5, average='macro'), 4) for i in range(6)]
    print("\n", "classes:", LABELS)
    print("\n", "per class aucs:", aucs, "aucs average:", np.round(np.mean(aucs), 4))
    print("\n", "per class f1-macro:", f1, "f1-macro average:", np.round(np.mean(f1), 4))
    print("\n", "per class precision:", precision, "precision average:", np.round(np.mean(precision), 4))
    print("\n", "per class recall:", recall, "recall average:", np.round(np.mean(recall), 4))


def predict_with_model(model: nn.Module, dataloader: DataLoader) -> (np.array, np.array):
    preds = []
    targets = []

    for batch in tqdm(dataloader):
        targets.append(batch.labels.cpu().numpy())
        batch = batch.to(model.device)
        # with torch.no_grad():
        with torch.inference_mode():
            if "xlm" in model.base_model.name_or_path:
                raw_predictions = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            else:
                raw_predictions = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, token_type_ids=batch.token_type_ids)
        preds.append(torch.softmax(raw_predictions.logits, -1).cpu().numpy())
    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    return targets, preds


def remap_and_binarize_labels(class_list: str) -> str:
    return_string_list = []
    for el in class_list:
        if el in GOEMOTIONS_TO_CEDR_MAPPING:
            return_string_list.append(GOEMOTIONS_TO_CEDR_MAPPING[el])
        else:
            return_string_list.append(el)
    return [float(str(i) in "".join(return_string_list)) for i in range(6)]


def save_model(config) -> None:
    pass


def stratified_split(dataset: datasets.Dataset) -> (datasets.Dataset, datasets.Dataset):
    skf = StratifiedKFold(n_splits=FOLDS)

    # В данный момент экземпляр датасета может содержать  несколько классов. С ростом номера класса частота
    # присутствия уменьшается, поэтому для стратификации из нескольких выберем класс с максимальным номером
    max_label = np.argmax(np.array(dataset["train"]["label"]) * np.array([i for i in range(1,7)]), axis=1)
    row_numbers = range(dataset["train"].num_rows)

    skf.get_n_splits(row_numbers, max_label)

    for (train_index, test_index) in skf.split(row_numbers, max_label):
        break

    trainset = dataset["train"].select(train_index)
    devset = dataset["train"].select(test_index)
    return trainset, devset
    

def transform_pipeline_prediction(prediction: Dict) -> Dict:
    return {"класс": LABEL_TRANSLATION[prediction["label"]], "вероятность": prediction["score"]}