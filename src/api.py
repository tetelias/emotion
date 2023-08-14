from typing import Dict

import torch
from fastapi import FastAPI
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, pipeline

from constants_local import LABEL_TRANSLATION
from utils import transform_pipeline_prediction


rubert_base_model = BertForSequenceClassification.from_pretrained("weights/rubert_combo_35epochs").cuda()  
rubert_base_tokenizer = AutoTokenizer.from_pretrained("weights/rubert_combo_35epochs")
xlm_base_model = AutoModelForSequenceClassification.from_pretrained("weights/xlm_combo_16epochs").cuda()
xlm_base_tokenizer = AutoTokenizer.from_pretrained("weights/xlm_combo_16epochs")
rubert_onnx_model = ORTModelForSequenceClassification.from_pretrained("weights/rubert_combo_35epochs_onnx") 
xlm_onnx_model = ORTModelForSequenceClassification.from_pretrained("weights/xlm_combo_16epochs_onnx")
TASK = "text-classification"
rubert_pipeline = pipeline(TASK, model=rubert_onnx_model, tokenizer=rubert_base_tokenizer)
xlm_pipeline = pipeline(TASK, model=xlm_onnx_model, tokenizer=xlm_base_tokenizer)
prediction1 = rubert_pipeline("Ты плохой человек!")
prediction2 = xlm_pipeline("Ты плохой человек!")
print(prediction1)
print(prediction2)

app = FastAPI()

def predict():
    pass

@app.get("/v1_predict_with_rubert")
def fetch_prediction_v1(text: str) -> Dict:
    inputs = rubert_base_tokenizer(text, return_tensors="pt")
    inputs = {key:inputs[key].cuda() for key in inputs}
    with torch.no_grad():
        probabilities = torch.sigmoid(rubert_base_model(**inputs).logits)
    return {LABEL_TRANSLATION[rubert_base_model.config.id2label[i]]:round(probabilities[0][i].item(), 4) \
            for i in range(6) if probabilities[0][i] > 0.5}

@app.get("/v2_predict_with_rubert_onnx")
def fetch_prediction_v2(text: str) -> Dict:
    prediction = rubert_pipeline(text)
    return transform_pipeline_prediction(prediction[0])

@app.get("/v3_predict_with_xlm")
def fetch_prediction_v3(text: str) -> Dict:
    inputs = xlm_base_tokenizer(text, return_tensors="pt")
    inputs = {key:inputs[key].cuda() for key in inputs}
    with torch.no_grad():
        probabilities = torch.sigmoid(xlm_base_model(**inputs).logits)
    return {LABEL_TRANSLATION[xlm_base_model.config.id2label[i]]:round(probabilities[0][i].item(), 4) \
            for i in range(6) if probabilities[0][i].item() > 0.5}

@app.get("/v4_predict_with_xlm_onnx")
def fetch_prediction_v4(text: str) -> Dict:
    prediction = xlm_pipeline(text)
    return transform_pipeline_prediction(prediction[0])