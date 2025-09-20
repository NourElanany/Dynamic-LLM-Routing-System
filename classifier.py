import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    pipeline
)
from datasets import Dataset
model_path = "best_model"

pipe = pipeline("text-classification", model=model_path, tokenizer=model_path, device=0 if torch.cuda.is_available() else -1)

queries = [
    "Develop a multi-step plan to reduce carbon emissions in a mid-sized city.",
    "The model should understand moderately difficult content.",
    "Quantum physics requires advanced understanding.",
    "Develop a multi-step plan to reduce carbon emissions in a mid-sized city, considering economic, social, and political factors."
]

for q in queries:
    pred = pipe(q)[0]
    print(f"🔹 Text: {q}")
    print(f"   → Predicted: {pred['label']} (Score: {pred['score']:.4f})\n")

def classify_text(text):
    pred = pipe(text)[0]
    return pred['label']

