import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================================
# 1. LOAD TEST DATA
# ==========================================================
df_test = pd.read_csv("/mnt/data/test.csv")

# Keep only rows with proper text
df_test = df_test[df_test["Review"].apply(lambda x: isinstance(x, str))]

# ==========================================================
# 2. CLEAN TEXT FUNCTION (same as training)
# ==========================================================
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df_test["clean_review"] = df_test["Review"].apply(clean_text)

# Drop empty
df_test = df_test[df_test["clean_review"].str.strip() != ""]

texts = df_test["clean_review"].tolist()

# ==========================================================
# 3. MODELS TO LOAD
# ==========================================================
model_names = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "xlnet-base-cased"
]

label_map = {0: "negative", 1: "positive"}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================
# 4. PREDICTION FUNCTION
# ==========================================================
def predict(model_name, texts):
    print(f"\n=== Predicting with {model_name} ===")

    tokenizer = AutoTokenizer.from_pretrained(f"./{model_name}-finetuned")
    model = AutoModelForSequenceClassification.from_pretrained(
        f"./{model_name}-finetuned"
    ).to(device)

    predictions = []

    for text in texts:
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().item()

        predictions.append(label_map[pred])

    return predictions


# ==========================================================
# 5. RUN PREDICTIONS FOR ALL MODELS & SAVE
# ==========================================================
for model_name in model_names:
    preds = predict(model_name, texts)

    out_df = pd.DataFrame({
        "Review": df_test["Review"],
        "Prediction": preds
    })

    output_path = f"./{model_name}_predictions.csv"
    out_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

