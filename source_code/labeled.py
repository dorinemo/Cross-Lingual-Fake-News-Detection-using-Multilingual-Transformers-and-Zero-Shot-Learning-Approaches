# Cross-lingual Fake News Detection (Tamil)
# Demonstration notebook:
# - Zero-shot classification (multilingual NLI)
# - Fine-tune multilingual transformer (supervised)
# Cell 1 — installs and setup
# Run this cell first. In Colab the GPU runtime is recommended (Runtime > Change runtime type > GPU).

!pip install -q transformers datasets sentencepiece accelerate evaluate scikit-learn torch torchvision torchaudio

# Hugging Face login optional: if you have large models on HF private repo
# from huggingface_hub import login
# login(token="hf_xxx")
# Cell 2 — imports and device check
import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns  # seaborn is fine for visualization in Colab

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
# Cell 3 — example small Tamil dataset (you'll replace this with your real CSV)
# Structure: `text`, `label` where label in {'fake','real'}
sample_data = [
    ("இந்த செய்தி அரசியல் தலைவரின் மரணத்தை பற்றி தவறான தகவல் தருகிறது.", "fake"),
    ("நகராட்சி புதிய குடிநீர் திட்டத்தை இன்று அறிவித்தது.", "real"),
    ("இது ஒரு கருத்து கட்டுரை அல்ல, அதனை உண்மையாகவே எடுத்துக் கொள்ள வேண்டாம்.", "fake"),
    ("மாணவர்களுக்கு இலவச நிதியுதவி திட்டம் கல்வித்துறை மூலம் தொடங்கப்பட்டது.", "real"),
    ("இந்த புகைப்படம் மற்ற இடத்தில் எடுக்கபட்டது; நிகழ்வு தற்போது நடந்ததல்ல.", "fake"),
    ("புதிய மருத்துவமனை மாவட்டத்தில் திறக்கப்பட їїப்ட்டது.", "real")
]

df = pd.DataFrame(sample_data, columns=["text", "label"])
print(df)
# Cell 4 — ZERO-SHOT using multilingual NLI
# We'll use a multilingual XNLI-style model (xlm-roberta-large-xnli) or similar if available.
# This is for zero-shot: provide candidate labels and the model returns probabilities.

zs_model = "joeddav/xlm-roberta-large-xnli"  # multilingual XNLI model often used for zero-shot
# Alternative: "typeform/distilbert-base-uncased-mnli" (English-only) — but for Tamil use multilingual.
zs_pipeline = pipeline("zero-shot-classification", model=zs_model, device=0 if torch.cuda.is_available() else -1)

candidate_labels = ["real", "fake"]  # labels for zero-shot

def zero_shot_predict(texts):
    outputs = []
    for t in texts:
        res = zs_pipeline(t, candidate_labels, hypothesis_template="This text is {}.")
        outputs.append({"text": t, "labels": res["labels"], "scores": res["scores"], "pred": res["labels"][0]})
    return outputs

# Demo:
texts = df["text"].tolist()
zs_results = zero_shot_predict(texts)
for r in zs_results:
    print("TEXT:", r["text"])
    print(" PRED:", r["pred"], " SCORES:", r["scores"])
    print("-"*40)
# Evaluate zero-shot against the small sample (for demonstration)
y_true = df["label"].tolist()
y_pred = [r["pred"] for r in zs_results]
print("Accuracy (zero-shot):", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))
# Cell 5 — Fine-tune a multilingual transformer (supervised)
# We'll use 'xlm-roberta-base' as a compact multilingual encoder, with a classification head.
# For production you may choose larger models like xlm-roberta-large or indic-bert variants supporting Tamil.

model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Map labels to integers
label2id = {"fake":0, "real":1}
id2label = {v:k for k,v in label2id.items()}
df["label_id"] = df["label"].map(label2id)
# Create Hugging Face dataset from pandas (small demo)
from datasets import Dataset

dataset = Dataset.from_pandas(df[["text", "label_id"]])
dataset = dataset.train_test_split(test_size=0.33, seed=42)  # small split for demo
train_ds = dataset["train"]
test_ds = dataset["test"]

# Tokenize
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)

train_ds = train_ds.remove_columns(["text"])
test_ds = test_ds.remove_columns(["text"])
train_ds.set_format(type="torch")
test_ds.set_format(type="torch")
# Compute metrics for Trainer
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Training arguments (tiny for demo)
training_args = TrainingArguments(
    output_dir="./finetune_xlmrb_demo",
    evaluation_strategy="epoch",
    save_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to="none",
    seed=42,
)
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Start training (fast on small demo; for real data: increase epochs, use gradient accumulation, larger batch)
trainer.train()
# Evaluate on test set (supervised)
eval_res = trainer.evaluate()
print("Eval results:", eval_res)

# Predict detailed labels
preds_output = trainer.predict(test_ds)
logits = preds_output.predictions
preds = np.argmax(logits, axis=-1)
labels = preds_output.label_ids
print(classification_report(labels, preds, target_names=["fake","real"], digits=4))
print("Confusion matrix:")
cm = confusion_matrix(labels, preds)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=["fake","real"], yticklabels=["fake","real"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix (supervised)")
plt.show()
# Cell 6 — Using the fine-tuned model for predictions
def supervised_predict(text_list):
    enc = tokenizer(text_list, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
    return [id2label[int(p)] for p in preds]

demo_texts = [
    "புதிய திட்டம் மாணவர்களுக்கு உதவியாக இருக்கும் என அரசு கூறுகிறது.",
    "இந்த செய்தி முழுமையாக பொய்; எந்த வங்கியும் இவ்வாறு அறிவிக்கவில்லை."
]
print("Supervised preds:", supervised_predict(demo_texts))
# Cell 7 — Notes & next steps for a real Tamil dataset
notes = """
1. Replace sample CSV with real Tamil dataset with columns 'text' and 'label' (labels: 'real'/'fake').
2. For better performance:
   - Use a larger multilingual model (xlm-roberta-large) or Tamil-specific models (IndicBERT variants).
   - Use more epochs, proper train/val/test splits, class weighting or upsampling if classes imbalanced.
   - Do data cleaning, normalization (Unicode normalization, remove weird HTML, URLs), and transliteration handling if needed.
3. For zero-shot:
   - For more nuanced classes, use richer hypothesis templates (e.g., "This news is {label}." / "This article is {label}.")
   - Zero-shot works out-of-the-box but often lags supervised fine-tuned models on domain-specific data.
4. Save model:
   - trainer.save_model("path_to_save")
   - tokenizer.save_pretrained("path_to_save")
"""
print(notes)
