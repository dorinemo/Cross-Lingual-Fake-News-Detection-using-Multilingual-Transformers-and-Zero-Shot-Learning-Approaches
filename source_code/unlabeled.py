# Cell 1 — Install libraries
!pip install -q transformers torch accelerate pandas
# Cell 2 — Imports
import pandas as pd
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
print("Using device:", "GPU" if device==0 else "CPU")

# Load multilingual zero-shot model
classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", device=device)

candidate_labels = ["real", "fake"]  # target classes
# Cell 3 — Example Unlabeled Tamil dataset
# 🔹 Replace with your own CSV (must have a "text" column, but no labels)

data = {
    "text": [
        "இந்த செய்தி அரசு மாணவர்களுக்கு புதிய உதவித்திட்டத்தை அறிவித்தது.",
        "பிரபல நடிகர் நேற்று இறந்தார் என்ற செய்தி சமூக வலைத்தளங்களில் பரவுகிறது.",
        "நகராட்சி இன்று புதிய மருத்துவமனை திறந்தது.",
        "இந்த வீடியோ தற்போது நிகழ்ந்தது என கூறப்படுகின்றது ஆனால் அது 2015-ல் எடுக்கப்பட்டது."
    ]
}
df = pd.DataFrame(data)

# If you have a CSV, use this instead:
# from google.colab import files
# uploaded = files.upload()
# df = pd.read_csv("your_unlabeled_tamil_news.csv")  # must have 'text' column
# Cell 4 — Generate zero-shot predictions
preds = []
scores = []

for text in df["text"]:
    res = classifier(text, candidate_labels, hypothesis_template="This news is {}.")
    preds.append(res["labels"][0])   # top prediction
    scores.append(res["scores"][0])  # confidence of top prediction

df["prediction"] = preds
df["confidence"] = scores

print(df)
# Cell 5 — Save results for manual evaluation
df.to_csv("tamil_fake_news_predictions.csv", index=False)
print("✅ Predictions saved as tamil_fake_news_predictions.csv")

