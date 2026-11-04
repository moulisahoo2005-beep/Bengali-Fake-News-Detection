import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./BanglaBERT_Final_Model"

tokenizer = AutoTokenizer.from_pretrained(
    model_path, local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path, local_files_only=True
)

model.eval()

with open(f"{model_path}/best_threshold.json", encoding="utf-8") as f:
    threshold = json.load(f)["best_threshold"]

news = "বাংলাদেশে আগামীকাল থেকে সব শিক্ষাপ্রতিষ্ঠান অনির্দিষ্টকালের জন্য বন্ধ ঘোষণা করা হয়েছে।"

print("Original Bengali News:", news)

inputs = tokenizer(
    news,
    return_tensors="pt",
    max_length=512,
    truncation=True,
    padding=True
)

with torch.inference_mode():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)

fake_probability = probabilities[0][0].item()

prediction = (
    "Fake News"
    if fake_probability >= threshold
    else "Authentic News"
)

confidence = (
    fake_probability
    if prediction == "Fake News"
    else 1 - fake_probability
)

print("\nPrediction:", prediction)
print(f"Fake News Probability: {fake_probability * 100:.2f}%")
print(f"Authentic News Probability: {(1-fake_probability) * 100:.2f}%")
print(f"Confidence: {confidence * 100:.2f}%")
print("Threshold:", threshold)