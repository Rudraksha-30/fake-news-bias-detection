from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "backend/models/fake_news_model"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print("Model loaded successfully!\n")

def predict(text, threshold=0.6):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    if probs[1] > threshold:
        label = "FAKE"
    elif probs[0] > threshold:
        label = "REAL"
    else:
        label = "UNCERTAIN"

    confidence = max(probs[0], probs[1]) * 100
    return label, confidence, probs

# Test samples — mix of obvious and tricky cases
test_cases = [
    # Clearly FAKE
    "Scientists confirm that vaccines cause autism in children.",
    "Government puts microchips in COVID vaccines to track citizens.",
    "Drinking bleach cures cancer according to new study.",
    "Obama born in Kenya, secret documents reveal.",

    # Clearly REAL
    "NASA successfully launches new James Webb telescope upgrade mission.",
    "Federal Reserve raises interest rates by 0.25% to combat inflation.",
    "Apple announces new iPhone model with improved battery life.",
    "WHO reports decline in global malaria cases over the past decade.",

    # Tricky / Ambiguous (expect UNCERTAIN)
    "The economy showed mixed signals in the latest quarterly report.",
    "Some experts believe the new policy could have unintended consequences.",
]

print("=" * 60)
print("       FAKE NEWS DETECTOR — MODEL TEST")
print(f"       Threshold: 0.6  |  Device: {DEVICE}")
print("=" * 60)

fake_count      = 0
real_count      = 0
uncertain_count = 0

for i, text in enumerate(test_cases, 1):
    label, confidence, probs = predict(text)

    if label == "FAKE":
        emoji = "🔴"
        fake_count += 1
    elif label == "REAL":
        emoji = "🟢"
        real_count += 1
    else:
        emoji = "🟡"
        uncertain_count += 1

    print(f"\n[{i}] {emoji}  {label}  ({confidence:.1f}% confident)")
    print(f"     REAL: {probs[0]*100:.1f}%  |  FAKE: {probs[1]*100:.1f}%")
    print(f"     Text: {text[:75]}")
    print("-" * 60)

# Summary
print(f"\n{'=' * 60}")
print(f"  SUMMARY")
print(f"  🔴 FAKE      : {fake_count}")
print(f"  🟢 REAL      : {real_count}")
print(f"  🟡 UNCERTAIN : {uncertain_count}")
print(f"  Total tested : {len(test_cases)}")
print(f"{'=' * 60}")