from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import os

# ── App setup ────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

app = Flask(__name__,
    template_folder=os.path.join(BASE_DIR, "frontend/templates"),
    static_folder=os.path.join(BASE_DIR, "frontend/static")
)
CORS(app)

# ── Config ───────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "backend/models/fake_news_model")
THRESHOLD  = 0.6
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model once at startup ───────────────────────────
print(f"Loading model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print("Model ready!")

# ── Text cleaning ─────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Prediction ────────────────────────────────────────────
def predict(text: str):
    cleaned = clean_text(text)
    inputs  = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    if probs[1] > THRESHOLD:
        label = "FAKE"
    elif probs[0] > THRESHOLD:
        label = "REAL"
    else:
        label = "UNCERTAIN"

    return {
        "label":      label,
        "confidence": round(float(max(probs)) * 100, 2),
        "real_prob":  round(float(probs[0]) * 100, 2),
        "fake_prob":  round(float(probs[1]) * 100, 2),
        "threshold":  THRESHOLD
    }

# ── Routes ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data["text"].strip()
    if len(text) < 10:
        return jsonify({"error": "Text too short. Minimum 10 characters."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Text too long. Maximum 5000 characters."}), 400

    result = predict(text)
    result["input_text"] = text[:200]
    return jsonify(result), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "device": str(DEVICE)}), 200

# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)