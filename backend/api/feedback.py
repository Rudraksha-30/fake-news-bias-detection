import json
import os
from datetime import datetime

FEEDBACK_FILE = "data/feedback.json"

def save_feedback(text: str, predicted_label: str,
                  correct_label: str, confidence: float):
    """Save user feedback for future retraining."""

    os.makedirs("data", exist_ok=True)

    # Load existing feedback
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            feedback_data = json.load(f)
    else:
        feedback_data = []

    # Add new entry
    entry = {
        "id":              len(feedback_data) + 1,
        "timestamp":       datetime.now().isoformat(),
        "text":            text[:500],
        "predicted_label": predicted_label,
        "correct_label":   correct_label,
        "confidence":      confidence,
        "is_correct":      predicted_label == correct_label
    }

    feedback_data.append(entry)

    # Save back
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_data, f, indent=2)

    return entry

def get_feedback_stats():
    """Return summary stats of all collected feedback."""

    if not os.path.exists(FEEDBACK_FILE):
        return {"total": 0, "correct": 0, "incorrect": 0, "accuracy": 0}

    with open(FEEDBACK_FILE, "r") as f:
        data = json.load(f)

    total     = len(data)
    correct   = sum(1 for d in data if d["is_correct"])
    incorrect = total - correct

    return {
        "total":     total,
        "correct":   correct,
        "incorrect": incorrect,
        "accuracy":  round(correct / total * 100, 2) if total > 0 else 0,
        "corrections_needed": [
            d for d in data if not d["is_correct"]
        ][-5:]  # last 5 wrong predictions
    }