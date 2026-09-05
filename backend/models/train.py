import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16   # increased — data is balanced so no memory pressure
EPOCHS     = 2
LR         = 5e-6

# Dataset class
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Load data
def load_data(path):
    df = pd.read_csv(path).dropna(subset=["cleaned_text"])
    print(f"  Loaded {path}: {len(df)} samples")
    print(f"  REAL: {(df['label']==0).sum()} | FAKE: {(df['label']==1).sum()}")
    return df["cleaned_text"].tolist(), df["label"].tolist()

def train():
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nLoading data...")
    train_texts, train_labels = load_data("data/processed/combined_train.csv")
    val_texts,   val_labels   = load_data("data/processed/combined_val.csv")

    train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer)
    val_dataset   = FakeNewsDataset(val_texts,   val_labels,   tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    print("\nLoading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
    "backend/models/fake_news_model"  # continue from best checkpoint
)
    model.to(DEVICE)

    # Data is perfectly balanced so simple CrossEntropyLoss is enough
    loss_fn   = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    # Scheduler
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    best_val_acc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)

        # Training
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            # Progress every 100 batches
            if (i + 1) % 100 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} | "
                      f"Loss: {total_loss/(i+1):.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"\nAvg Training Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["labels"].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                true.extend(labels.cpu().numpy())

        acc = accuracy_score(true, preds)
        print(f"Validation Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(true, preds, target_names=["REAL", "FAKE"]))

        # Save best model only
        if acc > best_val_acc:
            best_val_acc = acc
            model.save_pretrained("backend/models/fake_news_model")
            tokenizer.save_pretrained("backend/models/fake_news_model")
            print(f"✅ Best model saved! (accuracy: {acc:.4f})")

    print(f"\n🎯 Training complete! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train()