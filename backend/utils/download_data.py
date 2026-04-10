from datasets import load_dataset
import pandas as pd
import os

def download_liar():
    print("Downloading LIAR dataset...")
    dataset = load_dataset("ucsbnlp/liar")

    os.makedirs("data/raw", exist_ok=True)

    for split in ["train", "validation", "test"]:
        df = pd.DataFrame(dataset[split])

        label_map = {
            0: 1,
            1: 1,
            2: 1,
            3: 0,
            4: 0,
            5: 0
        }

        df["binary_label"] = df["label"].map(label_map).astype(int)

        df = df.dropna(subset=["statement", "label"])

        df = df[["statement", "binary_label"]].rename(
            columns={"statement": "text", "binary_label": "label"}
        )

        df.to_csv(f"data/raw/liar_{split}.csv", index=False)

        print(f"Saved {split}: {len(df)} samples")

if __name__ == "__main__":
    download_liar()