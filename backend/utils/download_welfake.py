from datasets import load_dataset
import pandas as pd
import os

def download_welfake():
    os.makedirs("data/raw", exist_ok=True)

    print("Downloading WELFake dataset...")
    dataset = load_dataset("davanstrien/WELFake")

    df = pd.DataFrame(dataset["train"])
    print("Columns:", df.columns.tolist())
    print("Label distribution:\n", df["label"].value_counts())

    # Combine title + text for richer input
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    # Shuffle and split 80/10/10
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train = df.iloc[:int(0.8*n)]
    val   = df.iloc[int(0.8*n):int(0.9*n)]
    test  = df.iloc[int(0.9*n):]

    train.to_csv("data/raw/wel_train.csv", index=False)
    val.to_csv("data/raw/wel_val.csv",     index=False)
    test.to_csv("data/raw/wel_test.csv",   index=False)

    print(f"\nTrain: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("✅ WELFake dataset saved!")

if __name__ == "__main__":
    download_welfake()