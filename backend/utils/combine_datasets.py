import pandas as pd
import re
import os
from sklearn.utils import resample

def clean(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def balance_dataset(df, samples_per_class):
    """Undersample each class to samples_per_class."""
    real = df[df["label"] == 0]
    fake = df[df["label"] == 1]

    real_sampled = resample(real, n_samples=min(samples_per_class, len(real)),
                            random_state=42, replace=False)
    fake_sampled = resample(fake, n_samples=min(samples_per_class, len(fake)),
                            random_state=42, replace=False)

    return pd.concat([real_sampled, fake_sampled]).sample(frac=1, random_state=42)

def load_and_clean(path):
    df = pd.read_csv(path)
    if "text" not in df.columns and "statement" in df.columns:
        df = df.rename(columns={"statement": "text"})
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    df["cleaned_text"] = df["text"].apply(clean)
    df = df[df["cleaned_text"].str.len() > 10]
    return df

def combine_stratified():
    os.makedirs("data/processed", exist_ok=True)

    split_files = {
        "train": ("data/raw/liar_train.csv",      "data/raw/wel_train.csv"),
        "val":   ("data/raw/liar_validation.csv",  "data/raw/wel_val.csv"),
        "test":  ("data/raw/liar_test.csv",        "data/raw/wel_test.csv"),
    }

    # Samples per class per dataset
    LIAR_PER_CLASS = 4000   # keep LIAR fully (hard political cases)
    WEL_PER_CLASS  = 6000   # 1.5x LIAR (more clean article patterns)

    VAL_PER_CLASS  = 500
    TEST_PER_CLASS = 500

    per_class = {
        "train": (LIAR_PER_CLASS, WEL_PER_CLASS),
        "val":   (VAL_PER_CLASS,  VAL_PER_CLASS),
        "test":  (TEST_PER_CLASS, TEST_PER_CLASS),
    }

    for split, (liar_path, wel_path) in split_files.items():
        liar_n, wel_n = per_class[split]

        print(f"\nProcessing {split} split...")

        # Load both datasets
        liar_df = load_and_clean(liar_path)
        wel_df  = load_and_clean(wel_path)

        print(f"  LIAR raw : {len(liar_df)} samples")
        print(f"  WELFake raw : {len(wel_df)} samples")

        # Balance each dataset separately
        liar_balanced = balance_dataset(liar_df, liar_n)              # full LIAR
        wel_balanced  = balance_dataset(wel_df,  int(wel_n * 1.5))    # 1.5x WELFake

        print(f"  LIAR balanced : {len(liar_balanced)} samples "
              f"(REAL: {(liar_balanced['label']==0).sum()} | "
              f"FAKE: {(liar_balanced['label']==1).sum()})")
        print(f"  WELFake balanced : {len(wel_balanced)} samples "
              f"(REAL: {(wel_balanced['label']==0).sum()} | "
              f"FAKE: {(wel_balanced['label']==1).sum()})")

    
        combined = pd.concat([liar_balanced, wel_balanced], ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

        # Save
        combined.to_csv(f"data/processed/combined_{split}.csv", index=False)

        print(f"\n  ✅ Total {split} : {len(combined)} samples")
        print(f"  Label distribution:")
        print(f"  REAL : {(combined['label']==0).sum()}")
        print(f"  FAKE : {(combined['label']==1).sum()}")
        print(f"  {'─'*40}")

    print("\n🎯 Stratified combination complete!")
    print("Files saved to data/processed/:")
    print("  → combined_train.csv")
    print("  → combined_val.csv")
    print("  → combined_test.csv")

if __name__ == "__main__":
    print("=" * 40)
    print("  Combining LIAR + WELFake datasets")
    print("  Strategy: LIAR full + WELFake 1.5x")
    print("=" * 40)
    combine_stratified()
