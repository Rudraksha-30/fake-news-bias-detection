import re
import pandas as pd

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy().dropna(subset=[text_col])
    df["cleaned_text"] = df[text_col].apply(clean_text)
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)

    for split in ["train", "validation", "test"]:
        df = pd.read_csv(f"data/raw/liar_{split}.csv")
        df = preprocess_dataframe(df)
        df.to_csv(f"data/processed/liar_{split}.csv", index=False)
        print(f"Processed {split}: {len(df)} samples")