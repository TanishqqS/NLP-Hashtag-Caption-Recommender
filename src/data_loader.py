import pandas as pd
from .preprocess import clean_text

def load_posts(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"caption", "hashtags"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    df = df.copy()
    df["caption_clean"] = df["caption"].astype(str).map(clean_text)
    df["hashtags"] = df["hashtags"].astype(str)
    return df
