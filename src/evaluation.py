from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from .recommender import HashtagRecommender

def _tagset(s: str) -> set:
    return set((s or "").split())

def precision_at_k(true_tags: set, pred_tags: List[str], k: int) -> float:
    pred_k = set(pred_tags[:k])
    if not pred_k:
        return 0.0
    return len(true_tags.intersection(pred_k)) / float(k)

def recall_at_k(true_tags: set, pred_tags: List[str], k: int) -> float:
    pred_k = set(pred_tags[:k])
    if not true_tags:
        return 0.0
    return len(true_tags.intersection(pred_k)) / float(len(true_tags))

def evaluate(csv_path: str, test_size: float = 0.25, seed: int = 42, top_k_neighbors: int = 5, top_tags: int = 6) -> pd.DataFrame:
    from .data_loader import load_posts
    df = load_posts(csv_path)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    model = HashtagRecommender().fit(train_df)

    rows = []
    for _, row in test_df.iterrows():
        pred = model.recommend(row["caption_clean"], top_k=top_k_neighbors, top_tags=top_tags)
        true_tags = _tagset(row["hashtags"])

        rows.append({
            "caption": row["caption"],
            "p_at_3": precision_at_k(true_tags, pred.hashtags, 3),
            "r_at_3": recall_at_k(true_tags, pred.hashtags, 3),
            "p_at_6": precision_at_k(true_tags, pred.hashtags, 6),
            "r_at_6": recall_at_k(true_tags, pred.hashtags, 6),
        })

    return pd.DataFrame(rows)
