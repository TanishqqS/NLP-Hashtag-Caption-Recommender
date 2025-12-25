from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Recommendation:
    hashtags: List[str]
    neighbors: List[Tuple[str, float]]

class HashtagRecommender:
    def __init__(self, min_df: int = 1, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(min_df=min_df, max_features=max_features, ngram_range=(1, 2))
        self.matrix = None
        self.df = None

    def fit(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.matrix = self.vectorizer.fit_transform(self.df["caption_clean"].tolist())
        return self

    def recommend(self, caption_clean: str, top_k: int = 5, top_tags: int = 6) -> Recommendation:
        if self.matrix is None or self.df is None:
            raise RuntimeError("Model not fitted. Call fit(df) first.")

        q = self.vectorizer.transform([caption_clean])
        sims = cosine_similarity(q, self.matrix).ravel()
        idx = np.argsort(sims)[::-1][:top_k]

        neighbors = [(self.df.loc[i, "caption"], float(sims[i])) for i in idx]

        tags = []
        for i in idx:
            tags.extend(self.df.loc[i, "hashtags"].split())

        seen = set()
        dedup = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                dedup.append(t)

        return Recommendation(hashtags=dedup[:top_tags], neighbors=neighbors)
