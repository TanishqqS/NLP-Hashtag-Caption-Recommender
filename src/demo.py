from .data_loader import load_posts
from .recommender import HashtagRecommender
from .preprocess import clean_text

def main():
    df = load_posts("data/raw/social_posts_synthetic.csv")
    model = HashtagRecommender().fit(df)

    examples = [
        "late night co op session",
        "morning run and coffee after",
        "sunset walk on the beach",
        "working on my dissertation chapter",
    ]

    for cap in examples:
        pred = model.recommend(clean_text(cap), top_k=5, top_tags=6)
        print()
        print(f"Caption: {cap}")
        print("Suggested hashtags:", " ".join(pred.hashtags))
        print("Closest matches:")
        for text, score in pred.neighbors[:3]:
            print(f"  {score:.3f}  {text}")

if __name__ == "__main__":
    main()
