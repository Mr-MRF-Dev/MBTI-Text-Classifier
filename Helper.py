import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# 1. Load Dataset
# -----------------------------
def load_mbti_dataset(path):
    df = pd.read_csv(path)
    df.columns = ["label", "posts"]
    return df


# -----------------------------
# 2. Text Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text.strip()


# -----------------------------
# 3. Aggregate User Posts
# -----------------------------
def aggregate_posts(posts):
    return " ".join(posts.split("|||"))


# -----------------------------
# 4. Full Preprocessing Pipeline
# -----------------------------
def preprocess_mbti_data(csv_path):
    df = load_mbti_dataset(r"mbti_1.csv")

    # Aggregate + Clean
    df["text"] = df["posts"].apply(aggregate_posts)
    df["text"] = df["text"].apply(clean_text)

    X_text = df["text"]
    y = df["label"]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english"
    )

    X = vectorizer.fit_transform(X_text)

    return X, y, vectorizer


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    X, y, vectorizer = preprocess_mbti_data("mbti_1.csv")

    print("Feature matrix shape:", X.shape)
    print("Number of labels:", y.nunique())
