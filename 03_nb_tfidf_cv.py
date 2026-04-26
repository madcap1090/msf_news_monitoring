# !pip install pandas scikit-learn

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold

# --- Load data ---
df = pd.read_csv("msf_articles.csv")

if df.empty:
    raise ValueError("msf_articles.csv is empty.")

# Classification, sentiment, and country detection were already done upstream.
# This script only trains an ML classifier using the existing category labels.

df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
df = df.drop_duplicates(subset=["title", "url"]).copy()

print(f"Articles after deduplication: {len(df)}")

model_df = df[df["category"] != "Other"].copy()

X = model_df["text"]
y = model_df["category"]

print("Training examples:", len(model_df))
print(y.value_counts())

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("nb", MultinomialNB())
])

if len(model_df) == 0:
    print("No labelled examples for ML training. Falling back to existing category.")
    df["category_ml"] = df["category"]

else:
    min_class_count = y.value_counts().min()

    if len(model_df) >= 6 and min_class_count >= 2:
        n_splits = min(3, min_class_count)

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="f1_macro"
        )

        print("CV F1 scores:", cv_scores)
        print("Mean CV F1:", cv_scores.mean())
    else:
        print("Not enough labelled data for cross-validation.")

    pipeline.fit(X, y)
    df["category_ml"] = pipeline.predict(df["text"])


output_cols = [
    "source",
    "author",
    "title",
    "description",
    "url",
    "publishedAt",
    "detected_countries",
    "category",
    "category_ml",
    "sentiment",
]

df[output_cols].to_csv("msf_articles_tf_idf.csv", index=False)
df[output_cols].to_json("msf_articles_tf_idf.json", orient="records", indent=2)

df[output_cols].head()