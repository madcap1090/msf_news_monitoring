import pandas as pd
import numpy as np
import spacy

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("msf_articles.csv")

if df.empty:
    raise ValueError("msf_articles.csv is empty.")

df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

nlp = spacy.load("en_core_web_sm")

model_df = df[df["category"] != "Other"].copy()

if model_df.empty:
    raise ValueError("No labelled examples available for training.")

X = np.vstack([nlp(text).vector for text in model_df["text"]])
y = model_df["category"]

class_counts = y.value_counts()

print("Class distribution:")
print(class_counts, "\n")

clf = LogisticRegression(max_iter=1000)

if len(model_df) < 10:
    print("Very small dataset — skipping train/test split.")
    print("Training on all available labelled data.\n")

    clf.fit(X, y)

else:
    min_class_count = class_counts.min()

    if min_class_count < 2:
        print("Not enough samples per class for stratified split.")
        print("Falling back to non-stratified train/test split.\n")
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

X_all = np.vstack([nlp(text).vector for text in df["text"]])

df["category_spacy_vectors"] = clf.predict(X_all)

output_cols = [
    "source",
    "author",
    "title",
    "description",
    "url",
    "publishedAt",
    "detected_countries",
    "category",
    "category_spacy_vectors",
    "sentiment",
]

df[output_cols].to_csv("msf_articles_spacy_vectors.csv", index=False)
df[output_cols].to_json("msf_articles_spacy_vectors.json", orient="records", indent=2)

df[output_cols].head()