# !pip install requests pandas python-dotenv spacy pycountry scikit-learn
# !python -m spacy download en_core_web_sm

import os
import requests
import pandas as pd
from datetime import datetime, timedelta


import spacy
import pycountry

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from nocommit import NEWSAPI_KEY

if not NEWSAPI_KEY:
    raise ValueError("Missing NEWSAPI_KEY. Add it to a .env file or environment variable.")

BASE_URL = "https://newsapi.org/v2/everything"

nlp = spacy.load("en_core_web_sm")

MSF_COUNTRIES = {
    "Afghanistan", "Bangladesh", "Burundi", "Central African Republic",
    "Chad", "Democratic Republic of the Congo", "Ethiopia",
    "Haiti", "Iraq", "Lebanon", "Mali", "Myanmar", "Niger", "Nigeria",
    "Palestine", "Somalia", "South Sudan", "Sudan", "Syria", "Ukraine",
    "Yemen"
}

COUNTRY_ALIASES = {
    "DR Congo": "Democratic Republic of the Congo",
    "DRC": "Democratic Republic of the Congo",
    "Congo": "Democratic Republic of the Congo",
    "Gaza": "Palestine",
    "West Bank": "Palestine",
}

query = (
    "humanitarian OR medical OR hospital OR disease OR outbreak OR vaccine "
    "OR malnutrition OR refugee OR displacement OR conflict OR war "
    "OR violence OR crisis"
)


params = {
    "q": query,
    "language": "en",
    "sortBy": "publishedAt",
    "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
    "apiKey": NEWSAPI_KEY,
    "pageSize": 100,
}

response = requests.get(BASE_URL, params=params)
response.raise_for_status()

data = response.json()
articles = data.get("articles", [])

print(f"Retrieved articles: {len(articles)}")

df = pd.DataFrame(articles)

if df.empty:
    raise ValueError("No articles returned from NewsAPI.")

df = df[["source", "author", "title", "description", "url", "publishedAt"]].copy()

df["source"] = df["source"].apply(
    lambda x: x["name"] if isinstance(x, dict) else x
)

df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

df = df.drop_duplicates(subset=["title", "url"]).copy()

print(f"Articles after deduplication: {len(df)}")

def classify_rule_based(text):
    text = str(text).lower()

    if any(k in text for k in ["war", "conflict", "violence", "attack", "fighting"]):
        return "Conflict"

    if any(k in text for k in ["hospital", "disease", "health", "outbreak", "vaccine", "cholera", "malaria"]):
        return "Health"

    if any(k in text for k in ["flood", "earthquake", "disaster", "drought"]):
        return "Disaster"

    if any(k in text for k in ["aid", "refugee", "humanitarian", "displacement", "displaced"]):
        return "Humanitarian"

    return "Other"


def sentiment_rule_based(text):
    text = str(text).lower()

    negative = ["death", "crisis", "shortage", "violence", "attack", "war", "killed"]
    positive = ["aid", "support", "recovery", "vaccination", "relief"]

    if any(k in text for k in negative):
        return "Negative"

    if any(k in text for k in positive):
        return "Positive"

    return "Neutral"


def detect_countries(text):
    doc = nlp(str(text))
    detected = set()

    for ent in doc.ents:
        if ent.label_ == "GPE":
            name = ent.text.strip()

            if name in COUNTRY_ALIASES:
                detected.add(COUNTRY_ALIASES[name])

            elif name in MSF_COUNTRIES:
                detected.add(name)

            else:
                try:
                    country = pycountry.countries.lookup(name)
                    if country.name in MSF_COUNTRIES:
                        detected.add(country.name)
                except LookupError:
                    pass

    return sorted(detected)

df["detected_countries"] = df["text"].apply(detect_countries)

df = df[df["detected_countries"].apply(len) > 0].copy()

df["category_rule_based"] = df["text"].apply(classify_rule_based)
df["sentiment"] = df["text"].apply(sentiment_rule_based)

print(f"Articles mentioning MSF countries: {len(df)}")
df["category_rule_based"].value_counts()

model_df = df[df["category_rule_based"] != "Other"].copy()

X = model_df["text"]
y = model_df["category_rule_based"]

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
    print("No labelled examples for ML training. Falling back to rule-based category.")
    df["category_ml"] = df["category_rule_based"]

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

    # Final fit on all available labelled data
    pipeline.fit(X, y)

    # Predict all articles
    df["category_ml"] = pipeline.predict(df["text"])


output_cols = [
    "source",
    "author",
    "title",
    "description",
    "url",
    "publishedAt",
    "detected_countries",
    "category_rule_based",
    "category_ml",
    "sentiment",
]

df[output_cols].to_csv("msf_articles.csv", index=False)
df[output_cols].to_json("msf_articles.json", orient="records", indent=2)

df[output_cols].head()

df[output_cols].to_csv("msf_articles.csv", index=False)
df[output_cols].to_json("msf_articles.json", orient="records", indent=2)

df[output_cols].head()