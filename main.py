# !pip install requests pandas spacy pycountry
# !python -m spacy download en_core_web_sm

import requests
import pandas as pd
from datetime import datetime, timedelta
import spacy
import pycountry

nlp = spacy.load("en_core_web_sm")

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://newsapi.org/v2/everything"

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

query = """
(humanitarian OR medical OR hospital OR disease OR outbreak OR vaccine OR malnutrition
OR refugee OR displacement OR conflict OR war OR violence OR crisis)
"""

params = {
    "q": query,
    "language": "en",
    "sortBy": "publishedAt",
    "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
    "apiKey": API_KEY,
    "pageSize": 50,
}

response = requests.get(BASE_URL, params=params)
response.raise_for_status()

articles = response.json().get("articles", [])

df = pd.DataFrame(articles)

df = df[["source", "author", "title", "description", "url", "publishedAt"]].copy()

df["source"] = df["source"].apply(
    lambda x: x["name"] if isinstance(x, dict) else x
)

df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")


def classify(text):
    text = str(text).lower()

    if any(k in text for k in ["war", "conflict", "violence", "attack"]):
        return "Conflict"
    if any(k in text for k in ["hospital", "disease", "health", "outbreak", "vaccine"]):
        return "Health"
    if any(k in text for k in ["flood", "earthquake", "disaster"]):
        return "Disaster"
    if any(k in text for k in ["aid", "refugee", "humanitarian", "displacement"]):
        return "Humanitarian"

    return "Other"


def sentiment(text):
    text = str(text).lower()

    negative = ["death", "crisis", "shortage", "violence", "attack", "war"]
    positive = ["aid", "support", "recovery", "vaccination"]

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
df["category"] = df["text"].apply(classify)
df["sentiment"] = df["text"].apply(sentiment)

df = df[df["detected_countries"].apply(len) > 0].copy()

output_cols = [
    "source",
    "author",
    "title",
    "description",
    "url",
    "publishedAt",
    "detected_countries",
    "category",
    "sentiment",
]

df[output_cols].to_csv("msf_articles.csv", index=False)
df[output_cols].to_json("msf_articles.json", orient="records", indent=2)

df[output_cols].head()