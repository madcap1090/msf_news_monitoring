# !pip install requests pandas spacy pycountry spacytextblob
# !python -m spacy download en_core_web_sm

import importlib
import requests
import pandas as pd
from datetime import datetime, timedelta

import spacy
import pycountry

from nocommit import NEWSAPI_KEY

utils = importlib.import_module("00_utils")
sentiment = utils.sentiment

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


query = """
(humanitarian OR medical OR hospital OR disease OR outbreak OR vaccine OR malnutrition
OR refugee OR displacement OR conflict OR war OR violence OR crisis)
"""

params = {
    "q": query.strip(),
    "language": "en",
    "sortBy": "publishedAt",
    "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
    "apiKey": NEWSAPI_KEY,
    "pageSize": 100,
}

response = requests.get(BASE_URL, params=params)
response.raise_for_status()

articles = response.json().get("articles", [])
print(f"Retrieved articles: {len(articles)}")

df = pd.DataFrame(articles)

if df.empty:
    raise ValueError("No articles returned from NewsAPI.")

df = df[[
    "source",
    "author",
    "title",
    "description",
    "url",
    "publishedAt",
    "content",
]].copy()

df["source"] = df["source"].apply(
    lambda x: x["name"] if isinstance(x, dict) else x
)

df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
df = df.drop_duplicates(subset=["title", "url"]).copy()

df["detected_countries"] = df["text"].apply(detect_countries)
df["category"] = df["text"].apply(classify_rule_based)
df["sentiment"] = df["text"].apply(sentiment)

df.to_csv("input_data.csv", index=False)

print(f"Saved {len(df)} classified articles to input_data.csv")

df.head()