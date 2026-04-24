# Install deps if needed
# !pip install requests pandas

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"  # replace
BASE_URL = "https://newsapi.org/v2/everything"

# MSF countries (sample)
countries = [
    "Sudan", "Yemen", "Syria", "Afghanistan", "Haiti",
    "DR Congo", "Ethiopia", "South Sudan", "Ukraine"
]

query = " OR ".join(countries) + " AND (health OR hospital OR conflict OR humanitarian)"

params = {
    "q": query,
    "language": "en",
    "sortBy": "publishedAt",
    "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
    "apiKey": API_KEY,
    "pageSize": 50
}

response = requests.get(BASE_URL, params=params)
articles = response.json().get("articles", [])

df = pd.DataFrame(articles)
df = df[["source", "author", "title", "description", "url", "publishedAt"]]

# flatten source
df["source"] = df["source"].apply(lambda x: x["name"] if isinstance(x, dict) else x)

def classify(text):
    text = str(text).lower()

    if any(k in text for k in ["war", "conflict", "violence", "attack"]):
        return "Conflict"
    elif any(k in text for k in ["hospital", "disease", "health", "outbreak"]):
        return "Health"
    elif any(k in text for k in ["flood", "earthquake", "disaster"]):
        return "Disaster"
    elif any(k in text for k in ["aid", "refugee", "humanitarian"]):
        return "Humanitarian"
    return "Other"


def sentiment(text):
    text = str(text).lower()

    negative = ["death", "crisis", "shortage", "violence"]
    positive = ["aid", "support", "recovery"]

    if any(k in text for k in negative):
        return "Negative"
    elif any(k in text for k in positive):
        return "Positive"
    return "Neutral"


df["category"] = df["description"].apply(classify)
df["sentiment"] = df["description"].apply(sentiment)

df.to_csv("msf_articles.csv", index=False)
df.to_json("msf_articles.json", orient="records", indent=2)

df.head()