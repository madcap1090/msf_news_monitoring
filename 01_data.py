# ! python -m spacy download en_core_web_sm
import requests
import pandas as pd
from datetime import datetime, timedelta

from nocommit import NEWSAPI_KEY

BASE_URL = "https://newsapi.org/v2/everything"

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

df = pd.DataFrame(articles)

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

df.to_csv("input_data.csv", index=False)

print(f"Saved {len(df)} articles to input_data.csv")