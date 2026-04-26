# !pip install pandas spacy pycountry
# !python -m spacy download en_core_web_sm

import pandas as pd
import spacy
import pycountry

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

# --- Load data ---
df = pd.read_csv("input_data.csv")

# --- NOTE: classification and sentiment are already computed upstream ---
# We only perform country detection + filtering here

# --- Build text (needed for country detection) ---
df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

# --- Country detection ---
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

# --- Filter relevant countries ---
df = df[df["detected_countries"].apply(len) > 0].copy()

# --- Quick sanity check (printed once) ---
print("Category distribution:")
print(df["category"].value_counts(), "\n")

print("Sentiment distribution:")
print(df["sentiment"].value_counts(), "\n")

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