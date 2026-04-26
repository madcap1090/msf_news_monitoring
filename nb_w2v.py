# !pip install gensim scikit-learn

from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import re

def simple_tokenize(text):
    text = str(text).lower()
    return re.findall(r"\b[a-z]{2,}\b", text)


model_df = df[df["category_rule_based"] != "Other"].copy()

model_df["tokens"] = model_df["text"].apply(simple_tokenize)

sentences = model_df["tokens"].tolist()

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,  # skip-gram
    seed=42
)

def document_vector(tokens, model):
    vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


X = np.vstack(
    model_df["tokens"].apply(lambda tokens: document_vector(tokens, w2v_model))
)

y = model_df["category_rule_based"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred))

df["tokens"] = df["text"].apply(simple_tokenize)

X_all = np.vstack(
    df["tokens"].apply(lambda tokens: document_vector(tokens, w2v_model))
)

df["category_word2vec"] = clf.predict(X_all)

output_cols = [
    "source",
    "author",
    "title",
    "description",
    "url",
    "publishedAt",
    "detected_countries",
    "category_rule_based",
    "category_word2vec",
    "sentiment",
]