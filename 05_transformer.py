# !pip install pandas transformers datasets evaluate accelerate torch scikit-learn

import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- Load processed data ---
df = pd.read_csv("msf_articles.csv")

if df.empty:
    raise ValueError("msf_articles.csv is empty.")

# Classification, sentiment, and country detection were already done upstream
df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

model_df = df[df["category"] != "Other"].copy()

if model_df.empty:
    print("No labelled examples for transformer training. Falling back to existing category.")
    df["category_transformer"] = df["category"]

else:
    labels = sorted(model_df["category"].unique())

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    model_df["label"] = model_df["category"].map(label2id)

    print("Training examples:", len(model_df))
    print("Class distribution:")
    print(model_df["category"].value_counts(), "\n")

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    class_counts = model_df["label"].value_counts()
    min_class_count = class_counts.min()

    # --- Dataset preparation ---
    if len(model_df) < 10:
        print("⚠️ Very small dataset — skipping train/test split.")
        print("➡️ Training transformer on all labelled data.\n")

        train_dataset = Dataset.from_pandas(
            model_df[["text", "label"]].reset_index(drop=True)
        )

        train_dataset = train_dataset.map(tokenize, batched=True)
        train_dataset = train_dataset.remove_columns(["text"])
        train_dataset.set_format("torch")

        eval_dataset = None

    else:
        if min_class_count < 2:
            print("⚠️ Not enough samples per class for stratified split.")
            print("➡️ Falling back to non-stratified split.\n")
            stratify = None
        else:
            stratify = model_df["label"]

        train_df, test_df = train_test_split(
            model_df[["text", "label"]],
            test_size=0.2,
            random_state=42,
            stratify=stratify
        )

        train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
        test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

        train_dataset = train_dataset.map(tokenize, batched=True)
        test_dataset = test_dataset.map(tokenize, batched=True)

        train_dataset = train_dataset.remove_columns(["text"])
        test_dataset = test_dataset.remove_columns(["text"])

        train_dataset.set_format("torch")
        test_dataset.set_format("torch")

        eval_dataset = test_dataset

    # --- Training ---
    training_args = TrainingArguments(
        output_dir="./transformer_results",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    # --- Evaluation ---
    if eval_dataset is not None:
        predictions = trainer.predict(eval_dataset)

        y_test = predictions.label_ids
        y_pred = np.argmax(predictions.predictions, axis=1)

        target_names = [id2label[i] for i in range(len(labels))]

        print("Classification report:")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
    else:
        print("Evaluation skipped (dataset too small).")

    # --- Predict full dataset ---
    full_dataset = Dataset.from_pandas(df[["text"]].reset_index(drop=True))
    full_dataset = full_dataset.map(tokenize, batched=True)
    full_dataset = full_dataset.remove_columns(["text"])
    full_dataset.set_format("torch")

    full_predictions = trainer.predict(full_dataset)
    full_pred_ids = np.argmax(full_predictions.predictions, axis=1)

    df["category_transformer"] = [id2label[i] for i in full_pred_ids]


# --- Save output ---
output_cols = [
    "source",
    "author",
    "title",
    "description",
    "url",
    "publishedAt",
    "detected_countries",
    "category",
    "category_transformer",
    "sentiment",
]

df[output_cols].to_csv("msf_articles_transformer.csv", index=False)
df[output_cols].to_json("msf_articles_transformer.json", orient="records", indent=2)

df[output_cols].head()