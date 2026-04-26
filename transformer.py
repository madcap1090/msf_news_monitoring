# !pip install transformers datasets evaluate accelerate torch scikit-learn

import numpy as np
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

model_df = df[df["category_rule_based"] != "Other"].copy()

if len(model_df) == 0:
    print("No labelled examples for transformer training. Falling back to rule-based category.")
    df["category_transformer"] = df["category_rule_based"]

else:
    labels = sorted(model_df["category_rule_based"].unique())

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    model_df["label"] = model_df["category_rule_based"].map(label2id)

    print("Training examples:", len(model_df))
    print(model_df["category_rule_based"].value_counts())

    train_df, test_df = train_test_split(
        model_df[["text", "label"]],
        test_size=0.2,
        random_state=42,
        stratify=model_df["label"]
    )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    model_name = "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    
    training_args = TrainingArguments(
    output_dir="./transformer_results",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

predictions = trainer.predict(test_dataset)

y_test = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

target_names = [id2label[i] for i in range(len(labels))]

evaluation_report_df = pd.DataFrame(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
).transpose()

confusion_matrix_df = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=[f"actual_{label}" for label in target_names],
    columns=[f"predicted_{label}" for label in target_names]
)

print("Evaluation report:")
display(evaluation_report_df)

print("Confusion matrix:")
display(confusion_matrix_df)

evaluation_report_df.to_csv("transformer_evaluation_report.csv")
confusion_matrix_df.to_csv("transformer_confusion_matrix.csv")

full_dataset = Dataset.from_pandas(df[["text"]].reset_index(drop=True))
full_dataset = full_dataset.map(tokenize, batched=True)
full_dataset = full_dataset.remove_columns(["text"])
full_dataset.set_format("torch")

full_predictions = trainer.predict(full_dataset)
full_pred_ids = np.argmax(full_predictions.predictions, axis=1)

df["category_transformer"] = [id2label[i] for i in full_pred_ids]

output_cols = [
    "source",
    "author",
    "title",
    "description",
    "url",
    "publishedAt",
    "detected_countries",
    "category_rule_based",
    "category_transformer",
    "sentiment",
]

df[output_cols].to_csv("msf_articles.csv", index=False)
df[output_cols].to_json("msf_articles.json", orient="records", indent=2)

df[output_cols].head()