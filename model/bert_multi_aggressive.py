# ============================================================================
# MULTILINGUAL XLM-RoBERTa CLASSIFIER
# LAST 4 LAYERS TRAINABLE + CONFUSION MATRIX + ERROR ANALYSIS
# ============================================================================

import os
import pathlib
import warnings
import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

BASE = "C:\\Users\\lisai\\Desktop\\idea\\idea_2025.2026\\datascience2\\project2\\datascience"
CSV_PATH = f"{BASE}/data_africa/ldamulti.csv"
OUTPUT_DIR = f"{BASE}/tmp/xlm_roberta_last4_layers"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# CLASSIFIER
# ============================================================================

class MultilingualBERTClassifier:

    def __init__(self):
        self.model_name = "xlm-roberta-base"

        self.topic_labels = {
            0: "Economic Development",
            1: "Natural Resources & Energy",
            2: "War & Conflict",
            3: "Social Services",
            4: "Politics & Governance",
            5: "Art, Technology and Sport"
        }

        self.text_column = "combined_text"
        self.label_column = "predicted_label_id"
        self.language_column = "language_code"

    # ----------------------------------------------------------------------

    def prepare_dataset(self, csv_path, max_samples=5000, test_size=0.15):

        df = pd.read_csv(csv_path)
        df = df[df[self.label_column].notna()]

        # Pick the best available text column; fail fast if it's URL-only
        candidate_text_cols = [self.text_column, "processed_text", "text", "description", "title"]
        found_text_col = None
        for col in candidate_text_cols:
            if col in df.columns and df[col].notna().any():
                found_text_col = col
                break
        if not found_text_col:
            raise ValueError("No usable text column found. Please provide article text (not URLs) in one of: combined_text, processed_text, text, description, title.")

        self.text_column = found_text_col

        # Detect if the chosen column is mostly URLs and stop early if so
        txt_series = df[self.text_column].fillna("").astype(str)
        url_like = txt_series.str.startswith(("http://", "https://", "www.")).mean()
        if url_like > 0.8:
            raise ValueError(f"The selected text column '{self.text_column}' appears to contain URLs, not article text. Please supply textual content for training.")

        df["text"] = txt_series
        df["label"] = df[self.label_column].astype(int)

        df = df.groupby("label").apply(
            lambda x: x.sample(min(len(x), max_samples), random_state=42)
        ).reset_index(drop=True)

        train_df, test_df = train_test_split(
            df,
            stratify=df["label"],
            test_size=test_size,
            random_state=42
        )

        self.test_texts = test_df["text"].tolist()
        self.test_labels = test_df["label"].tolist()

        return DatasetDict({
            "train": Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))
        })

    # ----------------------------------------------------------------------

    def create_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.topic_labels),
            id2label=self.topic_labels,
            label2id={v: k for k, v in self.topic_labels.items()}
        )

        # Freeze all layers
        for p in model.parameters():
            p.requires_grad = False

        # Unfreeze last 4 layers
        for layer in model.roberta.encoder.layer[-4:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Unfreeze classifier
        for p in model.classifier.parameters():
            p.requires_grad = True

        return model.to(DEVICE), tokenizer

    # ----------------------------------------------------------------------

    def tokenize(self, dataset, tokenizer):
        return dataset.map(
            lambda x: tokenizer(
                x["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            ),
            batched=True,
            remove_columns=["text"]
        )

    # ----------------------------------------------------------------------

    def train(self, model, tokenizer, dataset):

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="weighted"
            )
            acc = accuracy_score(labels, preds)
            return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=12,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=2,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
        )

        trainer.train()

        self.analyze_results(trainer, dataset)

    # ----------------------------------------------------------------------
    # CONFUSION MATRIX + ERROR ANALYSIS
    # ----------------------------------------------------------------------

    def analyze_results(self, trainer, dataset):

        preds = trainer.predict(dataset["test"])
        probs = torch.softmax(torch.tensor(preds.predictions), dim=1).numpy()
        y_pred = np.argmax(probs, axis=1)
        y_true = preds.label_ids

        print("\n" + "="*70)
        print("CONFUSION MATRIX (NORMALIZED)")
        print("="*70)

        cm = confusion_matrix(y_true, y_pred, normalize="true")
        cm_df = pd.DataFrame(cm, index=self.topic_labels.values(), columns=self.topic_labels.values())
        print(cm_df.round(3))

        cm_df.to_csv(f"{OUTPUT_DIR}/confusion_matrix.csv")

        # ----------------------------------------------------------
        # ERROR ANALYSIS
        # ----------------------------------------------------------

        records = []

        for i in range(len(y_true)):
            records.append({
                "text": self.test_texts[i][:500],
                "true_label": self.topic_labels[y_true[i]],
                "pred_label": self.topic_labels[y_pred[i]],
                "confidence": probs[i][y_pred[i]],
                "correct": y_true[i] == y_pred[i]
            })

        df_analysis = pd.DataFrame(records)

        print("\n" + "="*70)
        print("❌ WORST FAILURES (HIGH CONFIDENCE, WRONG)")
        print("="*70)

        worst = df_analysis[df_analysis.correct == False].sort_values("confidence", ascending=False).head(10)

        for i, row in worst.iterrows():
            print(f"\nCONFIDENCE: {row.confidence:.2f}")
            print(f"TRUE: {row.true_label}")
            print(f"PRED: {row.pred_label}")
            print(f"TEXT: {row.text}...")

        print("\n" + "="*70)
        print("✅ BEST PREDICTIONS (HIGH CONFIDENCE, CORRECT)")
        print("="*70)

        best = df_analysis[df_analysis.correct == True].sort_values("confidence", ascending=False).head(10)

        for i, row in best.iterrows():
            print(f"\nCONFIDENCE: {row.confidence:.2f}")
            print(f"LABEL: {row.true_label}")
            print(f"TEXT: {row.text}...")

        df_analysis.to_csv(f"{OUTPUT_DIR}/prediction_analysis.csv", index=False)

# ============================================================================
# MAIN
# ============================================================================

def main():
    clf = MultilingualBERTClassifier()
    dataset = clf.prepare_dataset(CSV_PATH)
    model, tokenizer = clf.create_model()
    tokenized = clf.tokenize(dataset, tokenizer)
    clf.train(model, tokenizer, tokenized)

if __name__ == "__main__":
    main()
