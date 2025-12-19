"""
============================================================================
BERT-BASE-UNCASED (ENGLISH) TRAINING SCRIPT FOR VS CODE
============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
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

# Set your local paths here
BASE = Path("./ta_session_2")  # Change this to your project folder
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "artifacts" / "bert_english_classifier"
TOKENS_DIR = BASE / "tokens"

# Create directories
for dir_path in [DATA_DIR, OUTPUT_DIR, TOKENS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "world_newsE_topics_assignments.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# SETUP CHECK
# ============================================================================

def check_setup():
    """Verify environment setup"""
    print("\n" + "="*70)
    print("ENVIRONMENT CHECK")
    print("="*70)
    
    print(f"\n📂 Project structure:")
    print(f"   Base directory: {BASE.absolute()}")
    print(f"   Data directory: {DATA_DIR.absolute()}")
    print(f"   Output directory: {OUTPUT_DIR.absolute()}")
    
    print(f"\n🔧 Device configuration:")
    if torch.cuda.is_available():
        print(f"   ✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("   ⚠️  Running on CPU (training will be slow)")
        print("   Consider using a GPU for faster training")
    
    print(f"\n📦 Required packages:")
    try:
        import transformers
        import datasets
        import sklearn
        print(f"   ✅ transformers v{transformers.__version__}")
        print(f"   ✅ datasets v{datasets.__version__}")
        print(f"   ✅ sklearn v{sklearn.__version__}")
    except ImportError as e:
        print(f"   ❌ Missing package: {e}")
        print("\n   Install with: pip install transformers datasets scikit-learn pandas numpy torch")
        return False
    
    if not CSV_PATH.exists():
        print(f"\n❌ Data file not found: {CSV_PATH}")
        print(f"\n   Please place 'world_newsE_topics_assignments.csv' in: {DATA_DIR}")
        return False
    
    print(f"\n✅ Data file found: {CSV_PATH}")
    return True


# ============================================================================
# CLASSIFIER
# ============================================================================

class EnglishBERTClassifier:
    """
    BERT-base-uncased classifier optimized for English news articles with:
    - Last 4 layers trainable (efficient training)
    - Confusion matrix visualization
    - Detailed error analysis
    """

    def __init__(self):
        self.model_name = "bert-base-uncased"  # English-specific BERT

        self.topic_labels = {
            0: "Economic Development",
            1: "Natural Resources & Energy",
            2: "War & Conflict",
            3: "Social Services",
            4: "Politics & Governance",
            5: "Art, Technology and Sport"
        }

        self.text_column = None  # Will auto-detect
        self.label_column = None  # Will auto-detect

        print(f"🇬🇧 English BERT Classifier Initialized")
        print(f"   Model: {self.model_name}")
        print(f"   Language: English only")
        print(f"   Device: {DEVICE}")

    # ----------------------------------------------------------------------
    # DATA PREPARATION
    # ----------------------------------------------------------------------

    def prepare_dataset(self, csv_path, max_samples=5000, test_size=0.15):
        """Prepare dataset from CSV"""

        print("\n" + "="*70)
        print("PREPARING DATASET")
        print("="*70)

        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} articles from CSV")

        # Auto-detect label column
        label_candidates = ["topic_id", "predicted_label_id", "label", "topic", "category"]
        found_label_col = None
        for col in label_candidates:
            if col in df.columns and df[col].notna().any():
                found_label_col = col
                break

        if not found_label_col:
            print(f"\n❌ ERROR: No label column found")
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError("No label column found. Expected one of: " + str(label_candidates))

        self.label_column = found_label_col
        print(f"✓ Using label column: '{self.label_column}'")

        # Remove missing labels
        df = df[df[self.label_column].notna()].copy()
        print(f"After removing missing labels: {len(df)} articles")

        # Find text column
        candidate_text_cols = [
            "combined_text", "processed_text", "text",
            "description", "title", "content"
        ]

        found_text_col = None
        for col in candidate_text_cols:
            if col in df.columns and df[col].notna().any():
                found_text_col = col
                break

        if not found_text_col:
            print(f"\n❌ ERROR: No text column found")
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError(
                "No usable text column found. Expected one of: " +
                str(candidate_text_cols)
            )

        self.text_column = found_text_col
        print(f"✓ Using text column: '{self.text_column}'")

        # Prepare data
        txt_series = df[self.text_column].fillna("").astype(str)

        # Check for URLs
        url_like = txt_series.str.startswith(("http://", "https://", "www.")).mean()
        if url_like > 0.8:
            raise ValueError(
                f"Column '{self.text_column}' contains URLs, not article text. "
                "Please provide textual content for training."
            )

        df["text"] = txt_series
        df["label"] = df[self.label_column].astype(int)

        # Show statistics
        avg_length = df["text"].str.len().mean()
        median_length = df["text"].str.len().median()
        print(f"   Average text length: {avg_length:.0f} characters")
        print(f"   Median text length: {median_length:.0f} characters")

        # Balance classes
        print(f"\nBalancing classes (max {max_samples} per class)...")
        df = df.groupby("label").apply(
            lambda x: x.sample(min(len(x), max_samples), random_state=42)
        ).reset_index(drop=True)

        print(f"Balanced dataset: {len(df)} articles")

        # Show label distribution
        print("\n📊 Label distribution:")
        for label_id, count in df["label"].value_counts().sort_index().items():
            label_name = self.topic_labels[label_id]
            pct = (count / len(df)) * 100
            print(f"   {label_id}: {label_name:<30} {count:>5} ({pct:.1f}%)")

        # Split data
        train_df, test_df = train_test_split(
            df,
            stratify=df["label"],
            test_size=test_size,
            random_state=42
        )

        # Store for later analysis
        self.test_texts = test_df["text"].tolist()
        self.test_labels = test_df["label"].tolist()

        print(f"\n✓ Train: {len(train_df)} | Test: {len(test_df)}")

        return DatasetDict({
            "train": Dataset.from_pandas(
                train_df[["text", "label"]].reset_index(drop=True)
            ),
            "test": Dataset.from_pandas(
                test_df[["text", "label"]].reset_index(drop=True)
            )
        })

    # ----------------------------------------------------------------------
    # MODEL CREATION
    # ----------------------------------------------------------------------

    def create_model(self):
        """Create BERT model with last 4 layers trainable"""

        print("\n" + "="*70)
        print("CREATING MODEL (LAST 4 LAYERS TRAINABLE)")
        print("="*70)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.topic_labels),
            id2label=self.topic_labels,
            label2id={v: k for k, v in self.topic_labels.items()}
        )

        print(f"\nModel: {self.model_name}")
        print(f"Total transformer layers: {len(model.bert.encoder.layer)}")

        # Freeze all layers first
        for p in model.parameters():
            p.requires_grad = False

        # Unfreeze last 4 encoder layers
        for layer in model.bert.encoder.layer[-4:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Unfreeze classifier head
        for p in model.classifier.parameters():
            p.requires_grad = True

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"\n📊 Parameter breakdown:")
        print(f"   Total:     {total_params:,}")
        print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"   Frozen:    {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"\n✓ Last 4 layers + classifier head trainable")

        return model.to(DEVICE), tokenizer

    # ----------------------------------------------------------------------
    # TOKENIZATION
    # ----------------------------------------------------------------------

    def tokenize(self, dataset, tokenizer):
        """Tokenize dataset"""

        print("\nTokenizing articles (max_length=512)...")

        tokenized = dataset.map(
            lambda x: tokenizer(
                x["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            ),
            batched=True,
            remove_columns=["text"]
        )

        print("✓ Tokenization complete")
        return tokenized

    # ----------------------------------------------------------------------
    # TRAINING
    # ----------------------------------------------------------------------

    def train(self, model, tokenizer, dataset):
        """Train the model"""

        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)

            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="weighted"
            )
            acc = accuracy_score(labels, preds)

            return {
                "accuracy": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall
            }

        args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
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
            logging_steps=50,
            report_to="none"
        )

        print(f"\n🔥 Training configuration:")
        print(f"   Epochs: 12")
        print(f"   Batch size: 8 (effective: 32 with accumulation)")
        print(f"   Learning rate: 2e-5")
        print(f"   Max sequence length: 512")
        print(f"   Training samples: {len(dataset['train'])}")
        print(f"   Test samples: {len(dataset['test'])}")

        if torch.cuda.is_available():
            print(f"\n⚡ Estimated training time: 1.5-3 hours on GPU")
        else:
            print(f"\n⏱️  Running on CPU - training will take significantly longer")

        print(f"\n🚀 Starting training...")

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

        # Final evaluation
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)

        results = trainer.evaluate()

        print(f"\n✨ FINAL RESULTS:")
        print(f"   Accuracy:  {results['eval_accuracy']:.4f}")
        print(f"   F1 Score:  {results['eval_f1']:.4f}")
        print(f"   Precision: {results['eval_precision']:.4f}")
        print(f"   Recall:    {results['eval_recall']:.4f}")

        # Per-class report
        preds = trainer.predict(dataset["test"])
        y_pred = np.argmax(preds.predictions, axis=1)
        y_true = preds.label_ids

        print("\n" + "="*70)
        print("PER-CLASS PERFORMANCE")
        print("="*70)
        print(classification_report(
            y_true,
            y_pred,
            target_names=list(self.topic_labels.values()),
            digits=3
        ))

        # Save model
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70)

        trainer.save_model(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))

        print(f"✓ Model saved to: {OUTPUT_DIR}")

        # Run error analysis
        self.analyze_errors(trainer, dataset)

        return trainer

    # ----------------------------------------------------------------------
    # ERROR ANALYSIS
    # ----------------------------------------------------------------------

    def analyze_errors(self, trainer, dataset):
        """Detailed error analysis"""

        print("\n" + "="*70)
        print("ERROR ANALYSIS")
        print("="*70)

        preds = trainer.predict(dataset["test"])
        probs = torch.softmax(torch.tensor(preds.predictions), dim=1).numpy()
        y_pred = np.argmax(probs, axis=1)
        y_true = preds.label_ids

        # Confusion matrix
        print("\n📊 CONFUSION MATRIX (NORMALIZED)")
        print("-" * 70)

        cm = confusion_matrix(y_true, y_pred, normalize="true")
        cm_df = pd.DataFrame(
            cm,
            index=self.topic_labels.values(),
            columns=self.topic_labels.values()
        )
        print(cm_df.round(3))

        cm_path = OUTPUT_DIR / "confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        print(f"\n✓ Confusion matrix saved to: {cm_path}")

        # Error analysis
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

        # Worst failures
        print("\n" + "="*70)
        print("❌ WORST FAILURES (HIGH CONFIDENCE, WRONG)")
        print("="*70)

        worst = df_analysis[df_analysis.correct == False].sort_values(
            "confidence", ascending=False
        ).head(10)

        if len(worst) > 0:
            for idx, (i, row) in enumerate(worst.iterrows(), 1):
                print(f"\n{idx}. CONFIDENCE: {row.confidence:.3f}")
                print(f"   TRUE: {row.true_label}")
                print(f"   PRED: {row.pred_label}")
                print(f"   TEXT: {row.text[:200]}...")
        else:
            print("No errors found!")

        # Best predictions
        print("\n" + "="*70)
        print("✅ BEST PREDICTIONS (HIGH CONFIDENCE, CORRECT)")
        print("="*70)

        best = df_analysis[df_analysis.correct == True].sort_values(
            "confidence", ascending=False
        ).head(10)

        for idx, (i, row) in enumerate(best.iterrows(), 1):
            print(f"\n{idx}. CONFIDENCE: {row.confidence:.3f}")
            print(f"   LABEL: {row.true_label}")
            print(f"   TEXT: {row.text[:200]}...")

        # Save analysis
        analysis_path = OUTPUT_DIR / "prediction_analysis.csv"
        df_analysis.to_csv(analysis_path, index=False)
        print(f"\n✓ Full analysis saved to: {analysis_path}")

        # Statistics
        print("\n" + "="*70)
        print("CONFIDENCE STATISTICS")
        print("="*70)

        print(f"\nOverall:")
        print(f"   Mean confidence:   {df_analysis['confidence'].mean():.3f}")
        print(f"   Median confidence: {df_analysis['confidence'].median():.3f}")

        print(f"\nCorrect predictions:")
        correct_conf = df_analysis[df_analysis.correct]['confidence']
        print(f"   Mean confidence: {correct_conf.mean():.3f}")

        if len(worst) > 0:
            print(f"\nIncorrect predictions:")
            incorrect_conf = df_analysis[~df_analysis.correct]['confidence']
            print(f"   Mean confidence: {incorrect_conf.mean():.3f}")
            print(f"   Total errors: {len(incorrect_conf)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training workflow"""

    print("\n" + "="*80)
    print("ENGLISH BERT CLASSIFIER - TRAINING")
    print("="*80)

    # Check setup
    if not check_setup():
        return

    print("\n🎯 Configuration:")
    print("   • Model: BERT-base-uncased (English)")
    print("   • Strategy: Last 4 layers trainable (~25% of parameters)")
    print("   • Max samples: 5000 per class")
    print("   • Epochs: 12 (with early stopping)")
    print("   • Data file: world_newsE_topics_assignments.csv")

    # Initialize classifier
    clf = EnglishBERTClassifier()

    # Prepare dataset
    dataset = clf.prepare_dataset(CSV_PATH, max_samples=5000, test_size=0.15)

    # Create model
    model, tokenizer = clf.create_model()

    # Tokenize
    tokenized = clf.tokenize(dataset, tokenizer)

    # Train with analysis
    trainer = clf.train(model, tokenizer, tokenized)

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)

    print(f"\n📁 Outputs saved to: {OUTPUT_DIR}")
    print(f"   • Model files")
    print(f"   • confusion_matrix.csv")
    print(f"   • prediction_analysis.csv")

    print("\n💡 Next steps:")
    print("   • Review the confusion matrix and error analysis")
    print("   • Use the saved model for inference on new data")


# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    main()