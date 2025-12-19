"""English-only LDA helper.

Two steps:
1) export: Trains LDA, saves the model, and samples 200 English titles for manual topic correction (review CSV).
2) retrain: LOADS the saved model, runs predictions, and applies any manual topic corrections by article ID.

Defaults assume data_world/world_newsE.csv with columns: id, language, title.
"""

from pathlib import Path
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib # Added for model persistence

# Extra stopwords to knock out common French/Arabic fillers
ADDITIONAL_STOPWORDS = {
    'le', 'la', 'les', 'des', 'du', 'de', 'au', 'aux', 'en', 'et', 'un', 'une', 'ce', 'ces', 'cet', 'cette',
    'sur', 'par', 'pour', 'plus', 'sans', 'avec', 'dans', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'contre',
    'chez', 'lors', 'afin', 'ainsi', 'comme', 'tres', 'trop', 'pres', 'apres', 'avant', 'pendant', 'depuis',
    'entre', 'parmi',
    'al', 'allah', 'allahou', 'akbar', 'ibn', 'abu', 'bin', 'bint', 'ma', 'wa', 'fi', 'li', 'min', 'ala', 'ila'
}

DEFAULT_INPUT = Path(__file__).resolve().parent / "data_world" / "world_newsE_combined.csv"
DEFAULT_REVIEW = Path(__file__).resolve().parent / "data_world" / "world_newsE_review_200.csv"
DEFAULT_TOPICS = Path(__file__).resolve().parent / "data_world" / "world_newsE_topics1.csv"
DEFAULT_ASSIGN = Path(__file__).resolve().parent / "data_world" / "world_newsE_topics1_assignments.csv"
# New paths for model persistence
DEFAULT_MODEL_VEC = Path(__file__).resolve().parent / "data_world" / "lda_vectorizer.joblib"
DEFAULT_MODEL_LDA = Path(__file__).resolve().parent / "data_world" / "lda_model.joblib"


# 6 topic labels to apply to LDA results
TOPIC_LABELS = {
    0: 'Economic Development',
    1: 'Natural Resources & Energy',
    2: 'War & Conflict',
    3: 'Social Services',
    4: 'Politics & Governance',
    5: 'Art, Technology and Sport'
}


def load_english(input_path: Path, text_col: str = "title", lang_col: str = "language", id_col: str = "id") -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if lang_col not in df.columns:
        raise ValueError(f"Missing language column: {lang_col}")
    if text_col not in df.columns:
        raise ValueError(f"Missing text column: {text_col}")
    # Robustness check: ensure ID column exists
    if id_col not in df.columns:
        raise ValueError(f"Missing ID column: {id_col}. Please create it or specify the correct column name using --id-col.")
        
    mask = df[lang_col].astype(str).str.lower().str.startswith("en")
    english_df = df[mask].copy()
    if english_df.empty:
        raise ValueError("No English rows found after filtering.")
    english_df[text_col] = english_df[text_col].fillna("").astype(str)
    # Ensure ID column is treated as a string for robust merging
    english_df[id_col] = english_df[id_col].astype(str)
    return english_df


def apply_strict_english(df: pd.DataFrame, text_col: str, threshold: float = 0.6, min_words: int = 3) -> pd.DataFrame:
    """Drop rows whose text likely isn't English based on simple heuristics.

    Heuristics (no extra dependencies):
    - Ratio of ASCII letters to all letters must be >= threshold.
    - At least `min_words` words composed of ASCII letters (length >=2).
    """
    def is_english_text(t: str) -> bool:
        t = str(t)
        total_letters = sum(ch.isalpha() for ch in t)
        ascii_letters = sum(('A' <= ch <= 'Z') or ('a' <= ch <= 'z') for ch in t)
        letter_ratio = (ascii_letters / total_letters) if total_letters > 0 else 0.0
        ascii_words = re.findall(r"[A-Za-z]{2,}", t)
        return (letter_ratio >= threshold) and (len(ascii_words) >= min_words)

    mask = df[text_col].apply(is_english_text)
    removed = int((~mask).sum())
    if removed > 0:
        print(f"Strict English filter dropped {removed} rows (threshold={threshold}, min_words={min_words}).")
    return df[mask].copy()


def fit_lda(df: pd.DataFrame, text_col: str, n_topics: int = 6, max_features: int = 5000, random_state: int = 42):
    vec = CountVectorizer(stop_words="english", max_df=0.95, min_df=2, max_features=max_features)
    vec.fit(df[text_col])
    full_stop = set(vec.get_stop_words() or []) | ADDITIONAL_STOPWORDS
    vec.set_params(stop_words=full_stop)
    vec.stop_words_ = full_stop
    dtm = vec.transform(df[text_col])
    
    # Use 'online' method for potentially better scaling on large datasets
    lda = LatentDirichletAllocation(
        n_components=n_topics, 
        random_state=random_state, 
        learning_method="batch",
        # Added a few minor improvements:
        max_iter=5, # Number of passes over the data
        n_jobs=-1 # Use all available cores
    )
    lda.fit(dtm)
    doc_topic = lda.transform(dtm)
    return vec, lda, doc_topic


def save_model(vec: CountVectorizer, lda: LatentDirichletAllocation, vec_path: Path, lda_path: Path):
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, vec_path)
    joblib.dump(lda, lda_path)
    print(f"✓ Trained Vectorizer saved to: {vec_path}")
    print(f"✓ Trained LDA Model saved to: {lda_path}")


def load_model(vec_path: Path, lda_path: Path):
    if not vec_path.exists() or not lda_path.exists():
        raise FileNotFoundError("Model files not found. Run 'export' mode first to train and save the model.")
    vec = joblib.load(vec_path)
    lda = joblib.load(lda_path)
    return vec, lda


def top_words(lda: LatentDirichletAllocation, vec: CountVectorizer, n: int = 10) -> pd.DataFrame:
    vocab = np.array(vec.get_feature_names_out())
    rows = []
    for k, comp in enumerate(lda.components_):
        idx = comp.argsort()[:-n-1:-1]
        rows.append({"topic": k, "top_words": ", ".join(vocab[idx])})
    return pd.DataFrame(rows)


def save_assignments(df: pd.DataFrame, doc_topic: np.ndarray, assign_out: Path):
    dominant = doc_topic.argmax(axis=1)
    confidence = doc_topic.max(axis=1)
    out = df.copy()
    out["topic_id"] = dominant
    out["topic_conf"] = confidence
    # Map topic IDs to topic labels
    out["topic_label"] = out["topic_id"].map(TOPIC_LABELS)
    assign_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(assign_out, index=False)
    return assign_out


def export_review(df: pd.DataFrame, doc_topic: np.ndarray, topics_df: pd.DataFrame, output_path: Path, sample_size: int = 200, text_col: str = "title", id_col: str = "id"):
    dominant = doc_topic.argmax(axis=1)
    confidence = doc_topic.max(axis=1)
    df_temp = df.copy()
    df_temp["topic_id"] = dominant
    df_temp["topic_conf"] = confidence
    df_temp["topic_label"] = df_temp["topic_id"].map(TOPIC_LABELS)
    df_temp["top_words_for_topic"] = df_temp["topic_id"].map(topics_df.set_index("topic")["top_words"])
    
    review = df_temp if len(df_temp) <= sample_size else df_temp.sample(sample_size, random_state=42)
    
    # Use ID column in the review file for robust mapping during retrain
    review = review[[id_col, text_col, "topic_id", "topic_label", "topic_conf", "top_words_for_topic"]] 
    
    review.insert(0, "review_id", range(1, len(review) + 1))
    review["corrected_topic"] = ""
    review["notes"] = ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(output_path, index=False)
    return output_path


def apply_corrections(assign_df: pd.DataFrame, review_path: Path, id_col: str = "id") -> pd.DataFrame:
    """Applies human corrections using the unique ID column."""
    if not review_path.exists():
        print(f"Review file not found at {review_path}. No corrections applied.")
        return assign_df
        
    review = pd.read_csv(review_path)
    if "corrected_topic" not in review.columns:
        print("Review file found, but missing 'corrected_topic' column. No corrections applied.")
        return assign_df
        
    corr = review.copy()
    corr["corrected_topic"] = corr["corrected_topic"].fillna("").astype(str).str.strip()
    corr = corr[corr["corrected_topic"] != ""]
    
    if corr.empty:
        print("No corrected topics found in the review file. No corrections applied.")
        return assign_df
        
    # Convert corrected_topic to int, handling floats like "5.0"
    corr["corrected_topic"] = corr["corrected_topic"].apply(lambda x: int(float(x)))
    
    # **Robust Correction via ID**
    corr_map = dict(zip(corr[id_col].astype(str), corr["corrected_topic"]))
    
    # Use the map to override topic IDs based on the unique article ID
    assign_df["topic_id"] = assign_df.apply(
        lambda r: int(corr_map.get(r[id_col], r["topic_id"])), 
        axis=1
    )
    print(f"✓ Applied {len(corr)} manual corrections using the '{id_col}' column.")
    return assign_df


def print_confidence_snapshots(df: pd.DataFrame, text_col: str = "title", n: int = 10):
    """Print best and worst predictions by confidence."""
    if "topic_conf" not in df.columns or "topic_label" not in df.columns:
        print("WARNING: Cannot print snapshots: missing topic_conf or topic_label columns")
        return
    
    print("\n" + "="*80)
    print(f"PREDICTION CONFIDENCE SNAPSHOTS")
    print("="*80)
    
    # Overall stats
    conf = df["topic_conf"]
    print(f"\n[*] Overall Statistics:")
    print(f"   Total predictions: {len(conf)}")
    print(f"   Mean confidence:   {conf.mean():.3f}")
    print(f"   Median confidence: {conf.median():.3f}")
    print(f"   Std deviation:     {conf.std():.3f}")
    print(f"   Min confidence:    {conf.min():.3f}")
    print(f"   Max confidence:    {conf.max():.3f}")
    
    # Confidence distribution by threshold
    print(f"\n[*] Confidence Distribution:")
    for threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        count = (conf >= threshold).sum()
        pct = count / len(conf) * 100
        print(f"   >= {threshold:.1f}: {count:>6} ({pct:>5.1f}%)")
    
    # Best predictions
    print(f"\n[+] TOP {n} HIGHEST CONFIDENCE PREDICTIONS:")
    print("-" * 80)
    best = df.nlargest(n, "topic_conf")
    for idx, (i, row) in enumerate(best.iterrows(), 1):
        title_text = str(row[text_col])[:70] + "..." if len(str(row[text_col])) > 70 else str(row[text_col])
        print(f"{idx:>2}. [{row['topic_conf']:.3f}] {row['topic_label']}")
        print(f"    {title_text}")
    
    # Worst predictions
    print(f"\n[-] BOTTOM {n} LOWEST CONFIDENCE PREDICTIONS:")
    print("-" * 80)
    worst = df.nsmallest(n, "topic_conf")
    for idx, (i, row) in enumerate(worst.iterrows(), 1):
        title_text = str(row[text_col])[:70] + "..." if len(str(row[text_col])) > 70 else str(row[text_col])
        print(f"{idx:>2}. [{row['topic_conf']:.3f}] {row['topic_label']}")
        print(f"    {title_text}")
    
    # Per-topic confidence
    print(f"\n[*] AVERAGE CONFIDENCE BY TOPIC:")
    print("-" * 80)
    topic_stats = df.groupby("topic_label")["topic_conf"].agg(['mean', 'std', 'count'])
    topic_stats = topic_stats.sort_values('mean', ascending=False)
    for topic, row in topic_stats.iterrows():
        print(f"   {topic:<30} Mean: {row['mean']:.3f}  Std: {row['std']:.3f}  Count: {int(row['count']):>5}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="English LDA helper: export 200 for review, then retrain.")
    # Make mode optional with a sensible default so running without args works
    parser.add_argument("mode", nargs="?", default="export", choices=["export", "retrain"], help="export review sample or retrain LDA (default: export)")
    parser.add_argument("--strict-english", action="store_true", help="Drop rows whose titles do not look like English before sampling/retrain.")
    parser.add_argument("--english-threshold", type=float, default=0.6, help="Minimum ratio of ASCII letters to all letters (default: 0.6).")
    parser.add_argument("--english-min-words", type=int, default=3, help="Minimum number of ASCII-letter words required (default: 3).")
    parser.add_argument("--text-source", choices=["title", "description", "combined"], default="combined", help="Text source for LDA training: title only, description only, or combined (default: combined).")
    parser.add_argument("--input", dest="input_path", default=str(DEFAULT_INPUT), help="Input CSV path")
    parser.add_argument("--text-col", dest="text_col", default="title", help="Text column to model")
    parser.add_argument("--lang-col", dest="lang_col", default="language", help="Language column")
    parser.add_argument("--id-col", dest="id_col", default="id", help="Unique ID column for robust correction mapping (e.g., 'url', 'id').")
    parser.add_argument("--topics", dest="n_topics", type=int, default=6, help="Number of topics (default: 6)")
    parser.add_argument("--review", dest="review_path", default=str(DEFAULT_REVIEW), help="Review CSV path")
    parser.add_argument("--topics-out", dest="topics_out", default=str(DEFAULT_TOPICS), help="Topics CSV path")
    parser.add_argument("--assign-out", dest="assign_out", default=str(DEFAULT_ASSIGN), help="Assignments CSV path")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    review_path = Path(args.review_path)
    topics_out = Path(args.topics_out)
    assign_out = Path(args.assign_out)
    
    vec_path = DEFAULT_MODEL_VEC
    lda_path = DEFAULT_MODEL_LDA

    df_en = load_english(input_path, text_col=args.text_col, lang_col=args.lang_col, id_col=args.id_col)
    
    # Prepare text source based on user choice
    if args.text_source == "combined":
        # Combine title and description for richer LDA signal
        df_en['combined_text'] = (df_en['title'].fillna('') + ' ' + df_en['description'].fillna('')).str.strip()
        training_text_col = 'combined_text'
        print(f"📝 Using combined text (title + description) for LDA training.")
    elif args.text_source == "description":
        training_text_col = 'description'
        print(f"📝 Using description text only for LDA training.")
    else:  # title
        training_text_col = args.text_col
        print(f"📝 Using title text only for LDA training.")
    
    if args.strict_english:
        df_en = apply_strict_english(df_en, text_col=training_text_col, threshold=args.english_threshold, min_words=args.english_min_words)
    
    vec = None
    lda = None
    
    if args.mode == "export":
        # 1. Train the model
        print("Starting LDA training (export mode)...")
        vec, lda, doc_topic = fit_lda(df_en, text_col=training_text_col, n_topics=args.n_topics)
        
        # 2. Save model and top words
        save_model(vec, lda, vec_path, lda_path)
        topics_df = top_words(lda, vec)
        topics_out.parent.mkdir(parents=True, exist_ok=True)
        topics_df.to_csv(topics_out, index=False)
        
        # 3. Export review sample
        review_file = export_review(
            df_en, doc_topic, topics_df, review_path, 
            sample_size=200, text_col=args.text_col, id_col=args.id_col
        )
        
        # 4. Show confidence snapshots
        temp_df = df_en.copy()
        temp_df["topic_id"] = doc_topic.argmax(axis=1)
        temp_df["topic_conf"] = doc_topic.max(axis=1)
        temp_df["topic_label"] = temp_df["topic_id"].map(TOPIC_LABELS)
        print_confidence_snapshots(temp_df, text_col=args.text_col, n=10)
        
        print(f"\nReview file written: {review_file}")
        print("Fill 'corrected_topic' (integer topic id) and optional notes, then run 'retrain'.")

    elif args.mode == "retrain":
        # 1. Load the existing model
        print("Loading previously trained LDA model (retrain mode)...")
        vec, lda = load_model(vec_path, lda_path)
        
        # 2. Transform data with the loaded model
        print("Transforming data and predicting topics...")
        dtm = vec.transform(df_en[training_text_col])
        doc_topic = lda.transform(dtm)
        
        # 3. Apply initial assignments and corrections
        assign_df = df_en.copy()
        dominant = doc_topic.argmax(axis=1)
        confidence = doc_topic.max(axis=1)
        
        assign_df["topic_id"] = dominant
        assign_df["topic_conf"] = confidence
        
        corrected = apply_corrections(assign_df, review_path, id_col=args.id_col)
        
        # 4. Finalize and save
        corrected["topic_label"] = corrected["topic_id"].map(TOPIC_LABELS)
        assign_out.parent.mkdir(parents=True, exist_ok=True)
        corrected.to_csv(assign_out, index=False)
        print(f"\n✓ Final assignments (with corrections) saved: {assign_out}")
        
        # Show distribution by topic label
        print("\nTopic distribution:")
        print(corrected["topic_label"].value_counts().sort_index())
        
        # Show confidence snapshots
        print_confidence_snapshots(corrected, text_col=args.text_col, n=10)
        
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()