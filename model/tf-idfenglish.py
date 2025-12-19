import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import joblib
import re

class EnglishTopicLabeler:
    """
    English-only topic labeler with two-stage approach:
    1. Keyword-based labeling for initial predictions
    2. Semi-supervised learning (manual labels + high-confidence keyword labels)
    """
    
    def __init__(self):
        self.topic_labels = {
            0: 'Economic Development',
            1: 'Natural Resources & Energy',
            2: 'War & Conflict',
            3: 'Social Services',
            4: 'Politics & Governance',
            5: 'Art, Technology and Sport'
        }
        
        # ML model pipeline
        self.model = None
        self.trained = False
        self.metrics = {}
        
        # English keywords for initial labeling
        self.keywords = {
            'Economic Development': ['economy', 'economic', 'trade', 'trading', 'investment', 'invest',
                'business', 'market', 'finance', 'financial', 'bank', 'banking', 'growth', 
                'gdp', 'currency', 'stock', 'bonds', 'capital', 'entrepreneur', 'commerce',
                'export', 'import', 'revenue', 'profit', 'fiscal', 'monetary', 'inflation',
                'debt', 'loan', 'credit', 'development', 'corporate', 'sector'],
            
            'Natural Resources & Energy': ['oil', 'petroleum', 'crude', 'mining', 'mineral', 
                'coal', 'gold', 'diamond', 'energy', 'power', 'electricity', 'gas', 'natural gas',
                'renewable', 'solar', 'wind', 'hydro', 'nuclear', 'resources', 'fossil',
                'lithium', 'copper', 'refinery', 'drilling', 'pipeline', 'reserves', 'fuel'],
            
            'War & Conflict': ['war', 'conflict', 'violence', 'military', 'army', 'soldier', 'troop',
                'rebel', 'insurgent', 'militant', 'attack', 'bombing', 'strike', 'terrorism', 
                'terrorist', 'extremist', 'militia', 'ceasefire', 'casualties', 'killed', 
                'wounded', 'battle', 'fighting', 'clash', 'combat', 'weapon', 'arms', 'guns', 
                'gun', 'bomb', 'dead', 'death', 'kill', 'fire', 'fight', 'violent', 'assault'],
            
            'Social Services': ['health', 'healthcare', 'hospital', 'medical', 'doctor', 'nurse',
                'clinic', 'patient', 'disease', 'vaccine', 'medicine', 'treatment', 'education', 
                'school', 'university', 'teacher', 'student', 'learning', 'welfare', 'social',
                'pandemic', 'epidemic', 'covid', 'virus', 'malaria', 'tuberculosis', 'hiv', 
                'aids', 'hunger', 'nutrition', 'water', 'sanitation'],
            
            'Politics & Governance': ['politics', 'political', 'government', 'governance', 
                'president', 'minister', 'parliament', 'congress', 'election', 'vote', 'voting',
                'democracy', 'policy', 'law', 'legislation', 'regulation', 'cabinet', 'opposition',
                'party', 'campaign', 'referendum', 'constitution', 'diplomacy', 'treaty', 'summit'],
            
            'Art, Technology and Sport': ['art', 'artist', 'music', 'musician', 'painting', 
                'film', 'cinema', 'movie', 'sport', 'sports', 'football', 'soccer', 'basketball', 
                'tennis', 'athlete', 'championship', 'tournament', 'match', 'game', 'player',
                'team', 'technology', 'tech', 'innovation', 'digital', 'software', 'culture']
        }
    
    def preprocess_text(self, text):
        """Clean and normalize English text"""
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def keyword_label(self, df, text_cols=['title', 'description'], min_confidence=0.3):
        """
        Apply keyword-based labeling to generate initial predictions
        
        Parameters:
        -----------
        df : DataFrame
            Articles to label
        text_cols : list
            Text columns to use for matching
        min_confidence : float
            Minimum confidence to assign a label (0-1)
        
        Returns:
        --------
        DataFrame with keyword predictions
        """
        print("\n" + "="*80)
        print("KEYWORD-BASED LABELING")
        print("="*80)
        
        df = df.copy()
        
        # Combine text
        if isinstance(text_cols, list):
            df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
        else:
            df['combined_text'] = df[text_cols].fillna('')
        
        print(f"Processing {len(df)} articles...")
        
        # Apply keyword matching
        results = []
        for idx, row in df.iterrows():
            text = self.preprocess_text(row['combined_text'])
            
            # Count keyword matches per topic
            topic_scores = {}
            for topic_name, keywords in self.keywords.items():
                score = sum(1 for kw in keywords if kw in text)
                topic_scores[topic_name] = score
            
            # Determine best topic
            total_hits = sum(topic_scores.values())
            
            if total_hits > 0:
                best_topic = max(topic_scores, key=topic_scores.get)
                best_score = topic_scores[best_topic]
                confidence = best_score / total_hits
                
                if confidence >= min_confidence:
                    label_to_id = {v: k for k, v in self.topic_labels.items()}
                    results.append({
                        'predicted_label_name': best_topic,
                        'predicted_label_id': label_to_id[best_topic],
                        'prediction_confidence': confidence,
                        'keyword_hits': best_score
                    })
                else:
                    results.append({
                        'predicted_label_name': 'Unclassified',
                        'predicted_label_id': None,
                        'prediction_confidence': 0.0,
                        'keyword_hits': 0
                    })
            else:
                results.append({
                    'predicted_label_name': 'Unclassified',
                    'predicted_label_id': None,
                    'prediction_confidence': 0.0,
                    'keyword_hits': 0
                })
        
        # Add predictions to dataframe
        results_df = pd.DataFrame(results)
        df['predicted_label_name'] = results_df['predicted_label_name']
        df['predicted_label_id'] = results_df['predicted_label_id']
        df['prediction_confidence'] = results_df['prediction_confidence']
        df['keyword_hits'] = results_df['keyword_hits']
        
        # Statistics
        classified = df['predicted_label_name'] != 'Unclassified'
        n_classified = classified.sum()
        
        print(f"\n✅ Keyword labeling complete!")
        print(f"   Classified: {n_classified} ({n_classified/len(df)*100:.1f}%)")
        print(f"   Unclassified: {len(df) - n_classified} ({(len(df)-n_classified)/len(df)*100:.1f}%)")
        print(f"   Average confidence: {df[classified]['prediction_confidence'].mean():.3f}")
        
        print(f"\n📊 TOPIC DISTRIBUTION:")
        print(df['predicted_label_name'].value_counts())
        
        print(f"\n📊 CONFIDENCE DISTRIBUTION:")
        print(f"   High (>0.6): {(df['prediction_confidence'] > 0.6).sum()}")
        print(f"   Medium (0.4-0.6): {((df['prediction_confidence'] >= 0.4) & (df['prediction_confidence'] <= 0.6)).sum()}")
        print(f"   Low (0.3-0.4): {((df['prediction_confidence'] >= 0.3) & (df['prediction_confidence'] < 0.4)).sum()}")
        print(f"   Unclassified (<0.3): {(df['prediction_confidence'] < 0.3).sum()}")
        
        return df
    
    def train_semi_supervised(self, manual_df, keyword_df=None, 
                             keyword_confidence_threshold=0.6,
                             text_cols=['title', 'description'],
                             manual_label_col='topic_id',
                             do_cross_validation=True, cv_folds=5):
        """
        Train model using manual labels + high-confidence keyword labels
        
        Parameters:
        -----------
        manual_df : DataFrame
            Manually labeled data (high quality ground truth)
        keyword_df : DataFrame (optional)
            Keyword-labeled data (semi-supervised)
        keyword_confidence_threshold : float
            Only use keyword labels with confidence >= this threshold
        text_cols : list
            Text columns to use
        manual_label_col : str
            Column with manual labels (0-5)
        """
        print("\n" + "="*80)
        print("SEMI-SUPERVISED TRAINING")
        print("="*80)
        
        # Process manual labels
        manual_df = manual_df.copy()
        if isinstance(text_cols, list):
            # Only use columns that exist
            available_cols = [col for col in text_cols if col in manual_df.columns]
            if not available_cols:
                raise ValueError(f"None of the text columns {text_cols} found in manual_df. Available: {manual_df.columns.tolist()}")
            manual_df['combined_text'] = manual_df[available_cols].fillna('').agg(' '.join, axis=1)
        else:
            if text_cols not in manual_df.columns:
                raise ValueError(f"Text column '{text_cols}' not found in manual_df. Available: {manual_df.columns.tolist()}")
            manual_df['combined_text'] = manual_df[text_cols].fillna('')
        
        manual_df['processed_text'] = manual_df['combined_text'].apply(self.preprocess_text)
        
        # Get manual training data
        X_manual = manual_df['processed_text']
        y_manual = manual_df[manual_label_col]
        
        # Remove NaN labels
        valid_mask = y_manual.notna()
        X_manual = X_manual[valid_mask]
        y_manual = y_manual[valid_mask].astype(int)
        
        print(f"\n📊 MANUAL LABELS: {len(X_manual)} examples")
        print(f"   Topic distribution:")
        for label_id, count in y_manual.value_counts().sort_index().items():
            label_name = self.topic_labels[label_id]
            print(f"     {label_id}: {label_name:<30} {count:>5}")
        
        # Add high-confidence keyword labels if provided
        if keyword_df is not None:
            print(f"\n📊 KEYWORD LABELS: Processing {len(keyword_df)} examples...")
            
            keyword_df = keyword_df.copy()
            
            # Filter for high confidence AND classified
            high_conf_mask = (
                (keyword_df['prediction_confidence'] >= keyword_confidence_threshold) &
                (keyword_df['predicted_label_id'].notna()) &
                (keyword_df['predicted_label_name'] != 'Unclassified')
            )
            
            keyword_df_filtered = keyword_df[high_conf_mask].copy()
            print(f"   Filtered to {len(keyword_df_filtered)} high-confidence examples "
                  f"(threshold: {keyword_confidence_threshold})")
            
            if len(keyword_df_filtered) > 0:
                # Process keyword data
                if isinstance(text_cols, list):
                    # Only use columns that exist
                    available_cols = [col for col in text_cols if col in keyword_df_filtered.columns]
                    if not available_cols:
                        raise ValueError(f"None of the text columns {text_cols} found in keyword_df. Available: {keyword_df_filtered.columns.tolist()}")
                    keyword_df_filtered['combined_text'] = keyword_df_filtered[available_cols].fillna('').agg(' '.join, axis=1)
                else:
                    if text_cols not in keyword_df_filtered.columns:
                        raise ValueError(f"Text column '{text_cols}' not found in keyword_df. Available: {keyword_df_filtered.columns.tolist()}")
                    keyword_df_filtered['combined_text'] = keyword_df_filtered[text_cols].fillna('')
                
                keyword_df_filtered['processed_text'] = keyword_df_filtered['combined_text'].apply(self.preprocess_text)
                
                X_keyword = keyword_df_filtered['processed_text']
                y_keyword = keyword_df_filtered['predicted_label_id'].astype(int)
                
                print(f"   Topic distribution:")
                for label_id, count in y_keyword.value_counts().sort_index().items():
                    label_name = self.topic_labels[label_id]
                    print(f"     {label_id}: {label_name:<30} {count:>5}")
                
                # Combine manual + keyword labels
                X_combined = pd.concat([X_manual, X_keyword], ignore_index=True)
                y_combined = pd.concat([y_manual, y_keyword], ignore_index=True)
                
                print(f"\n✅ TOTAL TRAINING DATA: {len(X_combined)} examples")
                print(f"   - Manual: {len(X_manual)} ({len(X_manual)/len(X_combined)*100:.1f}%)")
                print(f"   - Keyword (high-conf): {len(X_keyword)} ({len(X_keyword)/len(X_combined)*100:.1f}%)")
            else:
                print("   ⚠️  No high-confidence keyword labels found, using manual only")
                X_combined = X_manual
                y_combined = y_manual
        else:
            print("   ℹ️  No keyword data provided, using manual labels only")
            X_combined = X_manual
            y_combined = y_manual
        
        print(f"\n📈 COMBINED TOPIC DISTRIBUTION:")
        for label_id, count in y_combined.value_counts().sort_index().items():
            label_name = self.topic_labels[label_id]
            print(f"   {label_id}: {label_name:<30} {count:>5}")
        
        # Create pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                sublinear_tf=True,
                stop_words='english'
            )),
            ('clf', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                C=1.0,
                random_state=42
            ))
        ])
        
        # Cross-validation
        if do_cross_validation:
            print(f"\n🔄 Running {cv_folds}-fold cross-validation...")
            cv_scores = cross_val_score(self.model, X_combined, y_combined, 
                                       cv=cv_folds, scoring='accuracy', n_jobs=-1)
            prec_scores = cross_val_score(self.model, X_combined, y_combined, 
                                         cv=cv_folds, scoring='precision_weighted', n_jobs=-1)
            f1_scores = cross_val_score(self.model, X_combined, y_combined,
                                       cv=cv_folds, scoring='f1_weighted', n_jobs=-1)
            
            print(f"   Accuracy:  {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
            print(f"   Precision: {prec_scores.mean():.3f} (+/- {prec_scores.std():.3f})")
            print(f"   F1 Score:  {f1_scores.mean():.3f} (+/- {f1_scores.std():.3f})")
        
        # Train final model
        print("\n🎯 Training final model...")
        self.model.fit(X_combined, y_combined)
        self.trained = True
        
        # Training set performance
        y_pred = self.model.predict(X_combined)
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            y_combined, y_pred, average='weighted', zero_division=0
        )
        train_acc = (y_combined == y_pred).mean()
        
        print("\n" + "="*80)
        print("TRAINING SET PERFORMANCE")
        print("="*80)
        print(classification_report(y_combined, y_pred, 
                                   target_names=list(self.topic_labels.values()),
                                   zero_division=0))
        
        # Check agreement with manual labels
        if len(X_manual) > 0:
            y_manual_pred = self.model.predict(X_manual)
            manual_accuracy = (y_manual == y_manual_pred).mean()
            print(f"\n✅ Agreement with manual labels: {manual_accuracy:.1%}")
            
            disagreements = (y_manual != y_manual_pred).sum()
            if disagreements > 0:
                print(f"   ⚠️  {disagreements} disagreements - review these for quality")
        
        # Store metrics
        if do_cross_validation:
            self.metrics = {
                'cv_accuracy_mean': float(cv_scores.mean()),
                'cv_accuracy_std': float(cv_scores.std()),
                'cv_precision_mean': float(prec_scores.mean()),
                'cv_precision_std': float(prec_scores.std()),
                'cv_f1_mean': float(f1_scores.mean()),
                'cv_f1_std': float(f1_scores.std()),
                'train_accuracy': float(train_acc),
                'train_precision': float(train_prec),
                'train_f1': float(train_f1)
            }
        else:
            self.metrics = {
                'train_accuracy': float(train_acc),
                'train_precision': float(train_prec),
                'train_f1': float(train_f1)
            }
        
        print("\n✅ Training complete!")
        return self
    
    def predict(self, df, text_cols=['title', 'description']):
        """Predict topics using trained ML model"""
        if not self.trained:
            raise ValueError("Model not trained. Call train_semi_supervised() first.")
        
        print("\n" + "="*80)
        print("ML PREDICTIONS")
        print("="*80)
        
        df = df.copy()
        
        # Combine text
        if isinstance(text_cols, list):
            df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
        else:
            df['combined_text'] = df[text_cols].fillna('')
        
        # Preprocess
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        # Predict
        print(f"🤖 Running ML predictions on {len(df)} articles...")
        X = df['processed_text']
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        confidences = probabilities.max(axis=1)
        
        df['ml_predicted_label_id'] = predictions
        df['ml_predicted_label_name'] = df['ml_predicted_label_id'].map(self.topic_labels)
        df['ml_prediction_confidence'] = confidences
        
        print(f"\n✅ Prediction complete!")
        print(f"   Average confidence: {confidences.mean():.3f}")
        print(f"   Median confidence: {np.median(confidences):.3f}")
        
        print(f"\n📊 TOPIC DISTRIBUTION:")
        print(df['ml_predicted_label_name'].value_counts())
        
        return df
    
    def save_model(self, path='english_topic_model.pkl'):
        """Save trained model"""
        if not self.trained:
            raise ValueError("No trained model to save")
        joblib.dump(self.model, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path='english_topic_model.pkl'):
        """Load trained model"""
        self.model = joblib.load(path)
        self.trained = True
        print(f"✅ Model loaded from {path}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

if __name__ == "__main__":
    
    # Setup paths
    DATA_DIR = Path(__file__).parent.parent / "data_world"
    
    # Input files
    UNLABELED_FILE = DATA_DIR / "world_newsE_combined.csv"  # Raw data to label
    MANUAL_FILE = DATA_DIR / "world_newsE_review_200.csv"   # Manual labels
    
    # Output files
    KEYWORD_OUTPUT = DATA_DIR / "world_newsE_keyword_labeled.csv"  # Step 1 output
    FINAL_OUTPUT = DATA_DIR / "world_newsE_final_labeled.csv"      # Step 2 output
    MODEL_FILE = DATA_DIR / "english_topic_model.pkl"
    
    print("\n" + "="*80)
    print("ENGLISH TOPIC LABELER - TWO-STAGE PIPELINE")
    print("="*80)
    print("\nStage 1: Keyword-based labeling")
    print("Stage 2: Semi-supervised learning (manual + high-confidence keyword)")
    
    # ========================================================================
    # STAGE 1: KEYWORD-BASED LABELING
    # ========================================================================
    print("\n" + "="*80)
    print("STAGE 1: KEYWORD-BASED LABELING")
    print("="*80)
    
    # Check if unlabeled file exists
    if not UNLABELED_FILE.exists():
        print(f"\n❌ ERROR: File not found at {UNLABELED_FILE}")
        print(f"\nPlease ensure 'world_newsE_combined.csv' is in: {DATA_DIR}")
        exit(1)
    
    print(f"\n✓ Found data file: {UNLABELED_FILE}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(UNLABELED_FILE)
    print(f"Loaded {len(df)} articles")
    
    # Initialize labeler
    labeler = EnglishTopicLabeler()
    
    # Apply keyword labeling
    keyword_labeled_df = labeler.keyword_label(
        df=df,
        text_cols=['title', 'description'],
        min_confidence=0.3
    )
    
    # Save keyword results
    keyword_labeled_df.to_csv(KEYWORD_OUTPUT, index=False)
    print(f"\n✅ Keyword-labeled data saved to: {KEYWORD_OUTPUT}")
    
    # ========================================================================
    # STAGE 2: SEMI-SUPERVISED LEARNING
    # ========================================================================
    print("\n" + "="*80)
    print("STAGE 2: SEMI-SUPERVISED LEARNING")
    print("="*80)
    
    # Check if manual labels exist
    if not MANUAL_FILE.exists():
        print(f"\n❌ ERROR: Manual labels not found at {MANUAL_FILE}")
        print(f"\nPlease ensure 'world_newsE_review_200.csv' is in: {DATA_DIR}")
        print("\nSkipping Stage 2. You can:")
        print("  1. Manually label some articles")
        print("  2. Save them as 'world_newsE_review_200.csv'")
        print("  3. Re-run this script")
        exit(1)
    
    print(f"\n✓ Found manual labels: {MANUAL_FILE}")
    
    # Load manual labels
    print("\nLoading manual labels...")
    manual_df = pd.read_csv(MANUAL_FILE)
    print(f"Loaded {len(manual_df)} manually labeled articles")
    
    # Check manual file has required columns
    if 'topic_id' not in manual_df.columns:
        print(f"\n❌ ERROR: 'topic_id' column not found in manual file")
        print(f"Available columns: {manual_df.columns.tolist()}")
        exit(1)
    
    # Train semi-supervised model
    labeler.train_semi_supervised(
        manual_df=manual_df,
        keyword_df=keyword_labeled_df,
        keyword_confidence_threshold=0.6,  # Only use high-confidence keyword labels
        text_cols=['title', 'description'],
        manual_label_col='topic_id',
        do_cross_validation=True,
        cv_folds=5
    )
    
    # Save model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    labeler.save_model(MODEL_FILE)
    
    # ========================================================================
    # FINAL PREDICTIONS
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL PREDICTIONS ON ALL DATA")
    print("="*80)
    
    # Predict on all data using trained model
    final_df = labeler.predict(
        df=df,
        text_cols=['title', 'description']
    )
    
    # Add keyword predictions for comparison
    final_df['keyword_label_name'] = keyword_labeled_df['predicted_label_name']
    final_df['keyword_label_id'] = keyword_labeled_df['predicted_label_id']
    final_df['keyword_confidence'] = keyword_labeled_df['prediction_confidence']
    
    # Save final results
    final_df.to_csv(FINAL_OUTPUT, index=False)
    print(f"\n✅ Final labeled data saved to: {FINAL_OUTPUT}")
    
    # ========================================================================
    # COMPARISON: KEYWORD vs ML
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON: KEYWORD vs ML PREDICTIONS")
    print("="*80)
    
    # Compare on classified articles
    classified_mask = final_df['keyword_label_name'] != 'Unclassified'
    comparison_df = final_df[classified_mask].copy()
    
    if len(comparison_df) > 0:
        agreement = (comparison_df['keyword_label_name'] == comparison_df['ml_predicted_label_name']).mean()
        print(f"\n📊 Agreement rate: {agreement:.1%}")
        print(f"   ({(comparison_df['keyword_label_name'] == comparison_df['ml_predicted_label_name']).sum()} / {len(comparison_df)} articles)")
        
        # Show disagreements
        disagreements = comparison_df[comparison_df['keyword_label_name'] != comparison_df['ml_predicted_label_name']]
        if len(disagreements) > 0:
            print(f"\n⚠️  {len(disagreements)} disagreements found")
            print("\nTop 5 disagreements:")
            cols = ['keyword_label_name', 'ml_predicted_label_name', 'keyword_confidence', 'ml_prediction_confidence']
            print(disagreements[cols].head())
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    
    print(f"\n📁 Output files:")
    print(f"   • Stage 1 (keyword): {KEYWORD_OUTPUT}")
    print(f"   • Stage 2 (ML): {FINAL_OUTPUT}")
    print(f"   • Model: {MODEL_FILE}")
    
    print(f"\n📊 Summary:")
    print(f"   Total articles: {len(df)}")
    print(f"   Manual labels used: {len(manual_df)}")
    print(f"   Keyword classified: {classified_mask.sum()}")
    print(f"   Final ML predictions: {len(final_df)}")
    
    print("\n💡 Next steps:")
    print("   1. Review predictions in world_newsE_final_labeled.csv")
    print("   2. Compare 'keyword_label_name' vs 'ml_predicted_label_name'")
    print("   3. For better results, add more manual labels and retrain")
    print("\n   To retrain with more manual labels:")
    print("   1. Add more labels to world_newsE_review_200.csv")
    print("   2. Re-run this script")