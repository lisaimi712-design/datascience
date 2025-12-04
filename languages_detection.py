import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect, detect_langs, LangDetectException
from collections import Counter

class LanguageAnalyzer:
    """
    Detects and analyzes languages in your collected news dataset
    Helps decide between DistilBERT (English) vs AfriBERTa (Multilingual)
    """
    
    def __init__(self, data_path: str = None, dataframe: pd.DataFrame = None):
        """
        Initialize with either a CSV file path or a DataFrame
        
        Args:
            data_path: Path to CSV file with collected news data
            dataframe: Or pass DataFrame directly
        """
        if dataframe is not None:
            self.df = dataframe
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Must provide either data_path or dataframe")
        
        # Language code to name mapping (expanded for African languages)
        self.language_names = {
            'en': 'English',
            'fr': 'French',
            'ar': 'Arabic',
            'pt': 'Portuguese',
            'sw': 'Swahili',
            'am': 'Amharic',
            'ha': 'Hausa',
            'yo': 'Yoruba',
            'ig': 'Igbo',
            'zu': 'Zulu',
            'af': 'Afrikaans',
            'so': 'Somali',
            'es': 'Spanish',
            'it': 'Italian',
            'de': 'German',
            'unknown': 'Unknown'
        }
        
        # Languages supported by different BERT models
        self.model_language_support = {
            'DistilBERT': ['en'],
            'AfriBERTa': ['en', 'am', 'ha', 'ig', 'om', 'pcm', 'rw', 'so', 'sw', 'ti', 'yo'],
            'mBERT': ['en', 'fr', 'ar', 'pt', 'sw', 'am', 'ha', 'yo', 'ig', 'zu', 'af', 'so'],
            'XLM-RoBERTa': ['en', 'fr', 'ar', 'pt', 'sw', 'am', 'ha', 'yo', 'ig', 'zu', 'af', 'so']
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of a single text
        
        Args:
            text: Text to analyze
        
        Returns:
            ISO language code (e.g., 'en', 'fr', 'ar')
        """
        if pd.isna(text) or not text or len(str(text).strip()) < 10:
            return 'unknown'
        
        try:
            return detect(str(text))
        except LangDetectException:
            return 'unknown'
    
    def detect_language_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Detect language with confidence score
        
        Args:
            text: Text to analyze
        
        Returns:
            Tuple of (language_code, confidence_score)
        """
        if pd.isna(text) or not text or len(str(text).strip()) < 10:
            return ('unknown', 0.0)
        
        try:
            langs = detect_langs(str(text))
            if langs:
                return (langs[0].lang, langs[0].prob)
            return ('unknown', 0.0)
        except LangDetectException:
            return ('unknown', 0.0)
    
    def analyze_dataset(self, text_column: str = 'title') -> pd.DataFrame:
        """
        Detect languages for all texts in dataset
        
        Args:
            text_column: Column name containing text to analyze (default: 'title')
        
        Returns:
            DataFrame with added language detection columns
        """
        print(f"Analyzing languages in '{text_column}' column...")
        print(f"Total articles: {len(self.df)}")
        
        # Detect language for each text
        results = []
        for idx, text in enumerate(self.df[text_column]):
            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(self.df)} articles...", end='\r')
            
            lang, confidence = self.detect_language_with_confidence(text)
            results.append({
                'language_code': lang,
                'language_confidence': confidence,
                'language_name': self.language_names.get(lang, lang.upper())
            })
        
        print(f"  Processed {len(self.df)}/{len(self.df)} articles... Done!")
        
        # Add results to dataframe
        results_df = pd.DataFrame(results)
        self.df = pd.concat([self.df, results_df], axis=1)
        
        return self.df
    
    def get_language_statistics(self) -> pd.DataFrame:
        """
        Get statistics about detected languages
        
        Returns:
            DataFrame with language counts and percentages
        """
        if 'language_code' not in self.df.columns:
            raise ValueError("Run analyze_dataset() first to detect languages")
        
        # Count languages
        lang_counts = self.df['language_name'].value_counts().reset_index()
        lang_counts.columns = ['Language', 'Count']
        lang_counts['Percentage'] = (lang_counts['Count'] / len(self.df) * 100).round(2)
        
        # Add average confidence
        avg_confidence = self.df.groupby('language_name')['language_confidence'].mean().round(3)
        lang_counts['Avg_Confidence'] = lang_counts['Language'].map(avg_confidence)
        
        return lang_counts
    
    def recommend_bert_model(self) -> Dict:
        """
        Recommend which BERT model to use based on language distribution
        
        Returns:
            Dictionary with recommendation and reasoning
        """
        if 'language_code' not in self.df.columns:
            raise ValueError("Run analyze_dataset() first to detect languages")
        
        # Get language distribution
        lang_counts = self.df['language_code'].value_counts()
        total = len(self.df)
        
        # Calculate coverage for each model
        model_coverage = {}
        for model_name, supported_langs in self.model_language_support.items():
            covered = sum(lang_counts.get(lang, 0) for lang in supported_langs)
            coverage_pct = (covered / total * 100) if total > 0 else 0
            model_coverage[model_name] = {
                'coverage_pct': coverage_pct,
                'articles_covered': covered,
                'supported_languages': supported_langs
            }
        
        # Determine best model
        best_model = max(model_coverage.items(), key=lambda x: x[1]['coverage_pct'])
        
        # Build recommendation
        recommendation = {
            'recommended_model': best_model[0],
            'coverage_percentage': round(best_model[1]['coverage_pct'], 2),
            'articles_covered': best_model[1]['articles_covered'],
            'total_articles': total,
            'all_model_coverage': model_coverage,
            'reasoning': self._build_reasoning(lang_counts, model_coverage, total)
        }
        
        return recommendation
    
    def _build_reasoning(self, lang_counts: pd.Series, 
                        model_coverage: Dict, total: int) -> str:
        """Build explanation for model recommendation"""
        
        english_pct = (lang_counts.get('en', 0) / total * 100) if total > 0 else 0
        
        reasoning_parts = []
        
        # Language distribution
        top_3_langs = lang_counts.head(3)
        lang_summary = ", ".join([
            f"{self.language_names.get(lang, lang)}: {count} ({count/total*100:.1f}%)" 
            for lang, count in top_3_langs.items()
        ])
        reasoning_parts.append(f"Top languages: {lang_summary}")
        
        # Model-specific reasoning
        if english_pct >= 95:
            reasoning_parts.append(
                f"Dataset is {english_pct:.1f}% English → DistilBERT is optimal (faster, efficient)"
            )
        elif english_pct >= 80:
            reasoning_parts.append(
                f"Dataset is {english_pct:.1f}% English → DistilBERT works, but consider multilingual if other languages matter"
            )
        else:
            multilang_pct = 100 - english_pct
            reasoning_parts.append(
                f"Dataset has {multilang_pct:.1f}% non-English content → Multilingual model required"
            )
        
        # African language presence
        african_langs = set(['am', 'ha', 'ig', 'so', 'sw', 'yo', 'zu'])
        detected_african = set(lang_counts.index) & african_langs
        if detected_african:
            african_lang_names = [self.language_names.get(lang, lang) for lang in detected_african]
            reasoning_parts.append(
                f"African languages detected: {', '.join(african_lang_names)} → AfriBERTa recommended"
            )
        
        return " | ".join(reasoning_parts)
    
    def visualize_languages(self, save_path: str = None):
        """
        Create visualizations of language distribution
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if 'language_name' not in self.df.columns:
            raise ValueError("Run analyze_dataset() first to detect languages")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Language Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Language distribution (pie chart)
        lang_counts = self.df['language_name'].value_counts()
        axes[0, 0].pie(lang_counts.values, labels=lang_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Language Distribution')
        
        # 2. Top 10 languages (bar chart)
        top_10 = lang_counts.head(10)
        axes[0, 1].barh(range(len(top_10)), top_10.values)
        axes[0, 1].set_yticks(range(len(top_10)))
        axes[0, 1].set_yticklabels(top_10.index)
        axes[0, 1].set_xlabel('Number of Articles')
        axes[0, 1].set_title('Top 10 Languages')
        axes[0, 1].invert_yaxis()
        
        # 3. Confidence distribution
        axes[1, 0].hist(self.df['language_confidence'], bins=30, edgecolor='black')
        axes[1, 0].set_xlabel('Detection Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Language Detection Confidence Distribution')
        axes[1, 0].axvline(self.df['language_confidence'].mean(), 
                          color='red', linestyle='--', label='Mean')
        axes[1, 0].legend()
        
        # 4. Model coverage comparison
        if 'language_code' in self.df.columns:
            recommendation = self.recommend_bert_model()
            model_names = list(recommendation['all_model_coverage'].keys())
            coverages = [recommendation['all_model_coverage'][m]['coverage_pct'] 
                        for m in model_names]
            
            bars = axes[1, 1].bar(model_names, coverages)
            axes[1, 1].set_ylabel('Coverage (%)')
            axes[1, 1].set_title('BERT Model Coverage Comparison')
            axes[1, 1].set_ylim(0, 100)
            
            # Highlight recommended model
            recommended_idx = model_names.index(recommendation['recommended_model'])
            bars[recommended_idx].set_color('green')
            bars[recommended_idx].set_alpha(0.7)
            
            # Add percentage labels
            for i, (name, coverage) in enumerate(zip(model_names, coverages)):
                axes[1, 1].text(i, coverage + 2, f'{coverage:.1f}%', 
                              ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def get_sample_by_language(self, language: str, n: int = 5) -> pd.DataFrame:
        """
        Get sample articles in a specific language
        
        Args:
            language: Language code (e.g., 'en', 'fr') or name (e.g., 'English')
            n: Number of samples to return
        
        Returns:
            DataFrame with sample articles
        """
        if 'language_code' not in self.df.columns:
            raise ValueError("Run analyze_dataset() first")
        
        # Handle both language code and name
        if language in self.language_names.values():
            # It's a language name, find the code
            lang_code = [k for k, v in self.language_names.items() if v == language][0]
            filtered = self.df[self.df['language_code'] == lang_code]
        else:
            # Assume it's a code
            filtered = self.df[self.df['language_code'] == language]
        
        return filtered.head(n)[['title', 'language_name', 'language_confidence', 'source_name']]
    
    def export_by_language(self, output_dir: str = 'data_by_language'):
        """
        Export separate CSV files for each language
        Useful for training language-specific models
        
        Args:
            output_dir: Directory to save language-specific files
        """
        import os
        
        if 'language_code' not in self.df.columns:
            raise ValueError("Run analyze_dataset() first")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for lang_code in self.df['language_code'].unique():
            if lang_code == 'unknown':
                continue
            
            lang_name = self.language_names.get(lang_code, lang_code)
            lang_df = self.df[self.df['language_code'] == lang_code]
            
            filename = f"{output_dir}/{lang_name.lower()}_{lang_code}_articles.csv"
            lang_df.to_csv(filename, index=False)
            print(f"Exported {len(lang_df)} {lang_name} articles to: {filename}")


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Initialize analyzer with your collected data
    analyzer = LanguageAnalyzer(data_path='african_news_clean.csv')
    
    # Or if you already have a DataFrame:
    # analyzer = LanguageAnalyzer(dataframe=your_dataframe)
    
    # Detect languages in the dataset
    print("=" * 60)
    print("STEP 1: Detecting Languages")
    print("=" * 60)
    df_with_languages = analyzer.analyze_dataset(text_column='title')
    
    # Get language statistics
    print("\n" + "=" * 60)
    print("STEP 2: Language Statistics")
    print("=" * 60)
    stats = analyzer.get_language_statistics()
    print(stats)
    
    # Get model recommendation
    print("\n" + "=" * 60)
    print("STEP 3: BERT Model Recommendation")
    print("=" * 60)
    recommendation = analyzer.recommend_bert_model()
    print(f"\nRECOMMENDED MODEL: {recommendation['recommended_model']}")
    print(f"Coverage: {recommendation['coverage_percentage']}% ({recommendation['articles_covered']}/{recommendation['total_articles']} articles)")
    print(f"\nReasoning: {recommendation['reasoning']}")
    
    print("\n" + "=" * 60)
    print("All Model Coverage:")
    print("=" * 60)
    for model, info in recommendation['all_model_coverage'].items():
        print(f"{model}: {info['coverage_pct']:.2f}% coverage")
    
    # Visualize language distribution
    print("\n" + "=" * 60)
    print("STEP 4: Creating Visualizations")
    print("=" * 60)
    analyzer.visualize_languages(save_path='language_analysis.png')
    
    # Get samples in different languages
    print("\n" + "=" * 60)
    print("Sample Articles by Language")
    print("=" * 60)
    print("\nEnglish samples:")
    print(analyzer.get_sample_by_language('en', n=3))
    
    # Export data by language (optional)
    # analyzer.export_by_language(output_dir='data_by_language')
    
    # Save the DataFrame with language info
    df_with_languages.to_csv('african_news_with_languages.csv', index=False)
    print("\n✓ Data with language detection saved to: african_news_with_languages.csv")