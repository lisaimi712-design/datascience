# African News Topic Classification & Analysis

This project provides a comprehensive pipeline for collecting, processing, and analyzing African news articles using machine learning models (BERT, DistilBERT, LDA) and various data visualization techniques.

## Overview

The project consists of:
- **Data Collection**: Fetching news from multiple sources (Guardian API, Kaggle, RSS feeds, NewsAPI, GDELT)
- **Data Quality**: Year inference pipeline with multi-method fallback strategies
- **Topic Modeling**: LDA and BERT-based classification for 6 topics
- **Model Training**: Advanced BERT fine-tuning with layer-wise learning rates and early stopping
- **Visualization**: Comprehensive charts comparing African vs World news distributions

## Topics

1. Economic Development
2. Natural Resources & Energy
3. War & Conflict
4. Social Services
5. Politics & Governance
6. Art, Technology and Sport

## Quick Start

1. Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Run scripts as needed (see sections below)

## Project Structure

```
├── data_africa/              # African news datasets
│   ├── ldamulti_final.csv   # Final processed African news (21,827 articles)
│   └── ldamulti_complete.csv # Complete with inferred years
├── data_world/              # World news datasets
│   ├── world_newsE_topics_assignments_complete.csv (22,602 articles)
│   ├── bert_english_classifier/  # Trained BERT model
│   │   ├── visualizations/      # Model evaluation plots
│   │   └── model files...
│   └── fetch scripts...
├── model/                   # ML model training scripts
│   ├── bert.py             # Advanced BERT with layer-wise LR
│   ├── DistillBert.py      # DistilBERT classifier
│   ├── TF-IDF.py           # TF-IDF baseline
│   └── Lora.py             # LoRA fine-tuning
├── combined_visualizations/ # Dataset comparison charts
└── visualization scripts...
```

## Data Collection

### Guardian API
```powershell
.\.venv\Scripts\python.exe data_world\fetch_guardian_api.py
```

### Kaggle Datasets
```powershell
.\.venv\Scripts\python.exe data_world\fetch_kaggle_datasets.py
```

## Data Processing

### Year Inference (Multi-Method)
Complete missing publication years using date parsing, text extraction, source medians, and collection dates:
```powershell
.\.venv\Scripts\python.exe infer_years_advanced.py
```

**Results:**
- African News: 100% year coverage (21,827 articles)
- World News: 100% year coverage (22,602 articles)

### Language Detection
```powershell
.\.venv\Scripts\python.exe languages_detection.py
```

## Topic Modeling

### LDA Multi-Language
```powershell
.\.venv\Scripts\python.exe lda_english.py
```

### LDA with TF-IDF
```powershell
.\.venv\Scripts\python.exe ldamulti_it-idf.py
```

## Model Training

### BERT Advanced (Recommended)
Fine-tuned BERT with:
- Layer-wise learning rate decay (2e-5 top → 5e-6 lower)
- Unfrozen last 3 transformer layers
- Early stopping on validation F1 (patience=5)

```powershell
.\.venv\Scripts\python.exe model\bert.py
```

### DistilBERT
```powershell
.\.venv\Scripts\python.exe model\DistillBert.py
```

### TF-IDF Baseline
```powershell
.\.venv\Scripts\python.exe model\TF-IDF.py
```

## Visualization

### Model Evaluation (4 plots)
- Confusion matrix
- ROC-AUC curves (per-class + micro-average 0.9615)
- t-SNE embeddings
- Confidence distributions

```powershell
.\.venv\Scripts\python.exe data_world\vision.py
```

**Output:** `data_world/bert_english_classifier/visualizations/`

### Dataset Comparison (8 plots)
```powershell
.\.venv\Scripts\python.exe combined_data_visualization.py
```

Generates:
1. Topic distribution comparison
2. Year distribution comparison
3. Language distribution comparison
4. Source distribution comparison
5. Dataset summary table
6. World topics by year heatmap
7. African topics by year heatmap
8. Articles trend over time

### Per-Topic Year Comparison (6 plots)
```powershell
.\.venv\Scripts\python.exe per_topic_year_comparison.py
```

### Cartesian Topic Distribution
```powershell
.\.venv\Scripts\python.exe simple_cartesian_topics.py
```

Generates clean Cartesian diagrams with 6 topic lines showing article counts over years.

### Topic Distribution Visualizations
```powershell
.\.venv\Scripts\python.exe data_world\visualizations.py
```

Generates:
- `topic_distribution_count.png` - Absolute counts
- `topic_distribution_percentage.png` - Percentage view

## Key Files

### Data Processing
- `infer_years_advanced.py` - Multi-method year inference
- `check_data_availability.py` - Data quality investigation
- `assign_default_years.py` - Source-aware default assignment
- `languages_detection.py` - Language detection and analysis

### Visualization Scripts
- `combined_data_visualization.py` - 8 dataset comparison charts
- `per_topic_year_comparison.py` - 6 per-topic analyses
- `cartesian_topic_distribution.py` - 5 Cartesian views
- `simple_cartesian_topics.py` - Clean 6-line topic trends
- `data_world/visualizations.py` - Topic distribution plots
- `data_world/vision.py` - Model evaluation visualizations

### Model Scripts
- `model/bert.py` - Advanced BERT training
- `model/DistillBert.py` - DistilBERT classifier
- `model/TF-IDF.py` - Baseline TF-IDF model
- `model/Lora.py` - LoRA fine-tuning
- `model/bert_multi_aggressive.py` - Aggressive BERT fine-tuning

## Dataset Statistics

### African News (ldamulti_final.csv)
- **Total Articles:** 21,827
- **Year Range:** 2017-2025 (peak: 2024-2025)
- **Sources:** GDELT, African RSS, NewsAPI
- **Languages:** Multiple (detected)
- **Topics:** 6 categories via ML prediction
- **Year Coverage:** 100% complete

### World News (world_newsE_topics_assignments_complete.csv)
- **Total Articles:** 22,602
- **Year Range:** 1900-2099 (main: 2017-2025)
- **Sources:** 46 sources (Guardian 7,913 + Kaggle 14,300 + others)
- **Languages:** Multiple
- **Topics:** 6 categories via LDA assignment
- **Year Coverage:** 100% complete

## Model Performance

### BERT English Classifier
- **Micro-average AUC:** 0.9615
- **Per-class AUC:** 0.9346 - 0.9744
- **Test samples:** 300 articles (256 max tokens)
- **Location:** `data_world/bert_english_classifier/`

### Topic Distribution (African News)
- Economic Development: 3,762 articles (17.2%)
- Natural Resources & Energy: 1,872 articles (8.6%)
- Politics & Governance: 1,913 articles (8.8%)
- Social Services: 1,736 articles (8.0%)
- War & Conflict: 1,195 articles (5.5%)
- Art, Technology and Sport: 1,045 articles (4.8%)

### Topic Distribution (World News)
- Natural Resources & Energy: 5,336 articles (23.6%)
- Politics & Governance: 4,805 articles (21.3%)
- Economic Development: 3,952 articles (17.5%)
- Social Services: 3,706 articles (16.4%)
- Art, Technology and Sport: 2,945 articles (13.0%)
- War & Conflict: 1,858 articles (8.2%)

## Requirements

Key dependencies (see `requirements.txt`):
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- transformers (Hugging Face)
- torch (PyTorch)
- datasets

## Output Directories

- `combined_visualizations/` - Comparison charts between datasets
- `data_world/bert_english_classifier/visualizations/` - Model evaluation plots
- Root directory - Individual visualization PNGs

## Notes

- All datasets have 100% year coverage after inference pipeline
- Year assignments are source-aware (realistic per publication)
- Model trained on `labeled_dataEFA_training_only.csv` (1,000 articles/class)
- Visualizations at 300 DPI for publication quality

## License

Project developed for African news analysis and topic classification research.
