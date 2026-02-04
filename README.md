# 👥 MBTI Text Classifier

![GitHub repo size](https://img.shields.io/github/repo-size/mr-mrf-dev/mbti-text-classifier)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
[![GitHub License](https://img.shields.io/github/license/mr-mrf-dev/mbti-text-classifier)](/LICENSE)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green.svg)

A machine learning project that classifies text into one of the 16 MBTI personality types based on user posts. This project uses classic ML techniques with TF-IDF feature extraction to analyze text and predict personality types.

## 📋 Overview

This project implements a complete ML pipeline for MBTI personality classification:

- 📥 Downloads and preprocesses the MBTI dataset from Kaggle
- 🧹 Performs text cleaning and TF-IDF feature extraction
- ⚖️ Handles class imbalance using Random Oversampling
- 🤖 Trains and evaluates three classic ML models
- 📊 Provides detailed error analysis and visualizations

## 🎯 MBTI Types

The Myers-Briggs Type Indicator consists of 16 personality types based on four dimensions:

| Dimension | Description |
| ----------- | ------------- |
| **E/I** | Extraversion vs Introversion |
| **S/N** | Sensing vs Intuition |
| **T/F** | Thinking vs Feeling |
| **J/P** | Judging vs Perceiving |

## 📁 Project Structure

```text
MBTI-Text-Classifier/
├── notebook.ipynb           # Main Jupyter notebook
├── requirements.txt         # Python dependencies
├── .pre-commit-config.yaml  # Pre-commit hooks
├── .gitignore
├── LICENSE
├── README.md
├── data/
│   ├── raw_data.csv         # Original dataset
│   └── processed_data.csv   # Cleaned dataset
├── models/
│   ├── svm_model.pkl        # Trained best model
│   └── tfidf_vectorizer.pkl # TF-IDF vectorizer
└── images/                  # Generated visualizations
    ├── 01_mbti_distribution.png
    ├── 02_original_class_distribution.png
    ├── 03_resampled_class_distribution.png
    ├── 04_model_performance_comparison.png
    ├── 05_confusion_matrices.png
    ├── 06_per_type_accuracy.png
    └── 07_dimension_accuracy.png
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Mr-MRF-Dev/MBTI-Text-Classifier.git
   cd MBTI-Text-Classifier
   ```

2. **Create and activate virtual environment:**

   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install pre-commit hooks:**

   ```bash
   pre-commit install
   ```

5. **Run the notebook:**

   ```bash
   jupyter notebook notebook.ipynb
   ```

   or using JupyterLab:

   ```bash
    jupyter lab notebook.ipynb
    ```

## 🤖 Models

Three classic ML models are trained and compared:

| Model | Accuracy | F1-Score |
| ------- | ---------- | ---------- |
| **SVM** | ~90% | ~0.90 |
| Logistic Regression | ~88% | ~0.87 |
| Naive Bayes | ~78% | ~0.78 |

## 📊 Results

The **SVM** model achieves the best performance:

- ✅ **Accuracy**: ~90%
- ✅ **F1-Score**: ~0.90
- ✅ **Best dimension**: S/N (~98%)
- ⚠️ **Hardest dimension**: J/P (~94%)

## 📈 Visualizations

The notebook generates various visualizations saved in the `images/` folder:

- MBTI type distribution
- Class distribution (before/after oversampling)
- Model performance comparison
- Confusion matrices for all models
- Per-type classification accuracy
- Dimension-level accuracy (E/I, S/N, T/F, J/P)

## 🛠️ Tech Stack

| Category | Technologies |
| ---------- | -------------- |
| Language | Python |
| Data | Pandas, NumPy |
| ML | Scikit-learn, imbalanced-learn |
| NLP | NLTK, TF-IDF |
| Visualization | Matplotlib, Seaborn |
| Dataset | KaggleHub |

## 🔧 Pre-commit Hooks

This project uses pre-commit for code quality:

- **black** - Code formatting
- **isort** - Import sorting
- **flake8** - Linting
- **nbqa** - Notebook code quality

## 🙏 Acknowledgments

- Dataset: [MBTI Dataset on Kaggle](https://www.kaggle.com/datasets/datasnaek/mbti-type)

## 🤝 Contributing

We welcome any contributions you may have. If you're interested in helping out, please fork the repository and create an [Issue](https://github.com/Mr-MRF-Dev/MBTI-Text-Classifier/issues) or [Pull Request](https://github.com/Mr-MRF-Dev/MBTI-Text-Classifier/pulls). We'll be happy to review your contributions.

## 📝 License

This project is licensed under the MIT License. For more information, please refer to the [LICENSE](/LICENSE) file.
