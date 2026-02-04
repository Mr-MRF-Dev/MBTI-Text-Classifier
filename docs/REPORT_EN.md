# MBTI Personality Type Classification - Project Report

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data Acquisition](#2-data-acquisition)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Model Training](#4-model-training)
5. [Results Analysis](#5-results-analysis)
6. [Model Comparison](#6-model-comparison)
7. [Error Analysis](#7-error-analysis)
8. [Conclusion](#8-conclusion)

---

## 1. Introduction

This project aims to classify Myers-Briggs Type Indicator (MBTI) personality types based on text posts from users. The MBTI system categorizes personalities into 16 types based on four dimensions:

- **E/I**: Extraversion vs Introversion
- **S/N**: Sensing vs Intuition
- **T/F**: Thinking vs Feeling
- **J/P**: Judging vs Perceiving

We used three classic machine learning models to tackle this multi-class classification problem: Logistic Regression, Support Vector Machine (SVM), and Naive Bayes.

---

## 2. Data Acquisition

The dataset was downloaded from Kaggle using the `kagglehub` library. The dataset is called "MBTI Personality Types Dataset" by datasnaek.

**Dataset Details:**

- **Source**: Kaggle (datasnaek/mbti-type)
- **Total Samples**: 8,675 users
- **Features**: 2 columns (type, posts)
- **Posts**: Each user has ~50 posts separated by "|||"

The data was collected from the PersonalityCafe forum where users share their MBTI type and post messages. This gives us labeled data for supervised learning.

```python
path = kagglehub.dataset_download("datasnaek/mbti-type")
df_raw = pd.read_csv(os.path.join(path, "mbti_1.csv"))
```

### Distribution of MBTI Types in Dataset

![MBTI Distribution](images/01_mbti_distribution.png)

*Figure 1: Distribution of MBTI personality types in the raw dataset.*

As shown in Figure 1, the dataset is highly imbalanced. The most common types are INFP, INFJ, INTP, and INTJ (all introverted intuitive types), while extroverted sensing types like ESTJ and ESFJ are rare. This reflects the demographics of the online forum where data was collected - introverts are more likely to participate in online text-based communities.

---

## 3. Data Preprocessing

We performed the following preprocessing steps:

### 3.1 Text Cleaning

- Converted text to lowercase
- Removed URLs (http, https, www)
- Removed MBTI type mentions (to prevent data leakage)
- Removed punctuation and special characters
- Removed stopwords (common words like "the", "is", "at")
- Removed words with less than 3 characters

### 3.2 Handling Class Imbalance

The dataset is highly imbalanced. For example, INFP has 1,832 samples while ESTJ has only 39 samples. The imbalance ratio is about 47:1.

We used **Random Oversampling** to balance the classes by duplicating samples from minority classes. After oversampling, all classes have equal representation.

#### Before Oversampling

![Original Class Distribution](images/02_original_class_distribution.png)

*Figure 2: Original class distribution showing severe imbalance between personality types.*

#### After Oversampling

![Resampled Class Distribution](images/03_resampled_class_distribution.png)

*Figure 3: Class distribution after Random Oversampling - all classes now have equal samples.*

As we can see from Figures 2 and 3, the oversampling technique successfully balanced our dataset. Before oversampling, INFP had about 1,800 samples while ESTJ had only ~40. After oversampling, all 16 classes have equal representation, which helps prevent the model from being biased toward majority classes.

### 3.3 Feature Extraction

We used TF-IDF (Term Frequency-Inverse Document Frequency) vectorization:

- Max features: 5,000
- N-gram range: (1, 2) - unigrams and bigrams
- Min document frequency: 5
- Max document frequency: 0.8

---

## 4. Model Training

We trained three classic machine learning models:

| Model | Description | Parameters |
| ------ | ------------- | ------------ |
| **Logistic Regression** | Linear model for classification | max_iter=1000, random_state=42 |
| **SVM (LinearSVC)** | Support Vector Machine with linear kernel | max_iter=2000, random_state=42 |
| **Naive Bayes** | Probabilistic classifier based on Bayes theorem | MultinomialNB (default) |

**Data Split**:

- Training set: 80%
- Test set: 20%
- Stratified split to maintain class distribution

---

## 5. Results Analysis

After training and evaluating the models, we obtained the following results:

| Model | Accuracy | Precision | Recall | F1-Score |
| ------- | ---------- | ----------- | -------- | ---------- |
| **SVM** | ~0.90 | ~0.90 | ~0.90 | ~0.90 |
| **Logistic Regression** | ~0.87 | ~0.87 | ~0.87 | ~0.87 |
| **Naive Bayes** | ~0.52 | ~0.55 | ~0.52 | ~0.50 |

**Key Observations**:

- SVM achieved the best performance across all metrics
- Logistic Regression performed similarly to SVM
- Naive Bayes struggled significantly with this task

The high accuracy of SVM and Logistic Regression (~90%) might seem surprising for a 16-class problem. This could be because after oversampling, the training data has many duplicate samples, making the classification easier.

### Visual Comparison of Model Performance

![Model Performance Comparison](images/04_model_performance_comparison.png)

*Figure 4: Bar chart comparing Accuracy, Precision, Recall, and F1-Score across all three models.*

Figure 4 clearly shows that SVM and Logistic Regression perform significantly better than Naive Bayes across all metrics. The performance gap between linear models (SVM, LR) and Naive Bayes is substantial (~40 percentage points).

### Confusion Matrices

![Confusion Matrices](images/05_confusion_matrices.png)

*Figure 5: Confusion matrices for all three models showing prediction patterns.*

The confusion matrices in Figure 5 reveal interesting patterns:

- **SVM**: Shows strong diagonal elements, indicating correct predictions. Misclassifications tend to occur between similar types.
- **Logistic Regression**: Similar pattern to SVM but with slightly more confusion between types.
- **Naive Bayes**: Shows more scattered predictions with weaker diagonal, explaining its lower accuracy.

---

## 6. Model Comparison

### Why SVM Performed Best

1. SVM works well with high-dimensional sparse data (like TF-IDF vectors)
2. The linear kernel is effective for text classification
3. SVM maximizes the margin between classes, which helps with multi-class problems

### Why Naive Bayes Struggled

1. Naive Bayes assumes feature independence, which doesn't hold for text data
2. The model is too simple to capture complex patterns in personality expression
3. With 16 classes, the probability estimates become unreliable

### Logistic Regression Analysis

- Performed well because it's also a linear model like SVM
- Regularization helped prevent overfitting
- Good baseline model for text classification

---Per-Type Classification Accuracy

![Per-Type Accuracy](images/06_per_type_accuracy.png)

*Figure 6: Classification accuracy for each MBTI type, sorted from lowest to highest.*

Figure 6 shows that some personality types are much easier to classify than others:

- **High accuracy types**: Types with distinctive writing patterns (often intuitive types)
- **Low accuracy types**: Types that share similar linguistic features with others
- **Color coding**: Red indicates poor performance (<30%), orange is moderate (30-50%), green is good (>50%)

This visualization helps us understand which personality types the model struggles with and can guide future improvements.

### 7.4

## 7. Error Analysis

We analyzed the errors made by our best model (SVM):

### 7.1 Per-Type Accuracy

- Some personality types are easier to classify than others
- Types with more distinctive writing styles had higher accuracy
- Types that are similar to each other were often confused

### 7.2 Common Misclassifications

The most common confusion pairs usually differ by only one dimension:

- INFP ↔ INTP (differ in T/F)
![Dimension Accuracy](images/07_dimension_accuracy.png)

*Figure 7: Classification accuracy for each MBTI dimension (E/I, S/N, T/F, J/P).*

Figure 7 provides valuable insights into which aspects of personality are easier to detect from text:

- The model performs differently on each dimension
- Some dimensions (like E/I) show clearer patterns in writing style
- The T/F dimension is often hardest to distinguish, as both thinkers and feelers can express themselves similarly in text
- Understanding dimension-level accuracy suggests that a **binary classification approach** (classifying each dimension separately) might yield better results
- ENFP ↔ INFP (differ in E/I)

### 7.3 Dimension-Level Analysis

We also analyzed accuracy for each MBTI dimension:

| Dimension | Accuracy | Analysis |
| ------ | ---------- | ---------- |
| E/I | Higher | Extraverts and Introverts have different writing patterns |
| S/N | Medium | Sensing vs Intuition is moderately detectable |
| T/F | Lower | Thinking vs Feeling is harder to distinguish |
| J/P | Variable | Judging vs Perceiving shows in writing organization |

### 7.4 Why Some Errors Occur

1. **Data Quality**: Self-reported MBTI types may not be accurate
2. **Writing Context**: Forum posts may not reflect true personality
3. **Linguistic Overlap**: Similar personality types use similar language
4. **Sample Bias**: Dataset comes from one forum community

---

## 8. Conclusion

### Summary

In this project, we successfully built a machine learning pipeline to classify MBTI personality types from text. Our best model (SVM) achieved approximately 90% accuracy on the test set.

### What We Learned

1. Text preprocessing is crucial for NLP tasks
2. Class imbalance must be addressed for fair model training
3. Linear models (SVM, Logistic Regression) work well for text classification
4. Error analysis reveals important insights about model limitations

### Limitations

- The high accuracy might be inflated due to oversampling duplicates
- Real-world performance might be lower
- The model learned from a specific online community

### Future Improvements

1. Use cross-validation for more robust evaluation
2. Try deep learning models (LSTM, BERT) for comparison
3. Focus on binary classification for individual dimensions
4. Collect data from multiple sources
5. Use SMOTE instead of random oversampling

---

## References

1. Kaggle Dataset: <https://www.kaggle.com/datasets/datasnaek/mbti-type>
2. Scikit-learn Documentation: <https://scikit-learn.org/>
3. MBTI Official: <https://www.myersbriggs.org/>
