"""
MBTI Personality Type Text Classifier
======================================
A Machine Learning project to predict MBTI personality types based on social media posts.
Uses ONLY classical ML algorithms (no Deep Learning).

Author: Senior ML Engineer
Date: February 2026
Course: University Machine Learning Project
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing & Feature Extraction
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Classical ML Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Handling Imbalanced Data
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================================
# 1. DATA LOADING & INSPECTION
# ============================================================================

def load_and_inspect_data(filepath='mbti_1.csv'):
    """
    Load the MBTI dataset and display basic statistics.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("="*80)
    print("STEP 1: DATA LOADING & INSPECTION")
    print("="*80)
    
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Display basic info
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"   - Total samples: {df.shape[0]}")
    print(f"   - Features: {df.shape[1]}")
    
    # Check for missing values
    print(f"\n🔍 Missing Values:\n{df.isnull().sum()}")
    
    # Display first few rows
    print(f"\n📝 First 2 rows:")
    print(df.head(2))
    
    # Class distribution (showing imbalance)
    print(f"\n📈 Class Distribution (showing imbalance):")
    print("-" * 50)
    class_counts = df['type'].value_counts().sort_values(ascending=False)
    print(class_counts)
    
    # Visualize class distribution
    plt.figure(figsize=(12, 6))
    class_counts.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title('MBTI Type Distribution (Imbalanced Dataset)', fontsize=14, fontweight='bold')
    plt.xlabel('MBTI Type', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150)
    print("\n✅ Saved class distribution plot to 'class_distribution.png'")
    plt.close()
    
    # Calculate imbalance ratio
    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class
    print(f"\n⚠️  Imbalance Ratio: {imbalance_ratio:.2f}x")
    print(f"   (Most frequent: {max_class}, Least frequent: {min_class})")
    
    return df


# ============================================================================
# 2. TEXT PREPROCESSING & CLEANING
# ============================================================================

def clean_text(text):
    """
    Clean and preprocess text data.
    
    Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove special characters (keep only letters and spaces)
        4. Remove extra whitespace
        
    Args:
        text (str): Raw text string
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and digits (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_data(df):
    """
    Preprocess the MBTI dataset.
    
    The 'posts' column contains 50 posts separated by '|||'.
    We aggregate them into one text and clean it.
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Preprocessed dataframe with 'cleaned_posts' column
    """
    print("\n" + "="*80)
    print("STEP 2: TEXT PREPROCESSING & CLEANING")
    print("="*80)
    
    # Aggregate posts (replace ||| with space)
    print("\n🔧 Aggregating posts (50 posts per user)...")
    df['aggregated_posts'] = df['posts'].apply(lambda x: x.replace('|||', ' '))
    
    # Clean text
    print("🧹 Cleaning text (removing URLs, special chars, etc.)...")
    df['cleaned_posts'] = df['aggregated_posts'].apply(clean_text)
    
    # Display sample
    print("\n📄 Sample of cleaned text:")
    print("-" * 50)
    print(f"Original (first 200 chars):\n{df['posts'].iloc[0][:200]}...\n")
    print(f"Cleaned (first 200 chars):\n{df['cleaned_posts'].iloc[0][:200]}...")
    
    # Check average text length
    avg_length = df['cleaned_posts'].apply(lambda x: len(x.split())).mean()
    print(f"\n📏 Average text length: {avg_length:.0f} words per user")
    
    return df


# ============================================================================
# 3. HANDLING CLASS IMBALANCE
# ============================================================================

def balance_dataset(X, y, method='oversample'):
    """
    Handle class imbalance using oversampling or undersampling.
    
    Args:
        X (array-like): Feature matrix
        y (array-like): Labels
        method (str): 'oversample' or 'undersample'
        
    Returns:
        tuple: Balanced X and y
    """
    print("\n" + "="*80)
    print("STEP 3: HANDLING CLASS IMBALANCE")
    print("="*80)
    
    print(f"\n⚖️  Before balancing:")
    print(f"   - Total samples: {len(y)}")
    print(f"   - Class distribution:\n{pd.Series(y).value_counts().sort_index()}")
    
    if method == 'oversample':
        print(f"\n🔼 Applying Random Oversampling...")
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
    elif method == 'undersample':
        print(f"\n🔽 Applying Random Undersampling...")
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    else:
        print(f"\n⚠️  Unknown method '{method}'. Skipping balancing.")
        return X, y
    
    X_balanced, y_balanced = sampler.fit_resample(X.reshape(-1, 1) if X.ndim == 1 else X, y)
    
    print(f"\n✅ After balancing:")
    print(f"   - Total samples: {len(y_balanced)}")
    print(f"   - Class distribution:\n{pd.Series(y_balanced).value_counts().sort_index()}")
    
    return X_balanced, y_balanced


# ============================================================================
# 4. FEATURE EXTRACTION
# ============================================================================

def extract_features(X_train_text, X_test_text, max_features=5000):
    """
    Convert text to numerical features using TF-IDF.
    
    TF-IDF (Term Frequency - Inverse Document Frequency):
        - Measures importance of words in documents
        - Reduces weight of common words
        - Increases weight of rare, informative words
    
    Args:
        X_train_text (pd.Series): Training text data
        X_test_text (pd.Series): Test text data
        max_features (int): Maximum number of features to extract
        
    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, vectorizer)
    """
    print("\n" + "="*80)
    print("STEP 4: FEATURE EXTRACTION (TF-IDF)")
    print("="*80)
    
    print(f"\n🔢 Vectorizing text using TF-IDF...")
    print(f"   - Max features: {max_features}")
    print(f"   - N-gram range: (1, 2) [unigrams + bigrams]")
    print(f"   - Stop words: English")
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrams and bigrams
        stop_words='english',  # Remove common English words
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.8  # Ignore terms that appear in more than 80% of documents
    )
    
    # Fit on training data and transform both train and test
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"   - Training set shape: {X_train_tfidf.shape}")
    print(f"   - Test set shape: {X_test_tfidf.shape}")
    print(f"   - Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Show top features
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n📌 Sample features: {list(feature_names[:10])}")
    
    return X_train_tfidf, X_test_tfidf, vectorizer


# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

def train_models(X_train, y_train):
    """
    Train multiple classical ML models.
    
    Models:
        1. Naive Bayes (MultinomialNB) - Fast, works well with text
        2. Logistic Regression (SGDClassifier) - Linear model, efficient
        3. Support Vector Machine (LinearSVC) - Powerful, good for text
        4. Decision Tree - Non-linear, interpretable
        
    Args:
        X_train (sparse matrix): Training features
        y_train (array-like): Training labels
        
    Returns:
        dict: Dictionary of trained models
    """
    print("\n" + "="*80)
    print("STEP 5: MODEL TRAINING")
    print("="*80)
    
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression (SGD)': SGDClassifier(
            loss='log_loss',  # For logistic regression
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Support Vector Machine': LinearSVC(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=50,  # Limit depth to prevent overfitting
            random_state=RANDOM_STATE
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"   ✅ {name} trained successfully!")
    
    return trained_models


# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================

def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models and print performance metrics.
    
    Metrics:
        - Accuracy: Overall correctness
        - Precision: How many predicted positives are actually positive
        - Recall: How many actual positives are correctly identified
        - F1-Score: Harmonic mean of precision and recall
        
    Args:
        models (dict): Dictionary of trained models
        X_test (sparse matrix): Test features
        y_test (array-like): Test labels
        
    Returns:
        dict: Dictionary of predictions for each model
    """
    print("\n" + "="*80)
    print("STEP 6: MODEL EVALUATION")
    print("="*80)
    
    predictions = {}
    results = []
    
    for name, model in models.items():
        print(f"\n{'='*80}")
        print(f"📊 {name} - Performance Report")
        print(f"{'='*80}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print(f"\n📈 Classification Report:")
        print("-" * 80)
        report = classification_report(y_test, y_pred, zero_division=0)
        print(report)
        
        # Store results for comparison
        results.append({
            'Model': name,
            'Accuracy': accuracy
        })
    
    # Summary comparison
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*80)
    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    print(results_df.to_string(index=False))
    
    best_model = results_df.iloc[0]['Model']
    best_accuracy = results_df.iloc[0]['Accuracy']
    print(f"\n🏆 Best Model: {best_model} (Accuracy: {best_accuracy:.4f})")
    
    return predictions, best_model


# ============================================================================
# 7. CONFUSION MATRIX VISUALIZATION
# ============================================================================

def plot_confusion_matrix(y_test, y_pred, model_name, classes):
    """
    Plot and save confusion matrix for the best model.
    
    Confusion Matrix shows:
        - Rows: Actual classes
        - Columns: Predicted classes
        - Diagonal: Correct predictions
        - Off-diagonal: Misclassifications
        
    Args:
        y_test (array-like): True labels
        y_pred (array-like): Predicted labels
        model_name (str): Name of the model
        classes (list): List of class labels
    """
    print("\n" + "="*80)
    print("STEP 7: CONFUSION MATRIX VISUALIZATION")
    print("="*80)
    
    print(f"\n📊 Generating confusion matrix for {model_name}...")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Number of Predictions'}
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    filename = f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
    plt.savefig(filename, dpi=150)
    print(f"✅ Saved confusion matrix to '{filename}'")
    plt.close()


# ============================================================================
# 8. ERROR ANALYSIS
# ============================================================================

def error_analysis(X_test_text, y_test, y_pred, model_name, num_examples=5):
    """
    Analyze misclassified examples to understand model failures.
    
    This helps identify:
        - Short/ambiguous texts
        - Sarcastic or ironic posts
        - Edge cases between similar types
        
    Args:
        X_test_text (pd.Series): Original test text
        y_test (array-like): True labels
        y_pred (array-like): Predicted labels
        model_name (str): Name of the model
        num_examples (int): Number of examples to show
    """
    print("\n" + "="*80)
    print("STEP 8: ERROR ANALYSIS")
    print("="*80)
    
    print(f"\n🔍 Analyzing misclassifications for {model_name}...")
    
    # Find misclassified indices
    misclassified_mask = y_test != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    total_errors = len(misclassified_indices)
    print(f"\n❌ Total misclassifications: {total_errors}/{len(y_test)} ({total_errors/len(y_test)*100:.2f}%)")
    
    if total_errors == 0:
        print("✨ Perfect predictions! No errors to analyze.")
        return
    
    # Randomly sample some misclassified examples
    num_examples = min(num_examples, total_errors)
    sample_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
    
    print(f"\n📋 Showing {num_examples} misclassified examples:\n")
    
    for i, idx in enumerate(sample_indices, 1):
        text = X_test_text.iloc[idx]
        true_label = y_test.iloc[idx]
        pred_label = y_pred[idx]
        
        print(f"{'='*80}")
        print(f"❌ Example {i}:")
        print(f"{'='*80}")
        print(f"True Label:      {true_label}")
        print(f"Predicted Label: {pred_label}")
        print(f"\nText (first 300 chars):")
        print(f"{text[:300]}...")
        print(f"\n💭 Possible reasons for misclassification:")
        
        # Analyze text length
        word_count = len(text.split())
        if word_count < 50:
            print(f"   - Text is very short ({word_count} words) - insufficient information")
        
        # Check for similar types (e.g., INFP vs INTP)
        if sum(c1 == c2 for c1, c2 in zip(true_label, pred_label)) >= 3:
            print(f"   - Types are very similar (differ in only 1 dimension)")
        
        print()
    
    print("\n" + "="*80)
    print("💡 INSIGHTS FROM ERROR ANALYSIS:")
    print("="*80)
    print("""
    Common reasons for misclassification in MBTI prediction:
    
    1. **Ambiguous Posts**: Short or vague texts lack distinctive features
    2. **Similar Types**: Types differing in 1 dimension (e.g., INFP vs INFJ) are hard to distinguish
    3. **Sarcasm/Irony**: Models struggle with non-literal language
    4. **Multiple Behaviors**: People exhibit traits from multiple types
    5. **Context Loss**: Removing URLs/emojis may lose important context
    
    💡 Potential Improvements:
    - Use larger datasets for underrepresented types
    - Engineer features like text length, punctuation patterns
    - Use ensemble methods to combine model strengths
    - Consider multi-label classification for borderline cases
    """)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main pipeline that orchestrates the entire workflow.
    """
    print("\n" + "="*80)
    print("🧠 MBTI PERSONALITY TYPE CLASSIFIER")
    print("="*80)
    print("Using Classical Machine Learning (No Deep Learning)")
    print("="*80)
    
    # Step 1: Load and inspect data
    df = load_and_inspect_data('mbti_1.csv')
    
    # Step 2: Preprocess data
    df = preprocess_data(df)
    
    # Step 3: Split data BEFORE balancing (important!)
    print("\n" + "="*80)
    print("STEP 3: TRAIN-TEST SPLIT")
    print("="*80)
    print("\n✂️  Splitting data into Train (80%) and Test (20%)...")
    
    X = df['cleaned_posts']
    y = df['type']
    
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class distribution in both sets
    )
    
    print(f"   - Training set: {len(X_train_text)} samples")
    print(f"   - Test set: {len(X_test_text)} samples")
    
    # Step 4: Extract features (TF-IDF)
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(
        X_train_text, X_test_text, max_features=5000
    )
    
    # Step 5: Handle imbalance on TRAINING DATA ONLY
    # Note: We balance AFTER feature extraction to preserve the sparse matrix structure
    print("\n" + "="*80)
    print("STEP 4: HANDLING CLASS IMBALANCE (Training Set Only)")
    print("="*80)
    
    print(f"\n⚠️  Important: Balancing is applied ONLY to training data to prevent data leakage!")
    
    # Convert sparse matrix to dense for balancing
    print("\n🔄 Converting sparse matrix to dense for balancing...")
    X_train_dense = X_train_tfidf.toarray()
    
    # Apply oversampling
    X_train_balanced, y_train_balanced = balance_dataset(
        X_train_dense, 
        y_train, 
        method='oversample'  # You can change to 'undersample' if preferred
    )
    
    # Step 6: Train models
    trained_models = train_models(X_train_balanced, y_train_balanced)
    
    # Step 7: Evaluate models
    predictions, best_model_name = evaluate_models(trained_models, X_test_tfidf, y_test)
    
    # Step 8: Plot confusion matrix for best model
    plot_confusion_matrix(
        y_test, 
        predictions[best_model_name], 
        best_model_name,
        sorted(y.unique())
    )
    
    # Step 9: Error analysis
    error_analysis(
        X_test_text, 
        y_test, 
        predictions[best_model_name], 
        best_model_name,
        num_examples=5
    )
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print("\n📁 Generated files:")
    print("   - class_distribution.png")
    print("   - confusion_matrix_*.png")
    print("\n🎓 Educational Notes:")
    print("""
    This implementation demonstrates:
    ✓ Handling imbalanced datasets with oversampling
    ✓ Proper train-test split to prevent data leakage
    ✓ Text preprocessing and cleaning techniques
    ✓ TF-IDF feature extraction for text classification
    ✓ Training and comparing multiple classical ML models
    ✓ Comprehensive evaluation with multiple metrics
    ✓ Error analysis to understand model limitations
    
    Key Takeaways:
    • ALWAYS balance AFTER splitting (never before!)
    • Text cleaning significantly impacts performance
    • No single model is perfect - try multiple approaches
    • Error analysis helps identify improvement opportunities
    • Class imbalance is a real-world challenge in ML
    """)
    
    print("\n🚀 Next Steps for Improvement:")
    print("""
    1. Feature Engineering: Add text statistics (length, punctuation)
    2. Hyperparameter Tuning: Use GridSearchCV for optimal parameters
    3. Ensemble Methods: Combine models with VotingClassifier
    4. Advanced Preprocessing: Lemmatization, POS tagging
    5. Cross-Validation: Use K-Fold CV for robust evaluation
    """)


if __name__ == "__main__":
    main()
