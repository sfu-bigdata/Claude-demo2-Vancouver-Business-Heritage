"""
Evaluate Logistic Regression Model Performance

This script evaluates the trained Logistic Regression model using comprehensive
metrics and generates visualizations including confusion matrix and ROC curve.

Author: Claude Code
Date: 2025-12-08
"""

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_model(model_path):
    """Load trained model from disk."""
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
    return model

def load_data(data_path):
    """Load the sampled dataset."""
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    return df

def prepare_features(df):
    """
    Prepare features for evaluation using the same preprocessing as training.
    """
    # Define features to exclude (same as training)
    exclude_cols = [
        'LicenceRSN',
        'BusinessName',
        'BusinessTradeName',
        'near_heritage',
        'distance_to_nearest_heritage_m',
        'nearest_heritage_id'
    ]

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Separate features and target
    X = df[feature_cols].copy()
    y = df['near_heritage'].copy()

    return X, y

def plot_confusion_matrix(y_true, y_pred, output_path, title='Confusion Matrix'):
    """
    Plot and save confusion matrix with both counts and percentages.
    """
    print(f"\nGenerating confusion matrix...")

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Not Near (0)', 'Near Heritage (1)'],
                yticklabels=['Not Near (0)', 'Near Heritage (1)'])
    ax1.set_title(f'{title} - Counts', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)

    # Plot 2: Percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                xticklabels=['Not Near (0)', 'Near Heritage (1)'],
                yticklabels=['Not Near (0)', 'Near Heritage (1)'])
    ax2.set_title(f'{title} - Percentages (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_path}")
    plt.close()

    return cm

def plot_roc_curve(y_true, y_proba, output_path, title='ROC Curve'):
    """
    Plot and save ROC curve with AUC score.
    """
    print(f"\nGenerating ROC curve...")

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    # Create plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {output_path}")
    plt.close()

    return auc_score

def calculate_metrics(y_true, y_pred, y_proba):
    """
    Calculate all evaluation metrics.
    """
    print("\nCalculating performance metrics...")

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred)),
        'auc_roc': float(roc_auc_score(y_true, y_proba))
    }

    # Print metrics
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print("="*60)

    return metrics

def save_evaluation_results(metrics, cm, output_path):
    """
    Save evaluation results to JSON file.
    """
    print(f"\nSaving evaluation results to {output_path}...")

    results = {
        'model_type': 'LogisticRegression',
        'metrics': metrics,
        'confusion_matrix': {
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        },
        'classification_report': {
            'note': 'Detailed metrics per class',
            'total_samples': int(cm.sum())
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("Evaluation results saved successfully!")

def main():
    """Main evaluation pipeline."""
    print("="*80)
    print("Logistic Regression Model Evaluation")
    print("="*80)

    # Define paths
    model_path = Path('models/logistic_regression.pkl')
    data_path = Path('data/processed/businesses_sampled_20pct.csv')
    cm_output_path = Path('results/figures/logistic_regression_confusion_matrix.png')
    roc_output_path = Path('results/figures/logistic_regression_roc_curve.png')
    results_output_path = Path('results/logistic_regression_evaluation.json')

    # Load model
    model = load_model(model_path)

    # Load data
    df = load_data(data_path)

    # Prepare features
    X, y = prepare_features(df)

    # Split data (same as training)
    print("\nSplitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Test target distribution:\n{y_test.value_counts()}")

    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba)

    # Generate confusion matrix
    cm = plot_confusion_matrix(
        y_test,
        y_pred,
        cm_output_path,
        title='Logistic Regression - Confusion Matrix'
    )

    # Generate ROC curve
    auc_score = plot_roc_curve(
        y_test,
        y_proba,
        roc_output_path,
        title='Logistic Regression - ROC Curve'
    )

    # Save evaluation results
    save_evaluation_results(metrics, cm, results_output_path)

    # Print classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred,
                                target_names=['Not Near Heritage', 'Near Heritage']))

    print("\n" + "="*80)
    print("Evaluation completed successfully!")
    print("="*80)
    print(f"\nOutputs saved to:")
    print(f"  - Confusion Matrix: {cm_output_path}")
    print(f"  - ROC Curve: {roc_output_path}")
    print(f"  - Metrics JSON: {results_output_path}")

if __name__ == '__main__':
    main()
