"""
Evaluate Random Forest Model

This script evaluates the trained Random Forest classifier using comprehensive
metrics and visualizations on the test set.

Author: Task 4.2 - Vancouver Business Heritage Proximity Predictor
Date: December 2025
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20
INPUT_FILE = 'data/processed/businesses_sampled_20pct.csv'
MODEL_FILE = 'models/random_forest.pkl'
BASELINE_EVAL_FILE = 'results/logistic_regression_evaluation.json'
EVALUATION_OUTPUT = 'results/random_forest_evaluation.json'
CONFUSION_MATRIX_OUTPUT = 'results/figures/random_forest_confusion_matrix.png'
ROC_CURVE_OUTPUT = 'results/figures/random_forest_roc_curve.png'

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def load_data():
    """Load the sampled dataset."""
    print("="*70)
    print("Evaluating Random Forest Model")
    print("="*70)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    print(f"\nLoading dataset from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} records")

    return df


def prepare_features(df):
    """Prepare feature matrix and target variable."""
    print("\nPreparing features...")

    # Target variable
    if 'near_heritage' not in df.columns:
        print("ERROR: 'near_heritage' column not found")
        sys.exit(1)

    y = df['near_heritage']

    # Feature columns (same as training)
    feature_columns = [
        'latitude',
        'longitude',
        'distance_to_nearest_heritage_m',
        'YearIssued',
        'distance_km',
        'log_distance'
    ]

    # Verify all feature columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing feature columns: {missing_cols}")
        sys.exit(1)

    X = df[feature_columns]

    # Handle missing values
    missing_counts = X.isnull().sum()
    if missing_counts.any():
        print("\nWARNING: Missing values detected - dropping rows...")
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y = y[mask]

    print(f"Feature matrix shape: {X.shape}")

    return X, y


def split_data(X, y):
    """Split data into train/test sets (must match training split)."""
    print(f"\nSplitting data (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Test set: {len(X_test):,} samples")

    # Show class distribution in test set
    print("\nClass distribution in test set:")
    test_dist = y_test.value_counts().sort_index()
    for label, count in test_dist.items():
        pct = (count / len(y_test)) * 100
        label_name = "Not near heritage" if label == 0 else "Near heritage"
        print(f"  {label_name} ({label}): {count:,} ({pct:.1f}%)")

    return X_test, y_test


def load_model():
    """Load the trained Random Forest model."""
    print(f"\nLoading model from: {MODEL_FILE}")

    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: Model file not found: {MODEL_FILE}")
        sys.exit(1)

    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    print(f"Model loaded: {type(model).__name__}")
    print(f"Number of estimators: {model.n_estimators}")

    return model


def make_predictions(model, X_test):
    """Make predictions on test set."""
    print("\nMaking predictions on test set...")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

    print(f"Predictions generated: {len(y_pred):,}")

    return y_pred, y_pred_proba


def calculate_metrics(y_test, y_pred, y_pred_proba):
    """Calculate all performance metrics."""
    print("\nCalculating performance metrics...")

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
    }

    print("\nTest Set Performance Metrics:")
    print(f"{'Metric':<15} {'Score':<10}")
    print("-" * 30)
    print(f"{'Accuracy':<15} {metrics['accuracy']:>8.4f}")
    print(f"{'Precision':<15} {metrics['precision']:>8.4f}")
    print(f"{'Recall':<15} {metrics['recall']:>8.4f}")
    print(f"{'F1-Score':<15} {metrics['f1_score']:>8.4f}")
    print(f"{'ROC-AUC':<15} {metrics['roc_auc']:>8.4f}")

    return metrics


def plot_confusion_matrix(y_test, y_pred, output_path=CONFUSION_MATRIX_OUTPUT):
    """Generate and save confusion matrix visualization."""
    print(f"\nGenerating confusion matrix...")

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'label': 'Count'},
                xticklabels=['Not Near (0)', 'Near Heritage (1)'],
                yticklabels=['Not Near (0)', 'Near Heritage (1)'],
                ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)

    # Add percentages as text
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = (count / total) * 100
            ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                   ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_path}")

    plt.close()

    # Print confusion matrix breakdown
    print("\nConfusion Matrix:")
    print(f"  True Negatives (TN):  {cm[0, 0]:,}")
    print(f"  False Positives (FP): {cm[0, 1]:,}")
    print(f"  False Negatives (FN): {cm[1, 0]:,}")
    print(f"  True Positives (TP):  {cm[1, 1]:,}")


def plot_roc_curve(y_test, y_pred_proba, roc_auc, output_path=ROC_CURVE_OUTPUT):
    """Generate and save ROC curve with AUC score."""
    print(f"\nGenerating ROC curve...")

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'Random Forest (AUC = {roc_auc:.4f})')

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier (AUC = 0.5000)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest - ROC Curve', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to: {output_path}")

    plt.close()


def compare_with_baseline(rf_metrics):
    """Compare Random Forest performance with Logistic Regression baseline."""
    print("\nComparing with baseline (Logistic Regression)...")

    if not os.path.exists(BASELINE_EVAL_FILE):
        print(f"WARNING: Baseline evaluation file not found: {BASELINE_EVAL_FILE}")
        print("Skipping comparison.")
        return None

    # Load baseline metrics
    with open(BASELINE_EVAL_FILE, 'r') as f:
        baseline_data = json.load(f)
        # Handle different JSON formats
        baseline_metrics = baseline_data.get('test_metrics') or baseline_data.get('metrics', {})

    if not baseline_metrics:
        print("WARNING: No baseline metrics found in file.")
        return None

    # Normalize metric names
    if 'auc_roc' in baseline_metrics and 'roc_auc' not in baseline_metrics:
        baseline_metrics['roc_auc'] = baseline_metrics['auc_roc']

    # Calculate improvements
    comparison = {}
    print(f"\n{'Metric':<15} {'Baseline':<12} {'Random Forest':<15} {'Improvement'}")
    print("-" * 60)

    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        if metric in baseline_metrics and metric in rf_metrics:
            baseline_val = baseline_metrics[metric]
            rf_val = rf_metrics[metric]
            improvement = rf_val - baseline_val
            improvement_pct = (improvement / baseline_val) * 100 if baseline_val > 0 else 0

            comparison[metric] = {
                'baseline': baseline_val,
                'random_forest': rf_val,
                'absolute_improvement': float(improvement),
                'percent_improvement': float(improvement_pct)
            }

            # Format improvement display
            if improvement >= 0:
                imp_str = f"+{improvement:.4f} ({improvement_pct:+.2f}%)"
            else:
                imp_str = f"{improvement:.4f} ({improvement_pct:.2f}%)"

            print(f"{metric.replace('_', ' ').title():<15} {baseline_val:>10.4f}  "
                  f"{rf_val:>13.4f}  {imp_str}")

    return comparison


def save_evaluation_results(metrics, comparison, output_path=EVALUATION_OUTPUT):
    """Save evaluation results to JSON."""
    print(f"\nSaving evaluation results to: {output_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = {
        'model': 'RandomForestClassifier',
        'test_metrics': metrics,
        'comparison_with_baseline': comparison,
        'visualizations': {
            'confusion_matrix': CONFUSION_MATRIX_OUTPUT,
            'roc_curve': ROC_CURVE_OUTPUT
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("Evaluation results saved")


def main():
    """Main execution function."""
    # Load data
    df = load_data()

    # Prepare features
    X, y = prepare_features(df)

    # Split data (must match training split)
    X_test, y_test = split_data(X, y)

    # Load trained model
    model = load_model()

    # Make predictions
    y_pred, y_pred_proba = make_predictions(model, X_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

    # Generate visualizations
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_proba, metrics['roc_auc'])

    # Compare with baseline
    comparison = compare_with_baseline(metrics)

    # Save results
    save_evaluation_results(metrics, comparison)

    print("\n" + "="*70)
    print("Random Forest evaluation complete!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  Evaluation metrics: {EVALUATION_OUTPUT}")
    print(f"  Confusion matrix: {CONFUSION_MATRIX_OUTPUT}")
    print(f"  ROC curve: {ROC_CURVE_OUTPUT}")


if __name__ == "__main__":
    main()
