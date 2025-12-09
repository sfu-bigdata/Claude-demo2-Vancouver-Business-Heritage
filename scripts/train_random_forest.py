"""
Train Random Forest Classifier

This script trains a Random Forest classifier to predict whether a Vancouver
business is located near a heritage site (within 1km). Random Forest can capture
non-linear patterns and feature interactions.

Author: Task 3.3 - Vancouver Business Heritage Proximity Predictor
Date: December 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
import pickle
import json
import os
import sys

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 100
N_FOLDS = 5
INPUT_FILE = 'data/processed/businesses_sampled_20pct.csv'
MODEL_OUTPUT = 'models/random_forest.pkl'
CV_RESULTS_OUTPUT = 'results/random_forest_cv.json'
FEATURE_IMPORTANCES_OUTPUT = 'results/random_forest_importances.json'


def load_data():
    """Load the sampled dataset."""
    print("="*70)
    print("Training Random Forest Classifier")
    print("="*70)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    print(f"\nLoading dataset from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} records")

    return df


def prepare_features(df):
    """
    Prepare feature matrix and target variable.

    Uses the same features as the baseline model for fair comparison.
    """
    print("\nPreparing features...")

    # Target variable
    if 'near_heritage' not in df.columns:
        print("ERROR: 'near_heritage' column not found")
        sys.exit(1)

    y = df['near_heritage']

    # Feature columns (numerical features available in the dataset)
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

    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {', '.join(feature_columns)}")

    # Check for missing values
    missing_counts = X.isnull().sum()
    if missing_counts.any():
        print("\nWARNING: Missing values detected:")
        print(missing_counts[missing_counts > 0])
        print("Dropping rows with missing values...")
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y = y[mask]
        print(f"Remaining records: {len(X):,}")

    return X, y, feature_columns


def split_data(X, y):
    """Split data into train/test sets with stratification."""
    print(f"\nSplitting data (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")

    # Show class distribution
    print("\nClass distribution in training set:")
    train_dist = y_train.value_counts().sort_index()
    for label, count in train_dist.items():
        pct = (count / len(y_train)) * 100
        label_name = "Not near heritage" if label == 0 else "Near heritage"
        print(f"  {label_name} ({label}): {count:,} ({pct:.1f}%)")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train Random Forest classifier with balanced class weights.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained model
    """
    print(f"\nTraining Random Forest...")
    print(f"Configuration:")
    print(f"  n_estimators: {N_ESTIMATORS}")
    print(f"  class_weight: balanced")
    print(f"  random_state: {RANDOM_STATE}")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1  # Use all available cores
    )

    model.fit(X_train, y_train)

    print("Model training complete!")

    return model


def cross_validate_model(model, X_train, y_train):
    """
    Perform stratified k-fold cross-validation.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels

    Returns:
        Dictionary of cross-validation scores
    """
    print(f"\nPerforming {N_FOLDS}-fold stratified cross-validation...")

    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }

    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Perform cross-validation
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    # Calculate mean and std for each metric
    results = {}
    print("\nCross-validation results:")
    print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Range'}")
    print("-" * 60)

    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        scores = cv_results[f'test_{metric}']
        mean_score = scores.mean()
        std_score = scores.std()
        min_score = scores.min()
        max_score = scores.max()

        results[metric] = {
            'mean': float(mean_score),
            'std': float(std_score),
            'min': float(min_score),
            'max': float(max_score),
            'scores': scores.tolist()
        }

        print(f"{metric.capitalize():<15} {mean_score:>10.4f}  {std_score:>10.4f}  "
              f"[{min_score:.4f}, {max_score:.4f}]")

    return results


def extract_feature_importances(model, feature_names):
    """
    Extract and display feature importances from the trained model.

    Args:
        model: Trained Random Forest model
        feature_names: List of feature names

    Returns:
        Dictionary mapping feature names to importance scores
    """
    print("\nFeature importances:")
    print(f"{'Feature':<40} {'Importance':<12} {'Visual'}")
    print("-" * 70)

    importances = model.feature_importances_

    # Sort by importance (descending)
    indices = np.argsort(importances)[::-1]

    feature_importance_dict = {}

    for i in indices:
        feature = feature_names[i]
        importance = importances[i]
        feature_importance_dict[feature] = float(importance)

        # Create visual bar
        bar_length = int(importance * 50)  # Scale to 50 chars max
        bar = '█' * bar_length

        print(f"{feature:<40} {importance:>10.4f}  {bar}")

    return feature_importance_dict


def save_model(model, output_path=MODEL_OUTPUT):
    """Save the trained model to disk."""
    print(f"\nSaving model to: {output_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    # Get file size
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"Model saved ({file_size_kb:.1f} KB)")


def save_cv_results(cv_results, output_path=CV_RESULTS_OUTPUT):
    """Save cross-validation results to JSON."""
    print(f"\nSaving cross-validation results to: {output_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Add metadata
    results_with_metadata = {
        'model': 'RandomForestClassifier',
        'configuration': {
            'n_estimators': N_ESTIMATORS,
            'class_weight': 'balanced',
            'random_state': RANDOM_STATE,
            'n_folds': N_FOLDS
        },
        'cv_results': cv_results
    }

    with open(output_path, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)

    print("Cross-validation results saved")


def save_feature_importances(importances, output_path=FEATURE_IMPORTANCES_OUTPUT):
    """Save feature importances to JSON."""
    print(f"\nSaving feature importances to: {output_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Add metadata
    importances_with_metadata = {
        'model': 'RandomForestClassifier',
        'n_estimators': N_ESTIMATORS,
        'feature_importances': importances
    }

    with open(output_path, 'w') as f:
        json.dump(importances_with_metadata, f, indent=2)

    print("Feature importances saved")


def main():
    """Main execution function."""
    # Load data
    df = load_data()

    # Prepare features
    X, y, feature_names = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Cross-validate
    cv_results = cross_validate_model(model, X_train, y_train)

    # Extract feature importances
    feature_importances = extract_feature_importances(model, feature_names)

    # Save outputs
    save_model(model)
    save_cv_results(cv_results)
    save_feature_importances(feature_importances)

    print("\n" + "="*70)
    print("Random Forest training complete!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  Model: {MODEL_OUTPUT}")
    print(f"  CV Results: {CV_RESULTS_OUTPUT}")
    print(f"  Feature Importances: {FEATURE_IMPORTANCES_OUTPUT}")
    print(f"\nConfiguration:")
    print(f"  Random state: {RANDOM_STATE}")
    print(f"  Train/test split: {1-TEST_SIZE:.0%}/{TEST_SIZE:.0%}")
    print(f"  Cross-validation folds: {N_FOLDS}")
    print(f"  Number of estimators: {N_ESTIMATORS}")


if __name__ == "__main__":
    main()
