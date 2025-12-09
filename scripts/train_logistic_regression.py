"""
Train Logistic Regression Baseline Model for Heritage Proximity Prediction

This script trains a Logistic Regression classifier to predict whether a business
is located within 1km of a heritage site. It serves as the baseline model for
comparison with more complex models.

Author: Claude Code
Date: 2025-12-08
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_data(data_path):
    """Load the sampled dataset."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['near_heritage'].value_counts()}")
    return df

def prepare_features(df):
    """
    Prepare features for training by selecting relevant columns and handling categorical variables.

    Features to exclude:
    - LicenceRSN: identifier
    - BusinessName, BusinessTradeName: identifiers
    - near_heritage: target variable
    - distance_to_nearest_heritage_m: raw distance (leakage)
    - nearest_heritage_id: identifier
    """
    # Define features to exclude
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

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"\nFeature columns: {len(feature_cols)}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

    return X, y, categorical_cols, numerical_cols

def create_preprocessing_pipeline(categorical_cols, numerical_cols):
    """Create preprocessing pipeline for categorical and numerical features."""

    # Create transformers
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    numerical_transformer = StandardScaler()

    # Combine into column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder='drop'
    )

    return preprocessor

def train_model(X_train, y_train, categorical_cols, numerical_cols):
    """
    Train Logistic Regression model with preprocessing pipeline.

    Uses class_weight='balanced' to handle class imbalance.
    """
    print("\nTraining Logistic Regression model...")

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)

    # Create full pipeline with Logistic Regression
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            max_iter=1000
        ))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    print("Model training completed!")
    return model_pipeline

def perform_cross_validation(X, y, categorical_cols, numerical_cols, n_folds=5):
    """
    Perform stratified k-fold cross-validation.

    Args:
        X: Feature matrix
        y: Target vector
        categorical_cols: List of categorical feature names
        numerical_cols: List of numerical feature names
        n_folds: Number of folds for cross-validation

    Returns:
        Dictionary with cross-validation scores
    """
    print(f"\nPerforming {n_folds}-fold stratified cross-validation...")

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)

    # Create full pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            max_iter=1000
        ))
    ])

    # Define stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }

    # Perform cross-validation
    cv_results = cross_validate(
        model_pipeline,
        X,
        y,
        cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    # Calculate mean and std for each metric
    results = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        train_key = f'train_{metric}'
        test_key = f'test_{metric}'

        results[f'{metric}_train_mean'] = float(cv_results[train_key].mean())
        results[f'{metric}_train_std'] = float(cv_results[train_key].std())
        results[f'{metric}_test_mean'] = float(cv_results[test_key].mean())
        results[f'{metric}_test_std'] = float(cv_results[test_key].std())

        print(f"{metric.capitalize():10s} - Train: {results[f'{metric}_train_mean']:.4f} (+/- {results[f'{metric}_train_std']:.4f}), "
              f"Test: {results[f'{metric}_test_mean']:.4f} (+/- {results[f'{metric}_test_std']:.4f})")

    # Add fold scores for detailed analysis
    results['fold_scores'] = {
        metric: cv_results[f'test_{metric}'].tolist()
        for metric in ['accuracy', 'precision', 'recall', 'f1']
    }

    return results

def save_model(model, output_path):
    """Save trained model to disk."""
    print(f"\nSaving model to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")

def save_cv_results(results, output_path):
    """Save cross-validation results to JSON."""
    print(f"Saving cross-validation results to {output_path}...")

    # Add metadata
    results['model_type'] = 'LogisticRegression'
    results['random_state'] = RANDOM_STATE
    results['class_weight'] = 'balanced'
    results['n_folds'] = 5

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved successfully!")

def main():
    """Main training pipeline."""
    print("="*80)
    print("Logistic Regression Baseline Model Training")
    print("="*80)

    # Define paths
    data_path = Path('data/processed/businesses_sampled_20pct.csv')
    model_output_path = Path('models/logistic_regression.pkl')
    results_output_path = Path('results/logistic_regression_cv.json')

    # Load data
    df = load_data(data_path)

    # Prepare features
    X, y, categorical_cols, numerical_cols = prepare_features(df)

    # Split data into train and test sets (80/20)
    print("\nSplitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Train target distribution:\n{y_train.value_counts()}")
    print(f"Test target distribution:\n{y_test.value_counts()}")

    # Perform cross-validation on training data
    cv_results = perform_cross_validation(X_train, y_train, categorical_cols, numerical_cols)

    # Train final model on full training set
    final_model = train_model(X_train, y_train, categorical_cols, numerical_cols)

    # Evaluate on test set
    print("\nEvaluating on held-out test set...")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = final_model.predict(X_test)
    test_results = {
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'test_precision': float(precision_score(y_test, y_pred)),
        'test_recall': float(recall_score(y_test, y_pred)),
        'test_f1': float(f1_score(y_test, y_pred))
    }

    print(f"Test Accuracy:  {test_results['test_accuracy']:.4f}")
    print(f"Test Precision: {test_results['test_precision']:.4f}")
    print(f"Test Recall:    {test_results['test_recall']:.4f}")
    print(f"Test F1:        {test_results['test_f1']:.4f}")

    # Add test results to cv_results
    cv_results.update(test_results)

    # Save model and results
    save_model(final_model, model_output_path)
    save_cv_results(cv_results, results_output_path)

    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)

if __name__ == '__main__':
    main()
