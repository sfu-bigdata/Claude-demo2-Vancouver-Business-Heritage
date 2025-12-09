"""
Create Sampled Dataset for Model Development

This script creates a 20% stratified sample of the full processed dataset
to enable faster iteration during model development while preserving class distribution.

Author: Task 3.1 - Vancouver Business Heritage Proximity Predictor
Date: December 2025
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

# Configuration
RANDOM_STATE = 42
SAMPLE_FRACTION = 0.20
INPUT_FILE = 'data/processed/businesses_with_heritage_labels.csv'
OUTPUT_FILE = 'data/processed/businesses_sampled_20pct.csv'


def load_full_dataset():
    """Load the full processed dataset."""
    print("="*60)
    print("Creating 20% Sampled Dataset")
    print("="*60)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    print(f"\nLoading full dataset from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} records")

    return df


def verify_target_column(df):
    """Verify the target column exists and check its distribution."""
    if 'near_heritage' not in df.columns:
        print("ERROR: 'near_heritage' column not found in dataset")
        sys.exit(1)

    print("\nOriginal dataset class distribution:")
    class_counts = df['near_heritage'].value_counts().sort_index()
    total = len(df)

    for label, count in class_counts.items():
        percentage = (count / total) * 100
        label_name = "Not near heritage" if label == 0 else "Near heritage"
        print(f"  {label_name} ({label}): {count:,} ({percentage:.1f}%)")

    return df


def create_stratified_sample(df, sample_fraction=SAMPLE_FRACTION, random_state=RANDOM_STATE):
    """
    Create a stratified sample maintaining class distribution.

    Args:
        df: Full dataset
        sample_fraction: Fraction of data to sample (0.20 = 20%)
        random_state: Random seed for reproducibility

    Returns:
        Sampled dataframe
    """
    print(f"\nCreating stratified sample ({sample_fraction*100:.0f}%)...")
    print(f"Random state: {random_state}")

    # Use train_test_split with stratification
    # We want the "test" split as our sample
    _, sampled_df = train_test_split(
        df,
        test_size=sample_fraction,
        stratify=df['near_heritage'],
        random_state=random_state
    )

    print(f"Sampled {len(sampled_df):,} records")

    return sampled_df


def verify_sample_distribution(original_df, sampled_df):
    """Verify that the sample preserves the class distribution."""
    print("\nVerifying class distribution preservation...")

    original_dist = original_df['near_heritage'].value_counts(normalize=True).sort_index()
    sampled_dist = sampled_df['near_heritage'].value_counts(normalize=True).sort_index()

    print("\nComparison:")
    print(f"{'Class':<20} {'Original %':<15} {'Sampled %':<15} {'Difference':<15}")
    print("-" * 65)

    for label in original_dist.index:
        orig_pct = original_dist[label] * 100
        samp_pct = sampled_dist[label] * 100
        diff = abs(orig_pct - samp_pct)
        label_name = "Not near (0)" if label == 0 else "Near (1)"
        print(f"{label_name:<20} {orig_pct:>12.2f}% {samp_pct:>14.2f}% {diff:>13.2f}%")

    # Check if distribution is preserved (within 0.5%)
    max_diff = max(abs(original_dist[label] - sampled_dist[label]) * 100
                   for label in original_dist.index)

    if max_diff < 0.5:
        print("\n✓ Class distribution preserved (max difference < 0.5%)")
        return True
    else:
        print(f"\n⚠ Warning: Class distribution difference is {max_diff:.2f}%")
        return False


def save_sampled_dataset(sampled_df, output_file=OUTPUT_FILE):
    """Save the sampled dataset to CSV."""
    print(f"\nSaving sampled dataset to: {output_file}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    sampled_df.to_csv(output_file, index=False)

    # Get file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Saved {len(sampled_df):,} records ({file_size_mb:.2f} MB)")


def main():
    """Main execution function."""
    # Load full dataset
    df = load_full_dataset()

    # Verify target column and show distribution
    df = verify_target_column(df)

    # Create stratified sample
    sampled_df = create_stratified_sample(df, SAMPLE_FRACTION, RANDOM_STATE)

    # Verify distribution is preserved
    verify_sample_distribution(df, sampled_df)

    # Save sampled dataset
    save_sampled_dataset(sampled_df, OUTPUT_FILE)

    print("\n" + "="*60)
    print("Sampling complete!")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Sample fraction: {SAMPLE_FRACTION*100:.0f}%")
    print(f"  Random state: {RANDOM_STATE}")
    print(f"  Input: {INPUT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"\nRecords: {len(df):,} → {len(sampled_df):,}")


if __name__ == "__main__":
    main()
