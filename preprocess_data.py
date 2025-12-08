"""
Data Preprocessing Script for Vancouver Business Heritage Proximity Predictor

This script:
1. Loads raw business and heritage site data
2. Cleans and validates coordinates
3. Calculates distances between businesses and nearest heritage sites
4. Creates binary labels (near_heritage: 1 if within 1km, 0 otherwise)
5. Engineers features for ML modeling
6. Saves processed data
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Distance threshold in meters
DISTANCE_THRESHOLD_M = 1000  # 1 kilometer


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw business and heritage site data."""
    print("Loading data...")

    businesses = pd.read_csv('data/raw/business-licences.csv', low_memory=False)
    heritage = pd.read_csv('data/raw/heritage-sites.csv')

    print(f"Loaded {len(businesses):,} business records")
    print(f"Loaded {len(heritage):,} heritage site records")

    return businesses, heritage


def parse_coordinates(df: pd.DataFrame, coord_col: str = 'geo_point_2d') -> pd.DataFrame:
    """Parse geo_point_2d column into separate lat/lon columns."""
    print(f"\nParsing coordinates from {coord_col}...")

    # Split the coordinate string "lat, lon" into separate columns
    coords = df[coord_col].str.split(',', expand=True)

    if coords.shape[1] == 2:
        df['latitude'] = pd.to_numeric(coords[0], errors='coerce')
        df['longitude'] = pd.to_numeric(coords[1], errors='coerce')
    else:
        df['latitude'] = np.nan
        df['longitude'] = np.nan

    # Count valid coordinates
    valid_coords = df[['latitude', 'longitude']].notna().all(axis=1).sum()
    print(f"Valid coordinates: {valid_coords:,} / {len(df):,}")

    return df


def clean_business_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter business license data."""
    print("\nCleaning business data...")

    initial_count = len(df)

    # Extract year from IssuedDate before any filtering/copying
    # Use utc=True to handle mixed timezones
    df.loc[:, 'YearIssued'] = pd.to_datetime(df['IssuedDate'], errors='coerce', utc=True).dt.year

    # Parse coordinates
    df = parse_coordinates(df)

    # Filter to only businesses with valid coordinates
    df = df[df[['latitude', 'longitude']].notna().all(axis=1)].copy()
    print(f"Businesses with valid coordinates: {len(df):,} / {initial_count:,}")

    # Filter to active/issued licenses only
    df = df[df['Status'] == 'Issued'].copy()
    print(f"Active (Issued) businesses: {len(df):,}")

    # Select and rename relevant columns
    df = df[[
        'LicenceRSN',
        'BusinessName',
        'BusinessTradeName',
        'BusinessType',
        'BusinessSubType',
        'House',
        'Street',
        'City',
        'PostalCode',
        'LocalArea',
        'NumberofEmployees',
        'IssuedDate',
        'YearIssued',
        'latitude',
        'longitude'
    ]].copy()

    return df


def clean_heritage_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter heritage site data."""
    print("\nCleaning heritage site data...")

    initial_count = len(df)

    # Parse coordinates
    df = parse_coordinates(df)

    # Filter to only sites with valid coordinates
    df = df[df[['latitude', 'longitude']].notna().all(axis=1)].copy()
    print(f"Heritage sites with valid coordinates: {len(df):,} / {initial_count:,}")

    # Filter to active sites only
    df = df[df['Status'] == 'Active'].copy()
    print(f"Active heritage sites: {len(df):,}")

    # Select relevant columns
    df = df[[
        'ID',
        'StreetNumber',
        'StreetName',
        'Category',
        'BuildingNameSpecifics',
        'EvaluationGroup',
        'LocalArea',
        'latitude',
        'longitude'
    ]].copy()

    return df


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on earth (in meters).
    Uses the Haversine formula.
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth radius in meters
    r = 6371000

    return c * r


def calculate_nearest_heritage_distance(businesses: pd.DataFrame, heritage: pd.DataFrame) -> pd.DataFrame:
    """Calculate distance from each business to the nearest heritage site."""
    print("\nCalculating distances to nearest heritage sites...")
    print("This may take a few minutes for large datasets...")

    # Extract heritage site coordinates as numpy arrays
    heritage_coords = heritage[['latitude', 'longitude']].values

    # Initialize arrays for results
    min_distances = np.zeros(len(businesses))
    nearest_heritage_ids = np.zeros(len(businesses), dtype=int)

    # Calculate distances (vectorized per business, loop over businesses)
    for idx, (bus_lat, bus_lon) in enumerate(businesses[['latitude', 'longitude']].values):
        if idx % 5000 == 0:
            print(f"  Processing business {idx:,} / {len(businesses):,}")

        # Calculate distance to all heritage sites
        distances = np.array([
            haversine_distance(bus_lat, bus_lon, her_lat, her_lon)
            for her_lat, her_lon in heritage_coords
        ])

        # Find nearest
        min_idx = np.argmin(distances)
        min_distances[idx] = distances[min_idx]
        nearest_heritage_ids[idx] = heritage.iloc[min_idx]['ID']

    # Add to dataframe
    businesses['distance_to_nearest_heritage_m'] = min_distances
    businesses['nearest_heritage_id'] = nearest_heritage_ids

    print(f"\nDistance statistics (in meters):")
    print(businesses['distance_to_nearest_heritage_m'].describe())

    return businesses


def create_labels(df: pd.DataFrame, threshold_m: int = DISTANCE_THRESHOLD_M) -> pd.DataFrame:
    """Create binary labels based on distance threshold."""
    print(f"\nCreating binary labels (threshold: {threshold_m}m)...")

    df['near_heritage'] = (df['distance_to_nearest_heritage_m'] <= threshold_m).astype(int)

    # Print label distribution
    label_counts = df['near_heritage'].value_counts().sort_index()
    print(f"\nLabel distribution:")
    print(f"  Not near heritage (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  Near heritage (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features for modeling."""
    print("\nEngineering additional features...")

    # Distance-based features
    df['distance_km'] = df['distance_to_nearest_heritage_m'] / 1000
    df['log_distance'] = np.log1p(df['distance_to_nearest_heritage_m'])

    # Business type encoding (will use one-hot encoding in modeling)
    df['BusinessType'] = df['BusinessType'].fillna('Unknown')
    df['BusinessSubType'] = df['BusinessSubType'].fillna('Unknown')

    # Local area encoding
    df['LocalArea'] = df['LocalArea'].fillna('Unknown')

    # Employee count (handle missing)
    df['NumberofEmployees'] = df['NumberofEmployees'].fillna(0)

    # Year issued (handle missing)
    df['YearIssued'] = df['YearIssued'].fillna(df['YearIssued'].median())

    print(f"Final feature set includes {df.shape[1]} columns")

    return df


def save_processed_data(df: pd.DataFrame):
    """Save processed dataset."""
    print("\nSaving processed data...")

    output_path = 'data/processed/businesses_with_heritage_labels.csv'
    df.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")
    print(f"Total records: {len(df):,}")
    print(f"Total features: {df.shape[1]}")


def main():
    """Main preprocessing pipeline."""
    print("="*60)
    print("Vancouver Business Heritage Proximity - Data Preprocessing")
    print("="*60)

    # Load data
    businesses, heritage = load_data()

    # Clean data
    businesses = clean_business_data(businesses)
    heritage = clean_heritage_data(heritage)

    # Calculate distances
    businesses = calculate_nearest_heritage_distance(businesses, heritage)

    # Create labels
    businesses = create_labels(businesses, DISTANCE_THRESHOLD_M)

    # Engineer features
    businesses = engineer_features(businesses)

    # Save processed data
    save_processed_data(businesses)

    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
