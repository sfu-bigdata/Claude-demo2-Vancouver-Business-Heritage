# Data Directory

## Overview

This directory contains datasets for the Vancouver Business Heritage Proximity Predictor project.

## Directory Structure

- `raw/` - Original, unmodified datasets downloaded from source
- `processed/` - Cleaned and processed datasets ready for ML modeling

---

## Raw Data Sources

### 1. Business Licences Dataset

**File**: `raw/business-licences.csv`

**Source**: [City of Vancouver Open Data Portal - Business Licences](https://opendata.vancouver.ca/explore/dataset/business-licences/)

**Download Date**: December 8, 2025

**Description**: Contains information about business licenses issued by the City of Vancouver, including business name, type, location, status, and geographic coordinates.

**Records**: ~132,635 businesses

**Key Fields**:
- `BusinessName`, `BusinessTradeName`
- `BusinessType`, `BusinessSubType`
- `House`, `Street`, `City`, `PostalCode`
- `LocalArea` (neighborhood)
- `geo_point_2d` (latitude, longitude)
- `Status` (Issued, Expired, etc.)
- `IssuedDate`, `ExpiredDate`

---

### 2. Heritage Sites Dataset

**File**: `raw/heritage-sites.csv`

**Source**: [City of Vancouver Open Data Portal - Heritage Sites](https://opendata.vancouver.ca/explore/dataset/heritage-sites/)

**Download Date**: December 8, 2025

**Description**: Contains information about heritage buildings, structures, streetscapes, and landscape resources in Vancouver that have been deemed to have heritage value. This data is from the Vancouver Heritage Register and is updated annually.

**Records**: ~2,496 heritage sites

**Key Fields**:
- `StreetNumber`, `StreetName`
- `Category` (Heritage Buildings, Landscapes, etc.)
- `BuildingNameSpecifics`
- `EvaluationGroup` (A, B, C ratings)
- `LocalArea` (neighborhood)
- `geo_point_2d` (latitude, longitude)
- Various designation flags (Municipal, Provincial, Federal, etc.)
- `Status` (Active, etc.)

---

## Data Usage Notes

### Business Licenses
- Dataset includes both active and expired licenses
- Some records may have missing geographic coordinates
- Multiple license revisions exist for the same business

### Heritage Sites
- Sites are categorized by type and evaluation group
- Evaluation groups (A/B/C) indicate heritage significance
- Coordinates are available in both `Geom` (GeoJSON) and `geo_point_2d` formats

---

## Data Processing Pipeline

The raw data will be processed through the following steps:
1. **Cleaning**: Remove invalid records, handle missing values
2. **Geocoding**: Validate/complete coordinate information
3. **Distance Calculation**: Calculate distance from each business to nearest heritage site
4. **Labeling**: Create binary labels (within 1km = 1, beyond 1km = 0)
5. **Feature Engineering**: Extract relevant features for ML modeling

Processed data will be saved to the `processed/` directory.

---

## API Access

Both datasets can be accessed programmatically via the Vancouver Open Data API:

**Business Licences API**:
```
https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/business-licences/
```

**Heritage Sites API**:
```
https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/heritage-sites/
```

---

## Additional Resources

- [Vancouver Heritage Site Finder](https://www.heritagesitefinder.ca/) - Interactive map interface
- [Vancouver Open Data Portal](https://opendata.vancouver.ca/) - Main portal for all city datasets
