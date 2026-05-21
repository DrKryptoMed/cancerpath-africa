"""
CancerPath-Africa
Agent 2 — Resource Mapper

Geospatial rule-based facility matching module.
Identifies the nearest oncology-capable health facilities
based on patient location and risk level.

Note: This is a geospatial optimisation module, not an ML model.
Transparency is achieved through explicit decision criteria display.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from geopy.distance import geodesic
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import MODELS_SAVED

# ── Constants ────────────────────────────────────────────
FACILITIES_PATH = (
    Path(__file__).resolve().parent.parent /
    'data/processed/kenya_oncology_facilities.csv'
)

# Facility capability tiers for oncology
FACILITY_TIERS = {
    'National Referral Hospital':   4,
    'Provincial General Hospital':  3,
    'District Hospital':            2,
    'Sub-District Hospital':        1,
    'Medical Centre':               1,
}

# Minimum facility tier by risk level
MIN_TIER_BY_RISK = {
    'HIGH':     2,  # District Hospital or above
    'MODERATE': 1,  # Any oncology-capable facility
    'LOW':      1,  # Any oncology-capable facility
}

# Maximum recommended referral distances by risk
MAX_DISTANCE_BY_RISK = {
    'HIGH':     300,  # km — urgent, willing to travel further
    'MODERATE': 200,
    'LOW':      100,
}

# ── Load Facilities ──────────────────────────────────────
def load_facilities():
    """Load geocoded Kenya oncology facilities"""
    df = pd.read_csv(FACILITIES_PATH)
    df = df[df['Latitude'].notna() & df['Longitude'].notna()]
    df['tier'] = df['Type'].map(FACILITY_TIERS).fillna(1)
    return df

# ── Core Mapping Function ────────────────────────────────
def find_nearest_facilities(
    patient_lat:   float,
    patient_lon:   float,
    risk_level:    str,
    top_n:         int = 3
) -> dict:
    """
    Find nearest oncology-capable facilities for a patient.

    Args:
        patient_lat:  patient latitude
        patient_lon:  patient longitude
        risk_level:   HIGH / MODERATE / LOW from Agent 1
        top_n:        number of facilities to return

    Returns:
        dict with ranked facilities and decision rationale
    """
    df = load_facilities()

    min_tier    = MIN_TIER_BY_RISK.get(risk_level, 1)
    max_dist    = MAX_DISTANCE_BY_RISK.get(risk_level, 200)

    # Filter by minimum capability tier
    df_filtered = df[df['tier'] >= min_tier].copy()

    # Calculate distances
    patient_coords = (patient_lat, patient_lon)
    df_filtered['distance_km'] = df_filtered.apply(
        lambda row: round(
            geodesic(
                patient_coords,
                (row['Latitude'], row['Longitude'])
            ).kilometers, 1
        ),
        axis=1
    )

    # Filter by max distance
    df_in_range = df_filtered[
        df_filtered['distance_km'] <= max_dist
    ].copy()

    # If nothing in range expand search
    if len(df_in_range) < top_n:
        df_in_range = df_filtered.copy()

    # Sort by distance
    df_ranked = df_in_range.sort_values(
        'distance_km'
    ).head(top_n).reset_index(drop=True)

    # Build results
    facilities = []
    for _, row in df_ranked.iterrows():
        facilities.append({
            'rank':          int(row.name) + 1,
            'name':          row['Facility Name'],
            'type':          row['Type'],
            'county':        row['County'],
            'owner':         row['Owner'],
            'distance_km':   row['distance_km'],
            'tier':          int(row['tier']),
            'latitude':      row['Latitude'],
            'longitude':     row['Longitude'],
            'operational':   row['Operational Status']
        })

    # Decision rationale
    rationale = _build_rationale(
        risk_level, min_tier, max_dist,
        len(df_filtered), facilities
    )

    return {
        'patient_location':  (patient_lat, patient_lon),
        'risk_level':        risk_level,
        'min_tier_required': min_tier,
        'max_distance_km':   max_dist,
        'facilities_found':  len(facilities),
        'top_facilities':    facilities,
        'primary_facility':  facilities[0] if facilities else None,
        'rationale':         rationale,
        'decision_criteria': {
            'risk_level':        risk_level,
            'min_facility_tier': min_tier,
            'tier_labels': {
                1: 'Sub-District Hospital / Medical Centre',
                2: 'District Hospital',
                3: 'Provincial General Hospital',
                4: 'National Referral Hospital'
            },
            'max_distance_km':   max_dist,
            'total_candidates':  len(df_filtered)
        }
    }

# ── Decision Rationale ───────────────────────────────────
def _build_rationale(
    risk_level, min_tier,
    max_dist, n_candidates, facilities
):
    """Build transparent decision explanation"""

    tier_names = {
        1: 'Sub-District Hospital or above',
        2: 'District Hospital or above',
        3: 'Provincial General Hospital or above',
        4: 'National Referral Hospital only'
    }

    if not facilities:
        return (
            f"No oncology-capable facilities found within "
            f"{max_dist}km. Extended search recommended."
        )

    primary = facilities[0]
    rationale = (
        f"Risk level {risk_level} requires minimum facility "
        f"tier {min_tier} ({tier_names[min_tier]}). "
        f"Search radius: {max_dist}km. "
        f"{n_candidates} eligible facilities found. "
        f"Nearest: {primary['name']} ({primary['county']} County, "
        f"{primary['distance_km']}km away)."
    )

    return rationale

# ── County-Level Fallback ────────────────────────────────
def find_by_county(county: str, risk_level: str) -> dict:
    """
    Fallback when GPS coordinates unavailable.
    Finds facilities within the same county.
    """
    df = load_facilities()
    min_tier = MIN_TIER_BY_RISK.get(risk_level, 1)

    df_county = df[
        (df['County'].str.lower() == county.lower()) &
        (df['tier'] >= min_tier)
    ].copy()

    if df_county.empty:
        # Expand to neighbouring counties
        return {
            'facilities_found': 0,
            'message': (
                f"No facilities found in {county} County "
                f"meeting tier {min_tier} requirement. "
                f"Nearest county referral recommended."
            )
        }

    facilities = []
    for _, row in df_county.iterrows():
        facilities.append({
            'name':    row['Facility Name'],
            'type':    row['Type'],
            'county':  row['County'],
            'owner':   row['Owner'],
            'tier':    int(row['tier'])
        })

    return {
        'county':           county,
        'risk_level':       risk_level,
        'facilities_found': len(facilities),
        'facilities':       facilities,
        'primary_facility': facilities[0]
    }

# ── Test Function ────────────────────────────────────────
if __name__ == "__main__":
    # Test with Nairobi coordinates
    print("Testing Agent 2 — Resource Mapper")
    print("="*50)

    result = find_nearest_facilities(
        patient_lat = -1.2921,
        patient_lon = 36.8219,
        risk_level  = 'HIGH',
        top_n       = 3
    )

    print(f"Risk Level:       {result['risk_level']}")
    print(f"Facilities Found: {result['facilities_found']}")
    print(f"\nDecision Rationale:")
    print(f"  {result['rationale']}")
    print(f"\nTop {result['facilities_found']} Facilities:")
    for f in result['top_facilities']:
        print(f"\n  [{f['rank']}] {f['name']}")
        print(f"       Type:     {f['type']}")
        print(f"       County:   {f['county']}")
        print(f"       Distance: {f['distance_km']} km")
        print(f"       Owner:    {f['owner']}")

    # Test county fallback
    print("\n" + "="*50)
    print("Testing County Fallback (Kisumu):")
    county_result = find_by_county('Kisumu', 'HIGH')
    print(f"Facilities in Kisumu: {county_result['facilities_found']}")
    if county_result['facilities_found'] > 0:
        print(f"Primary: {county_result['primary_facility']['name']}")