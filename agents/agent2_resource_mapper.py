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

# County centroids for Kenya — approximate centres
COUNTY_CENTROIDS = {
    "Nairobi": (-1.2921, 36.8219),
    "Mombasa": (-4.0435, 39.6682),
    "Kisumu": (-0.0917, 34.7679),
    "Nakuru": (-0.3031, 36.0800),
    "Eldoret": (0.5143, 35.2698),
    "Kiambu": (-1.1741, 36.8349),
    "Kakamega": (0.2827, 34.7519),
    "Garissa": (-0.4532, 39.6461),
    "Turkana": (3.1189, 35.5975),
    "Kisii": (-0.6817, 34.7667),
    "Meru": (0.0467, 37.6492),
    "Machakos": (-1.5177, 37.2634),
    "Nyeri": (-0.4167, 36.9500),
    "Kilifi": (-3.5107, 39.9093),
    "Kwale": (-4.1740, 39.4521),
    "Murang'a": (-0.7833, 37.1500),
    "Kirinyaga": (-0.5594, 37.3392),
    "Nyandarua": (-0.1833, 36.6000),
    "Laikipia": (0.3667, 36.7833),
    "Samburu": (1.2167, 36.9833),
    "Isiolo": (0.3542, 38.0009),
    "Marsabit": (2.3284, 37.9899),
    "Mandera": (3.9366, 41.8670),
    "Wajir": (1.7471, 40.0573),
    "Tana River": (-1.5000, 39.8333),
    "Lamu": (-2.2694, 40.9027),
    "Taita Taveta": (-3.3167, 38.4833),
    "Makueni": (-2.2588, 37.8942),
    "Kitui": (-1.3667, 38.0167),
    "Embu": (-0.5333, 37.4500),
    "Tharaka Nithi": (-0.2833, 37.9167),
    "Siaya": (-0.0612, 34.2873),
    "Homa Bay": (-0.5273, 34.4572),
    "Migori": (-1.0634, 34.4731),
    "Nyamira": (-0.5667, 34.9333),
    "Bomet": (-0.7833, 35.3333),
    "Kericho": (-0.3689, 35.2863),
    "Nandi": (0.1833, 35.1167),
    "Uasin Gishu": (0.5204, 35.2699),
    "Trans Nzoia": (1.0167, 34.9667),
    "West Pokot": (1.7333, 35.1167),
    "Elgeyo Marakwet": (0.7167, 35.5167),
    "Baringo": (0.8333, 35.9833),
    "Kajiado": (-2.0987, 36.7819),
    "Narok": (-1.0833, 35.8700),
    "Bungoma": (0.5635, 34.5606),
    "Busia": (0.4608, 34.1112),
    "Vihiga": (0.0667, 34.7167),
}

# ── County-Level Fallback ────────────────────────────────
def find_by_county(county: str, risk_level: str) -> dict:
    df = load_facilities()
    min_tier = MIN_TIER_BY_RISK.get(risk_level, 1)

    df_county = df[
        (df['County'].str.lower() == county.lower()) &
        (df['tier'] >= min_tier)
    ].copy()

    if df_county.empty:
        return {
            'facilities_found': 0,
            'message': (
                f"No facilities found in {county} County "
                f"meeting tier {min_tier} requirement."
            )
        }

    # Use county centroid for approximate distances
    centroid = COUNTY_CENTROIDS.get(county)
    if centroid and 'Latitude' in df_county.columns:
        df_county = df_county[
            df_county['Latitude'].notna()
        ].copy()
        df_county['distance_km'] = df_county.apply(
            lambda row: round(
                geodesic(
                    centroid,
                    (row['Latitude'], row['Longitude'])
                ).kilometers, 1
            ) if pd.notna(row['Latitude']) else None,
            axis=1
        )
        df_county = df_county.sort_values(
            'distance_km'
        ).reset_index(drop=True)
    else:
        df_county['distance_km'] = 'N/A'

    facilities = []
    for _, row in df_county.head(3).iterrows():
        facilities.append({
            'name':        row['Facility Name'],
            'type':        row['Type'],
            'county':      row['County'],
            'owner':       row['Owner'],
            'tier':        int(row['tier']),
            'distance_km': row.get('distance_km', 'N/A'),
            'latitude':    row.get('Latitude'),
            'longitude':   row.get('Longitude')
        })

    return {
        'county':           county,
        'risk_level':       risk_level,
        'facilities_found': len(df_county),
        'facilities':       facilities,
        'primary_facility': facilities[0] if facilities else None,
        'note': 'Distances estimated from county centroid'
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

