"""
CancerPath-Africa
Agent 4 — Referral Compliance Barrier Predictor

Loads the trained Logistic Regression model and provides:
- Compliance barrier prediction with optimised threshold (0.45)
- SHAP-based barrier importance explanation
- SHAP-grounded counterfactual intervention recommendations
- Patient-friendly plain language output
"""

import joblib
import json
import numpy as np
import pandas as pd
import shap
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import MODELS_SAVED, RANDOM_STATE

# ── Constants ────────────────────────────────────────────
FEATURE_NAMES = [
    'v012', 'v190', 'rural', 'v106',
    'hiv_tested', 'hiv_positive', 'v394', 'v483a'
]

FRIENDLY_NAMES = {
    'v012':         'Age',
    'v190':         'Wealth Index',
    'rural':        'Rural Location',
    'v106':         'Education Level',
    'hiv_tested':   'Ever Tested for HIV',
    'hiv_positive': 'HIV Status',
    'v394':         'Visited Facility Last 12 Months',
    'v483a':        'Minutes to Facility'
}

COMPLIANCE_LEVELS = {
    'HIGH_BARRIER':    {'label': 'HIGH BARRIER',    'emoji': '🔴'},
    'MODERATE_BARRIER':{'label': 'MODERATE BARRIER','emoji': '🟡'},
    'LOW_BARRIER':     {'label': 'LOW BARRIER',     'emoji': '🟢'},
}

# ── Load Model ───────────────────────────────────────────
def load_agent4():
    """Load model, threshold and counterfactual rules"""
    model_path     = MODELS_SAVED / 'agent4_compliance_final.pkl'
    threshold_path = MODELS_SAVED / 'agent4_threshold.json'
    cf_rules_path  = MODELS_SAVED / 'agent4_counterfactual_rules.json'

    pipeline = joblib.load(model_path)

    with open(threshold_path) as f:
        threshold = json.load(f)['agent4_optimal_threshold']

    with open(cf_rules_path) as f:
        cf_rules = json.load(f)

    return pipeline, threshold, cf_rules

# ── Core Prediction Function ─────────────────────────────
def predict_compliance(patient_data: dict) -> dict:
    """
    Predict referral compliance barrier for a patient.

    Args:
        patient_data: dict with keys matching FEATURE_NAMES
            - v012:         age (15-49)
            - v190:         wealth index (1-5)
            - rural:        rural location (0=urban, 1=rural)
            - v106:         education level (0-3)
            - hiv_tested:   ever tested for HIV (0/1)
            - hiv_positive: HIV status (0/1/2)
            - v394:         visited facility last 12 months (0/1)
            - v483a:        minutes to nearest facility

    Returns:
        dict with compliance barrier level, probability,
        SHAP explanation, and intervention recommendations
    """
    pipeline, threshold, cf_rules = load_agent4()

    # Build feature vector
    features = pd.DataFrame([{
        f: patient_data.get(f, 0) for f in FEATURE_NAMES
    }])

    # Cap minutes to facility
    features['v483a'] = features['v483a'].clip(upper=200)

    # Extract pipeline components
    scaler   = pipeline.named_steps['scaler']
    lr_model = pipeline.named_steps['model']

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict probability of compliance barrier
    prob_barrier = pipeline.predict_proba(features)[0][1]

    # Apply threshold
    prediction = int(prob_barrier >= threshold)

    # Determine barrier level
    if prob_barrier >= 0.75:
        barrier_level = 'HIGH_BARRIER'
    elif prob_barrier >= 0.45:
        barrier_level = 'MODERATE_BARRIER'
    else:
        barrier_level = 'LOW_BARRIER'

    # SHAP explanation
    try:
        X_train_sample = pd.read_csv(
            str(MODELS_SAVED.parent.parent /
                'data/processed/X_agent4.csv')
        ).sample(100, random_state=RANDOM_STATE)
        X_train_sample['v483a'] = X_train_sample['v483a'].clip(upper=200)
        X_train_scaled = scaler.transform(X_train_sample)
        explainer   = shap.LinearExplainer(lr_model, X_train_scaled)
    except Exception:
        explainer   = shap.LinearExplainer(lr_model, features_scaled)

    shap_values = explainer.shap_values(features_scaled)

    # Top barriers and protective factors
    shap_df = pd.DataFrame({
        'feature':     [FRIENDLY_NAMES[f] for f in FEATURE_NAMES],
        'shap_value':  shap_values[0],
        'feature_val': features.values[0]
    }).sort_values('shap_value', ascending=False)

    top_barriers = shap_df[
        shap_df['shap_value'] > 0
    ].head(3)

    protective_factors = shap_df[
        shap_df['shap_value'] < 0
    ].head(2)

    # Generate interventions
    interventions = _generate_interventions(
        patient_data, cf_rules, prob_barrier
    )

    # Plain language summary
    plain_summary = _generate_plain_summary(
        barrier_level, prob_barrier,
        top_barriers, interventions
    )

    return {
        'barrier_level':      barrier_level,
        'barrier_label':      COMPLIANCE_LEVELS[barrier_level]['label'],
        'probability':        round(prob_barrier, 4),
        'prediction':         prediction,
        'threshold_used':     threshold,
        'top_barriers':       top_barriers.to_dict('records'),
        'protective_factors': protective_factors.to_dict('records'),
        'shap_values':        shap_values[0].tolist(),
        'feature_names':      [FRIENDLY_NAMES[f] for f in FEATURE_NAMES],
        'feature_values':     features.values[0].tolist(),
        'expected_value':     explainer.expected_value,
        'interventions':      interventions,
        'plain_summary':      plain_summary,
    }

# ── Intervention Generator ───────────────────────────────
def _generate_interventions(patient_data: dict,
                             cf_rules: dict,
                             prob_barrier: float) -> list:
    """Generate SHAP-grounded intervention recommendations"""
    
    # Only generate interventions for moderate or high barrier patients
    if prob_barrier < 0.45:
        return []
    
    interventions = []

    # Rule 1 — Minutes to facility (SHAP rank 1)
    minutes = patient_data.get('v483a', 0)
    if minutes > 30:
        interventions.append({
            'rank':         1,
            'intervention': 'Transport Support',
            'detail':       f"Travel time {minutes:.0f} min → "
                            f"target <30 min",
            'feasibility':  'High',
            'action':       'Arrange community transport or '
                            'mobile clinic visit'
        })

    # Rule 2 — Wealth index (SHAP rank 2)
    wealth = patient_data.get('v190', 3)
    if wealth <= 2:
        interventions.append({
            'rank':         2,
            'intervention': 'Financial Support',
            'detail':       f"Wealth quintile {wealth:.0f} — "
                            f"poorest bracket",
            'feasibility':  'Medium',
            'action':       'Enrol in NHIS or provide '
                            'transport voucher'
        })

    # Rule 3 — Education (SHAP rank 3)
    edu = patient_data.get('v106', 1)
    if edu <= 1:
        interventions.append({
            'rank':         3,
            'intervention': 'Health Literacy',
            'detail':       f"Education level {edu:.0f} — "
                            f"limited health literacy",
            'feasibility':  'Medium',
            'action':       'Provide community health '
                            'education session'
        })

    # Rule 4 — Recent facility visit (SHAP rank 4)
    recent_visit = patient_data.get('v394', 0)
    if recent_visit == 0:
        interventions.append({
            'rank':         4,
            'intervention': 'Community Outreach',
            'detail':       "No facility visit in last 12 months",
            'feasibility':  'High',
            'action':       'Schedule outreach visit or '
                            'mobile clinic contact'
        })

    return interventions

# ── Plain Language Summary ───────────────────────────────
def _generate_plain_summary(barrier_level, prob,
                             top_barriers, interventions):
    """Generate plain English summary for health workers"""
    emoji = COMPLIANCE_LEVELS[barrier_level]['emoji']

    summary = (
        f"{emoji} {COMPLIANCE_LEVELS[barrier_level]['label']}\n"
        f"This patient has a {prob*100:.1f}% probability of facing "
        f"significant barriers to completing their referral.\n\n"
    )

    if top_barriers is not None and len(top_barriers) > 0:
        summary += "Primary barriers identified:\n"
        for _, row in top_barriers.iterrows():
            summary += f"  • {row['feature']}\n"

    if interventions:
        summary += "\nRecommended interventions:\n"
        for iv in interventions:
            summary += (
                f"  [{iv['rank']}] {iv['intervention']}: "
                f"{iv['action']}\n"
            )
    else:
        summary += (
            "\nNo major barriers identified. "
            "Standard follow-up recommended."
        )

    return summary

# ── Test Function ────────────────────────────────────────
if __name__ == "__main__":
    # Test with a high-barrier patient
    test_patient = {
        'v012':         28,    # age 28
        'v190':         4,     # higher wealth quintile
        'rural':        1,     # rural
        'v106':         1,     # primary education
        'hiv_tested':   1,     # tested for HIV
        'hiv_positive': 0,     # negative
        'v394':         0,     # no recent facility visit
        'v483a':        20,    # close to facility
    }

    print("Testing Agent 4 — Compliance Predictor")
    print("="*50)

    result = predict_compliance(test_patient)

    print(f"Barrier Level:  {result['barrier_label']}")
    print(f"Probability:    {result['probability']:.4f}")
    print(f"Threshold:      {result['threshold_used']}")
    print(f"\nPlain Summary:\n{result['plain_summary']}")
    print(f"\nTop Barriers:")
    for b in result['top_barriers']:
        print(f"  {b['feature']}: SHAP={b['shap_value']:.3f}")
    print(f"\nInterventions:")
    for iv in result['interventions']:
        print(f"  [{iv['rank']}] {iv['intervention']}: {iv['action']}")