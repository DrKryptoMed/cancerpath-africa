"""
CancerPath-Africa
Agent 1 : Cervical Cancer Screening Non-Uptake Risk Screener

Loads the trained Logistic Regression model and provides:
- Risk prediction with optimised threshold (0.30)
- SHAP-based feature importance explanation
- Profile-report inconsistency detection
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
    'hiv_tested', 'hiv_positive', 'v483a'
]

FRIENDLY_NAMES = {
    'v012':         'Age',
    'v190':         'Wealth Index',
    'rural':        'Rural Location',
    'v106':         'Education Level',
    'hiv_tested':   'Ever Tested for HIV',
    'hiv_positive': 'HIV Status',
    'v483a':        'Minutes to Facility'
}

RISK_LEVELS = {
    'HIGH':     {'label': 'HIGH RISK',     'color': 'red',    'emoji': '🔴'},
    'MODERATE': {'label': 'MODERATE RISK', 'color': 'orange', 'emoji': '🟡'},
    'LOW':      {'label': 'LOW RISK',      'color': 'green',  'emoji': '🟢'},
}

# ── Load Model and SHAP Artefacts ────────────────────────
def load_agent1():
    """Load model, threshold and SHAP explainer"""
    model_path     = MODELS_SAVED / 'agent1_risk_screener_final.pkl'
    threshold_path = MODELS_SAVED / 'agent1_threshold.json'

    pipeline = joblib.load(model_path)

    with open(threshold_path) as f:
        threshold_config = json.load(f)
    threshold = threshold_config['agent1_optimal_threshold']

    # Rebuild SHAP explainer from pipeline
    scaler   = pipeline.named_steps['scaler']
    lr_model = pipeline.named_steps['model']

    return pipeline, scaler, lr_model, threshold

# ── Core Prediction Function ─────────────────────────────
def predict_risk(patient_data: dict) -> dict:
    """
    Predicts cervical cancer screening non-uptake risk for a patient.

    Args:
        patient_data: dict with keys matching FEATURE_NAMES
            - v012:         age (15-49)
            - v190:         wealth index (1-5)
            - rural:        rural location (0=urban, 1=rural)
            - v106:         education level (0=none, 1=primary,
                            2=secondary, 3=higher)
            - hiv_tested:   ever tested for HIV (0=no, 1=yes)
            - hiv_positive: HIV status (0=negative, 1=positive,
                            2=unknown)
            - v483a:        minutes to nearest facility

    Returns:
        dict with risk level, probability, explanation, and flags
    """
    pipeline, scaler, lr_model, threshold = load_agent1()

    # Build feature vector
    features = pd.DataFrame([{
        f: patient_data.get(f, 0) for f in FEATURE_NAMES
    }])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict probability
    prob_never_screened = pipeline.predict_proba(features)[0][1]

    # Apply threshold
    prediction = int(prob_never_screened >= threshold)

    # Determine risk level
    if prob_never_screened >= 0.75:
        risk_level = 'HIGH'
    elif prob_never_screened >= 0.50:
        risk_level = 'MODERATE'
    else:
        risk_level = 'LOW'

   # SHAP explanation
    # Load training data for stable explainer background
    try:
        X_train_sample = pd.read_csv(
            str(MODELS_SAVED.parent.parent / 
                'data/processed/X_agent1.csv')
        ).sample(100, random_state=RANDOM_STATE)
        X_train_scaled = scaler.transform(X_train_sample)
        explainer   = shap.LinearExplainer(lr_model, X_train_scaled)
    except Exception:
        explainer   = shap.LinearExplainer(lr_model, features_scaled)
    
    shap_values = explainer.shap_values(features_scaled)

    # Top 3 contributing features
    shap_df = pd.DataFrame({
        'feature':     [FRIENDLY_NAMES[f] for f in FEATURE_NAMES],
        'shap_value':  shap_values[0],
        'feature_val': features.values[0]
    }).sort_values('shap_value', ascending=False)

    top_risk_factors = shap_df[
        shap_df['shap_value'] > 0
    ].head(3)

    protective_factors = shap_df[
        shap_df['shap_value'] < 0
    ].head(2)

    # Inconsistency detection
    inconsistency_flag = False
    inconsistency_msg  = None

    if patient_data.get('self_reported_screened') == True:
        if prob_never_screened > 0.65:
            inconsistency_flag = True
            inconsistency_msg  = (
                "Self-reported screening history is inconsistent "
                "with patient risk profile. Recommend verbal "
                "verification before deprioritising."
            )

    # Plain language summary
    plain_summary = _generate_plain_summary(
        risk_level, prob_never_screened,
        top_risk_factors, patient_data
    )

    return {
        'risk_level':          risk_level,
        'risk_label':          RISK_LEVELS[risk_level]['label'],
        'probability':         round(prob_never_screened, 4),
        'prediction':          prediction,
        'threshold_used':      threshold,
        'top_risk_factors':    top_risk_factors.to_dict('records'),
        'protective_factors':  protective_factors.to_dict('records'),
        'shap_values':         shap_values[0].tolist(),
        'feature_names':       [FRIENDLY_NAMES[f] for f in FEATURE_NAMES],
        'feature_values':      features.values[0].tolist(),
        'expected_value':      explainer.expected_value,
        'inconsistency_flag':  inconsistency_flag,
        'inconsistency_msg':   inconsistency_msg,
        'plain_summary':       plain_summary,
    }

# ── Plain Language Summary ───────────────────────────────
def _generate_plain_summary(risk_level, prob, top_factors, patient_data):
    """Generate plain English explanation for health workers"""
    emoji = RISK_LEVELS[risk_level]['emoji']

    summary = (
        f"{emoji} {RISK_LEVELS[risk_level]['label']}\n"
        f"This patient has a {prob*100:.1f}% probability of never "
        f"having been screened for cervical cancer.\n\n"
    )

    if top_factors is not None and len(top_factors) > 0:
        summary += "Primary risk drivers:\n"
        for _, row in top_factors.iterrows():
            summary += f"  • {row['feature']}\n"

    if risk_level == 'HIGH':
        summary += (
            "\nRecommended action: Urgent referral for cervical "
            "cancer screening within 7 days."
        )
    elif risk_level == 'MODERATE':
        summary += (
            "\nRecommended action: Schedule cervical cancer "
            "screening within 30 days."
        )
    else:
        summary += (
            "\nRecommended action: Routine monitoring. "
            "Encourage next scheduled screening."
        )

    return summary

# ── Test Function ────────────────────────────────────────
if __name__ == "__main__":
    # Test with a sample high-risk patient
    test_patient = {
        'v012':         22,    # age 22
        'v190':         1,     # poorest wealth quintile
        'rural':        1,     # rural
        'v106':         0,     # no education
        'hiv_tested':   0,     # never tested for HIV
        'hiv_positive': 2,     # unknown HIV status
        'v483a':        60,    # 60 minutes to facility
        'self_reported_screened': False
    }

    print("Testing Agent 1 — Risk Screener")
    print("="*50)
    result = predict_risk(test_patient)

    print(f"Risk Level:    {result['risk_label']}")
    print(f"Probability:   {result['probability']:.4f}")
    print(f"Threshold:     {result['threshold_used']}")
    print(f"\nPlain Summary:\n{result['plain_summary']}")
    print(f"\nTop Risk Factors:")
    for factor in result['top_risk_factors']:
        print(f"  {factor['feature']}: SHAP={factor['shap_value']:.3f}")
    print(f"\nInconsistency Flag: {result['inconsistency_flag']}")