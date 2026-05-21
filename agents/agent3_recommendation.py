"""
CancerPath-Africa
Agent 3 — Referral Navigation Agent

LLM-powered referral navigator that synthesises outputs from
Agent 1 (risk assessment) and Agent 2 (facility mapping) to
generate a plain-language referral action plan for health workers.

Uses chain-of-thought prompting with explicit reasoning steps.
Supports three LLMs via Groq API (LLaMA 3, Mixtral) and Google
Gemini with graceful offline fallback to template-based recommendations.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv(override=True)

from config.config import ACTIVE_LLM, LLM_MODELS

# Load model directly from env after dotenv loads
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "llama-3.3-70b-versatile")


# ── LLM Client Initialisation ────────────────────────────
def get_llm_client():
    """Initialise LLM client based on config"""
    if ACTIVE_LLM == 'groq':
        try:
            from groq import Groq
            return Groq(api_key=os.getenv('GROQ_API_KEY')), 'groq'
        except Exception as e:
            print(f"Groq unavailable: {e}. Falling back to Gemini.")

    if ACTIVE_LLM == 'gemini' or True:
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            return genai.GenerativeModel('gemini-1.5-flash'), 'gemini'
        except Exception as e:
            print(f"Gemini unavailable: {e}. Using offline template.")
            return None, 'offline'

# ── Chain-of-Thought Prompt ──────────────────────────────
SYSTEM_PROMPT = """You are a clinical referral navigation assistant 
for CancerPath-Africa, supporting community health workers in Kenya 
with cervical cancer screening triage.

Your role is to generate clear, actionable referral recommendations 
based on patient risk assessments and facility mapping outputs.

CRITICAL RULES:
- You are NOT a diagnostician. Never suggest a diagnosis.
- Frame all outputs as referral NAVIGATION, not clinical decisions.
- Use plain language suitable for a non-specialist health worker.
- Always show your reasoning step by step.
- Flag uncertainties explicitly — never guess.
- Keep recommendations practical and location-specific.
- Reference Sub-Saharan African healthcare context.
"""

def build_prompt(agent1_result: dict, agent2_result: dict,
                 patient_data: dict) -> str:
    """Build chain-of-thought prompt from agent outputs"""

    risk_level   = agent1_result.get('risk_level', 'UNKNOWN')
    probability  = agent1_result.get('probability', 0)
    risk_factors = agent1_result.get('top_risk_factors', [])
    plain_summary = agent1_result.get('plain_summary', '')

    primary_facility = agent2_result.get('primary_facility', {})
    all_facilities   = agent2_result.get('top_facilities', [])
    rationale        = agent2_result.get('rationale', '')

    risk_factors_text = '\n'.join([
        f"  - {r['feature']} (impact score: {r['shap_value']:.3f})"
        for r in risk_factors
    ]) if risk_factors else "  - No specific risk factors identified"

    facilities_text = '\n'.join([
        f"  {f['rank']}. {f['name']} — {f['type']} "
        f"({f['county']} County, {f['distance_km']}km)"
        for f in all_facilities
    ]) if all_facilities else "  No facilities found in range"

    prompt = f"""
PATIENT RISK ASSESSMENT (from Agent 1):
- Risk Level: {risk_level}
- Probability of never screened: {probability*100:.1f}%
- Key risk factors:
{risk_factors_text}

FACILITY MAPPING (from Agent 2):
- Nearest suitable facility: {primary_facility.get('name', 'Unknown')}
- Facility type: {primary_facility.get('type', 'Unknown')}
- Distance: {primary_facility.get('distance_km', 'Unknown')} km
- County: {primary_facility.get('county', 'Unknown')}
- Owner: {primary_facility.get('owner', 'Unknown')}
- All options:
{facilities_text}

PATIENT CONTEXT:
- Age: {patient_data.get('age', 'Unknown')}
- Location: {'Rural' if patient_data.get('rural') else 'Urban'}
- Education: {_edu_label(patient_data.get('education_level', 0))}
- HIV tested: {'Yes' if patient_data.get('hiv_tested') else 'No'}

Please generate a referral navigation recommendation following 
these exact steps:

STEP 1 — URGENCY ASSESSMENT:
Assess urgency based on risk level and probability.

STEP 2 — FACILITY RECOMMENDATION:
Recommend the most appropriate facility with specific justification.

STEP 3 — PRACTICAL GUIDANCE:
Give practical advice for getting to the facility 
(what to bring, who to ask for, what service to request).

STEP 4 — UNCERTAINTIES:
Flag any missing information that could affect this recommendation.

STEP 5 — HEALTH WORKER ACTION:
List 3 specific actions the health worker should take now.

Keep each step concise. Use plain language.
"""
    return prompt

# ── LLM Call ─────────────────────────────────────────────
def call_llm(prompt: str, client, client_type: str) -> str:
    """Call LLM with prompt and return response text"""
    try:
        if client_type == 'groq':
            response = client.chat.completions.create(
                model=ACTIVE_MODEL or 'llama-3.3-70b-versatile',
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            return response.choices[0].message.content

        elif client_type == 'gemini':
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
            response    = client.generate_content(full_prompt)
            return response.text

    except Exception as e:
        return None

# ── Offline Template Fallback ────────────────────────────
def generate_template_recommendation(
    agent1_result: dict,
    agent2_result: dict
) -> str:
    """
    Offline fallback — template-based recommendation
    used when no internet connection is available.
    """
    risk_level = agent1_result.get('risk_level', 'UNKNOWN')
    probability = agent1_result.get('probability', 0)
    primary    = agent2_result.get('primary_facility', {})

    urgency_map = {
        'HIGH':     'URGENT — within 7 days',
        'MODERATE': 'SOON — within 30 days',
        'LOW':      'ROUTINE — next scheduled visit'
    }

    template = f"""
REFERRAL NAVIGATION RECOMMENDATION (Offline Mode)
==================================================

STEP 1 — URGENCY:
{urgency_map.get(risk_level, 'See health worker')}
Risk probability: {probability*100:.1f}%

STEP 2 — RECOMMENDED FACILITY:
{primary.get('name', 'Nearest District Hospital')}
Type: {primary.get('type', 'District Hospital')}
Distance: {primary.get('distance_km', 'Unknown')} km
County: {primary.get('county', 'Unknown')}

STEP 3 — PRACTICAL GUIDANCE:
- Request cervical cancer screening service on arrival
- Bring any previous health records if available
- Arrive early — screening services often limited in afternoon
- Ask for the reproductive health or outpatient department

STEP 4 — UNCERTAINTIES:
- Full clinical history not available
- Insurance/NHIS status not confirmed
- Appointment availability not checked

STEP 5 — HEALTH WORKER ACTIONS:
1. Complete referral letter with patient details
2. Schedule follow-up call in 5 days to confirm appointment
3. Flag patient for community health volunteer follow-up
   if no response within 7 days
"""
    return template

# ── Helper ───────────────────────────────────────────────
def _edu_label(level):
    labels = {0: 'No education', 1: 'Primary',
              2: 'Secondary',    3: 'Higher'}
    return labels.get(int(level), 'Unknown')

# ── Core Recommendation Function ─────────────────────────
def generate_recommendation(
    agent1_result: dict,
    agent2_result: dict,
    patient_data:  dict
) -> dict:
    """
    Generate referral navigation recommendation.

    Args:
        agent1_result: output from Agent 1 predict_risk()
        agent2_result: output from Agent 2 find_nearest_facilities()
        patient_data:  original patient input dict

    Returns:
        dict with recommendation text, mode, and metadata
    """
    client, client_type = get_llm_client()

    if client is None or client_type == 'offline':
        recommendation = generate_template_recommendation(
            agent1_result, agent2_result
        )
        mode = 'offline_template'
    else:
        prompt = build_prompt(
            agent1_result, agent2_result, patient_data
        )
        llm_response = call_llm(prompt, client, client_type)

        if llm_response:
            recommendation = llm_response
            mode           = f'llm_{client_type}_{ACTIVE_MODEL}'
        else:
            recommendation = generate_template_recommendation(
                agent1_result, agent2_result
            )
            mode = 'offline_template_fallback'

    return {
        'recommendation': recommendation,
        'mode':           mode,
        'risk_level':     agent1_result.get('risk_level'),
        'facility':       agent2_result.get('primary_facility', {})
                          .get('name', 'Unknown'),
        'distance_km':    agent2_result.get('primary_facility', {})
                          .get('distance_km', 'Unknown'),
    }

# ── Test Function ────────────────────────────────────────
if __name__ == "__main__":
    # Mock agent outputs for testing
    mock_agent1 = {
        'risk_level':       'HIGH',
        'probability':      0.9489,
        'top_risk_factors': [
            {'feature': 'Ever Tested for HIV',
             'shap_value': 1.666},
            {'feature': 'Age',
             'shap_value': 0.587},
            {'feature': 'Education Level',
             'shap_value': 0.564}
        ],
        'plain_summary': '🔴 HIGH RISK — Urgent referral recommended'
    }

    mock_agent2 = {
        'risk_level':      'HIGH',
        'primary_facility': {
            'name':        'Kenyatta National Hospital',
            'type':        'National Referral Hospital',
            'county':      'Nairobi',
            'owner':       'Ministry of Health',
            'distance_km': 1.9
        },
        'top_facilities': [
            {
                'rank': 1,
                'name': 'Kenyatta National Hospital',
                'type': 'National Referral Hospital',
                'county': 'Nairobi',
                'distance_km': 1.9
            }
        ],
        'rationale': 'HIGH risk — District Hospital minimum required'
    }

    mock_patient = {
        'age':             22,
        'rural':           1,
        'education_level': 0,
        'hiv_tested':      0
    }

    print("Testing Agent 3 — Referral Navigator")
    print("="*50)

    result = generate_recommendation(
        mock_agent1, mock_agent2, mock_patient
    )

    print(f"Mode: {result['mode']}")
    print(f"Facility: {result['facility']}")
    print(f"\nRecommendation:\n{result['recommendation']}")