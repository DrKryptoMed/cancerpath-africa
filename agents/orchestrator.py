"""
CancerPath-Africa
Orchestrator — Multi-Agent Pipeline Controller

Connects all four agents into a unified end-to-end pipeline:
  Agent 1 → Risk Screener
  Agent 2 → Resource Mapper  
  Agent 3 → Referral Navigator
  Agent 4 → Compliance Predictor

Also handles:
  - Profile-report inconsistency detection
  - Offline fallback logic
  - Complete audit trail logging
  - End-to-end patient case summary
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.agent1_risk_screener  import predict_risk
from agents.agent2_resource_mapper import find_nearest_facilities, find_by_county
from agents.agent3_recommendation  import generate_recommendation
from agents.agent4_compliance      import predict_compliance

# ── Core Pipeline ────────────────────────────────────────
def run_pipeline(patient_data: dict) -> dict:
    """
    Run the complete CancerPath-Africa pipeline for a patient.

    Args:
        patient_data: dict containing all patient inputs:

        Agent 1 inputs:
          - v012:         age (15-49)
          - v190:         wealth index (1-5)
          - rural:        0=urban, 1=rural
          - v106:         education (0=none,1=primary,
                          2=secondary,3=higher)
          - hiv_tested:   0=no, 1=yes
          - hiv_positive: 0=negative, 1=positive, 2=unknown
          - v483a:        minutes to nearest facility

        Agent 4 additional inputs:
          - v394:         visited facility last 12 months (0/1)

        Location inputs (one of):
          - patient_lat, patient_lon: GPS coordinates
          - county: county name (fallback)

        Optional:
          - self_reported_screened: True/False

    Returns:
        Complete pipeline output dict with all agent results
        and unified patient summary
    """

    timestamp = datetime.datetime.now().isoformat()
    pipeline_output = {'timestamp': timestamp, 'patient_data': patient_data}

    print("\n" + "="*60)
    print("CancerPath-Africa — Running Patient Pipeline")
    print("="*60)

    # ── Step 1: Agent 1 — Risk Screening ────────────────
    print("\n[1/4] Agent 1: Cervical Cancer Risk Screening...")
    try:
        agent1_result = predict_risk(patient_data)
        pipeline_output['agent1'] = agent1_result
        print(f"      Result: {agent1_result['risk_label']} "
              f"({agent1_result['probability']*100:.1f}%)")
    except Exception as e:
        print(f"      ERROR: {e}")
        pipeline_output['agent1'] = {'error': str(e)}
        return pipeline_output

    # ── Step 2: Agent 2 — Resource Mapping ──────────────
    print("\n[2/4] Agent 2: Facility Resource Mapping...")
    try:
        lat = patient_data.get('patient_lat')
        lon = patient_data.get('patient_lon')
        county = patient_data.get('county')

        if lat and lon:
            agent2_result = find_nearest_facilities(
                patient_lat=lat,
                patient_lon=lon,
                risk_level=agent1_result['risk_level'],
                top_n=3
            )
        elif county:
            agent2_result = find_by_county(
                county=county,
                risk_level=agent1_result['risk_level']
            )
            # Normalise county fallback structure
            # to match GPS-based structure
            if agent2_result.get('facilities'):
                facilities_with_rank = []
                for i, f in enumerate(
                    agent2_result['facilities'][:3]
                ):
                    f['rank']        = i + 1
                    f['owner']       = f.get('owner', 'Unknown')
                    facilities_with_rank.append(f)
                agent2_result['primary_facility'] = \
                    facilities_with_rank[0]
                agent2_result['top_facilities'] = \
                    facilities_with_rank
                agent2_result['rationale'] = (
                    f"County-based search for {county}. "
                    f"{agent2_result['facilities_found']} "
                    f"oncology-capable facilities found. "
                    f"GPS coordinates unavailable — "
                    f"distances not calculated."
                )
        else:
            agent2_result = {
                'error': 'No location data provided',
                'primary_facility': None,
                'top_facilities': []
            }

        pipeline_output['agent2'] = agent2_result
        primary = agent2_result.get('primary_facility', {})
        if primary:
            print(f"      Result: {primary.get('name', 'Unknown')} "
                  f"({primary.get('distance_km', '?')}km)")
        else:
            print("      Result: No facility found")

    except Exception as e:
        print(f"      ERROR: {e}")
        pipeline_output['agent2'] = {'error': str(e)}

    # ── Step 3: Agent 3 — Referral Recommendation ───────
    print("\n[3/4] Agent 3: Generating Referral Recommendation...")
    try:
        patient_context = {
            'age':             patient_data.get('v012'),
            'rural':           patient_data.get('rural'),
            'education_level': patient_data.get('v106'),
            'hiv_tested':      patient_data.get('hiv_tested'),
        }
        agent3_result = generate_recommendation(
            agent1_result=pipeline_output['agent1'],
            agent2_result=pipeline_output['agent2'],
            patient_data=patient_context
        )
        pipeline_output['agent3'] = agent3_result
        print(f"      Result: Recommendation generated "
              f"(mode: {agent3_result['mode']})")
    except Exception as e:
        print(f"      ERROR: {e}")
        pipeline_output['agent3'] = {'error': str(e)}

    # ── Step 4: Agent 4 — Compliance Prediction ─────────
    print("\n[4/4] Agent 4: Compliance Barrier Prediction...")
    try:
        # Override v483a with actual oncology facility distance
        if pipeline_output.get('agent2', {}).get('primary_facility'):
            oncology_km = pipeline_output['agent2'][
                'primary_facility'
            ].get('distance_km', None)
            if oncology_km and oncology_km != 'N/A':
                try:
                    patient_data['v483a'] = min(
                        round(float(oncology_km) * 2), 200
                    )
                    print(f"      Distance override: {oncology_km}km → "
                        f"{patient_data['v483a']} min")
                except (ValueError, TypeError):
                    print("      Distance override skipped — no GPS data")
            else:
                print("      Distance override skipped — county fallback")

        agent4_result = predict_compliance(patient_data)
        pipeline_output['agent4'] = agent4_result
        print(f"      Result: {agent4_result['barrier_label']} "
            f"({agent4_result['probability']*100:.1f}%)")
    except Exception as e:
        print(f"      ERROR: {e}")
        pipeline_output['agent4'] = {'error': str(e)}

    # ── Step 5: Unified Summary ──────────────────────────
    pipeline_output['summary'] = _build_summary(pipeline_output)

    # ── Step 6: Audit Log ────────────────────────────────
    _log_pipeline(pipeline_output)

    print("\n" + "="*60)
    print("Pipeline Complete")
    print("="*60)

    return pipeline_output

# ── Summary Builder ──────────────────────────────────────
def _build_summary(output: dict) -> dict:
    """Build unified patient case summary"""
    a1 = output.get('agent1', {})
    a2 = output.get('agent2', {})
    a3 = output.get('agent3', {})
    a4 = output.get('agent4', {})

    primary = a2.get('primary_facility', {})

    return {
        'risk_level':        a1.get('risk_label', 'Unknown'),
        'risk_probability':  a1.get('probability', 0),
        'barrier_level':     a4.get('barrier_label', 'Unknown'),
        'barrier_probability': a4.get('probability', 0),
        'recommended_facility': primary.get('name', 'Unknown'),
        'facility_distance_km': primary.get('distance_km', 'Unknown'),
        'recommendation_mode':  a3.get('mode', 'Unknown'),
        'interventions':     a4.get('interventions', []),
        'inconsistency_flag': a1.get('inconsistency_flag', False),
        'top_risk_factors':  a1.get('top_risk_factors', []),
        'top_barriers':      a4.get('top_barriers', []),
    }

# ── Audit Logger ─────────────────────────────────────────
def _log_pipeline(output: dict):
    """Save pipeline run to audit log"""
    try:
        log_dir = Path(__file__).resolve().parent.parent / 'outputs/reports'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'pipeline_audit_log.jsonl'

        log_entry = {
            'timestamp':     output.get('timestamp'),
            'risk_level':    output.get('summary', {}).get('risk_level'),
            'barrier_level': output.get('summary', {}).get('barrier_level'),
            'facility':      output.get('summary', {}).get('recommended_facility'),
            'mode':          output.get('summary', {}).get('recommendation_mode'),
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"      Audit log skipped: {e}")

# ── Test Function ────────────────────────────────────────
if __name__ == "__main__":
    # Complete end-to-end test patient
    test_patient = {
        # Agent 1 features
        'v012':         22,
        'v190':         1,
        'rural':        1,
        'v106':         0,
        'hiv_tested':   0,
        'hiv_positive': 2,
        'v483a':        60,
        # Agent 4 additional
        'v394':         0,
        # Location
        'patient_lat':  -1.2921,
        'patient_lon':  36.8219,
        # Optional
        'self_reported_screened': False
    }

    result = run_pipeline(test_patient)

    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    summary = result['summary']
    print(f"Risk Level:         {summary['risk_level']}")
    print(f"Risk Probability:   {summary['risk_probability']*100:.1f}%")
    print(f"Barrier Level:      {summary['barrier_level']}")
    print(f"Barrier Probability:{summary['barrier_probability']*100:.1f}%")
    print(f"Facility:           {summary['recommended_facility']}")
    print(f"Distance:           {summary['facility_distance_km']} km")
    print(f"Rec Mode:           {summary['recommendation_mode']}")
    print(f"\nInterventions:")
    for iv in summary['interventions']:
        print(f"  [{iv['rank']}] {iv['intervention']}: {iv['action']}")
    print(f"\nTop Risk Factors:")
    for rf in summary['top_risk_factors']:
        print(f"  {rf['feature']}: SHAP={rf['shap_value']:.3f}")