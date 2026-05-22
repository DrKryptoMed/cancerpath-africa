"""
CancerPath-Africa
Multi-Agent Cervical Cancer Triage System

Entry point for Hugging Face Spaces deployment.
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)
sys.path.append(str(Path(__file__).resolve().parent))

from agents.orchestrator import run_pipeline

# ── Kenya Counties ───────────────────────────────────────
KENYA_COUNTIES = [
    "Baringo", "Bomet", "Bungoma", "Busia", "Elgeyo Marakwet",
    "Embu", "Garissa", "Homa Bay", "Isiolo", "Kajiado",
    "Kakamega", "Kericho", "Kiambu", "Kilifi", "Kirinyaga",
    "Kisii", "Kisumu", "Kitui", "Kwale", "Laikipia",
    "Lamu", "Machakos", "Makueni", "Mandera", "Marsabit",
    "Meru", "Migori", "Mombasa", "Murang'a", "Nairobi",
    "Nakuru", "Nandi", "Narok", "Nyamira", "Nyandarua",
    "Nyeri", "Samburu", "Siaya", "Taita Taveta", "Tana River",
    "Tharaka Nithi", "Trans Nzoia", "Turkana", "Uasin Gishu",
    "Vihiga", "Wajir", "West Pokot"
]

# ── SHAP Chart Builder ───────────────────────────────────
def build_shap_chart(feature_names, shap_values,
                     title, color='#2E75B6'):
    """Build interactive Plotly SHAP bar chart"""
    df = pd.DataFrame({
        'Feature':    feature_names,
        'SHAP Value': shap_values
    }).sort_values('SHAP Value', ascending=True)

    colors = [
        '#E74C3C' if v > 0 else '#2E75B6'
        for v in df['SHAP Value']
    ]

    fig = go.Figure(go.Bar(
        x=df['SHAP Value'],
        y=df['Feature'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in df['SHAP Value']],
        textposition='outside'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title='SHAP Value (impact on prediction)',
        yaxis_title='',
        height=350,
        margin=dict(l=20, r=80, t=50, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        xaxis=dict(zeroline=True, zerolinecolor='black',
                   zerolinewidth=1)
    )
    return fig

# ── Pipeline Runner ──────────────────────────────────────
def run_cancerpath(
    age, wealth_index, location, education,
    hiv_tested, hiv_status, facility_visit,
    self_reported, county
):
    """Main pipeline function called by Gradio"""

    # Map inputs to model features
    location_map   = {'Urban': 0, 'Rural': 1}
    edu_map        = {
        'No Education': 0, 'Primary': 1,
        'Secondary': 2,   'Higher': 3
    }
    hiv_tested_map = {'Yes': 1, 'No': 0}
    hiv_status_map = {
        'Negative': 0, 'Positive': 1, 'Unknown': 2
    }
    visit_map      = {'Yes': 1, 'No': 0}
    screened_map   = {'Yes': True, 'No': False}

    patient_data = {
        'v012':         int(age),
        'v190':         int(wealth_index),
        'rural':        location_map[location],
        'v106':         edu_map[education],
        'hiv_tested':   hiv_tested_map[hiv_tested],
        'hiv_positive': hiv_status_map[hiv_status],
        'v394':         visit_map[facility_visit],
        'v483a':        30,  # placeholder — overridden by orchestrator
        'county':       county,
        'self_reported_screened': screened_map[self_reported]
    }

    # Run pipeline
    try:
        result = run_pipeline(patient_data)
    except Exception as e:
        error_msg = f"Pipeline error: {str(e)}"
        empty_fig = go.Figure()
        return (error_msg,) * 2 + (empty_fig,) * 2 + (error_msg,) * 3

    a1 = result.get('agent1', {})
    a2 = result.get('agent2', {})
    a3 = result.get('agent3', {})
    a4 = result.get('agent4', {})
    summary = result.get('summary', {})

    # ── Tab 1: Risk Assessment ───────────────────────────
    risk_emoji = {'HIGH': '🔴', 'MODERATE': '🟡', 'LOW': '🟢'}
    risk_level = a1.get('risk_level', 'UNKNOWN')
    risk_text  = (
        f"## {risk_emoji.get(risk_level, '⚪')} "
        f"{a1.get('risk_label', 'Unknown')}\n\n"
        f"**Probability:** {a1.get('probability', 0)*100:.1f}%  \n"
        f"**Threshold used:** {a1.get('threshold_used', 0.30)}\n\n"
        f"---\n\n"
        f"{a1.get('plain_summary', '')}"
    )

    if a1.get('inconsistency_flag'):
        risk_text += (
            f"\n\n⚠️ **INCONSISTENCY FLAG:**  \n"
            f"{a1.get('inconsistency_msg', '')}"
        )

    # SHAP chart for Agent 1
    shap_fig1 = build_shap_chart(
        feature_names=a1.get('feature_names', []),
        shap_values=a1.get('shap_values', []),
        title='Feature Impact on Screening Non-Uptake',
        color='#2E75B6'
    )

    # ── Tab 2: Facility Map ──────────────────────────────
    facilities = a2.get('top_facilities', [])
    rationale  = a2.get('rationale', '')

    facility_text = f"**Decision Rationale:**  \n{rationale}\n\n---\n\n"
    for f in facilities:
        facility_text += (
            f"**[{f.get('rank', '?')}] {f.get('name', 'Unknown')}**  \n"
            f"Type: {f.get('type', 'Unknown')}  \n"
            f"County: {f.get('county', 'Unknown')}  \n"
            f"Distance: {f.get('distance_km', '?')} km  \n"
            f"Owner: {f.get('owner', 'Unknown')}  \n\n"
        )

    if not facilities:
        facility_text += "No facilities found. Try a different county."

    # ── Tab 3: Recommendation ────────────────────────────
    rec_text = (
        f"**Mode:** {a3.get('mode', 'Unknown')}  \n\n"
        f"---\n\n"
        f"{a3.get('recommendation', 'No recommendation generated.')}"
    )

    # ── Tab 4: Compliance ────────────────────────────────
    barrier_emoji = {
        'HIGH_BARRIER':     '🔴',
        'MODERATE_BARRIER': '🟡',
        'LOW_BARRIER':      '🟢'
    }
    barrier_level = a4.get('barrier_level', 'UNKNOWN')
    compliance_text = (
        f"## {barrier_emoji.get(barrier_level, '⚪')} "
        f"{a4.get('barrier_label', 'Unknown')}\n\n"
        f"**Probability:** {a4.get('probability', 0)*100:.1f}%  \n"
        f"**Threshold used:** {a4.get('threshold_used', 0.45)}\n\n"
        f"---\n\n"
        f"{a4.get('plain_summary', '')}"
    )

    interventions = a4.get('interventions', [])
    if interventions:
        compliance_text += "\n\n---\n\n**Recommended Interventions:**\n\n"
        for iv in interventions:
            compliance_text += (
                f"**[{iv['rank']}] {iv['intervention']}**  \n"
                f"{iv['action']}  \n"
                f"Feasibility: {iv['feasibility']}  \n\n"
            )

    # SHAP chart for Agent 4
    shap_fig4 = build_shap_chart(
        feature_names=a4.get('feature_names', []),
        shap_values=a4.get('shap_values', []),
        title='Feature Impact on Compliance Barrier',
        color='#E74C3C'
    )

    # ── Tab 5: Full Report ───────────────────────────────
    report_text = f"""## CancerPath-Africa — Patient Case Summary

---

### Risk Assessment
- **Risk Level:** {summary.get('risk_level', 'Unknown')}
- **Probability:** {summary.get('risk_probability', 0)*100:.1f}%
- **Top Risk Factors:**
"""
    for rf in summary.get('top_risk_factors', []):
        report_text += (
            f"  - {rf['feature']} "
            f"(SHAP: {rf['shap_value']:+.3f})\n"
        )

    report_text += f"""
---

### Facility Recommendation
- **Recommended Facility:** {summary.get('recommended_facility', 'Unknown')}
- **Distance:** {summary.get('facility_distance_km', 'Unknown')} km

---

### Compliance Prediction
- **Barrier Level:** {summary.get('barrier_level', 'Unknown')}
- **Probability:** {summary.get('barrier_probability', 0)*100:.1f}%
- **Interventions Required:** {len(summary.get('interventions', []))}

---

### System Information
- **Recommendation Mode:** {summary.get('recommendation_mode', 'Unknown')}
- **Inconsistency Flag:** {'⚠️ Yes' if summary.get('inconsistency_flag') else '✅ No'}
"""

    return (
        risk_text, shap_fig1,
        facility_text,
        rec_text,
        compliance_text, shap_fig4,
        report_text
    )

# ── Gradio Interface ─────────────────────────────────────
def build_interface():
    with gr.Blocks(
        title="CancerPath-Africa",
        theme=gr.themes.Soft()
    ) as demo:

        # Header
        gr.Markdown("""
# 🏥 CancerPath-Africa
### Cervical Cancer Triage & Referral Intelligence System — Kenya
*An Explainable Multi-Agent AI Framework for Community Health Workers*
---
""")

        with gr.Row():
            # ── Left Panel: Patient Input ────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 👤 Patient Information")

                age = gr.Slider(
                    minimum=15, maximum=49, value=25, step=1,
                    label="Age"
                )
                wealth_index = gr.Dropdown(
                    choices=[1, 2, 3, 4, 5], value=2,
                    label="Wealth Index (1=Poorest, 5=Richest)"
                )
                location = gr.Radio(
                    choices=["Urban", "Rural"], value="Rural",
                    label="Location"
                )
                education = gr.Dropdown(
                    choices=[
                        "No Education", "Primary",
                        "Secondary", "Higher"
                    ],
                    value="Primary",
                    label="Education Level"
                )
                hiv_tested = gr.Radio(
                    choices=["Yes", "No"], value="No",
                    label="Ever Tested for HIV"
                )
                hiv_status = gr.Dropdown(
                    choices=["Negative", "Positive", "Unknown"],
                    value="Unknown",
                    label="HIV Status"
                )
                facility_visit = gr.Radio(
                    choices=["Yes", "No"], value="No",
                    label="Visited Facility Last 12 Months"
                )
                self_reported = gr.Radio(
                    choices=["Yes", "No"], value="No",
                    label="Self-Reported Previously Screened"
                )

                gr.Markdown("### 📍 Location")
                county = gr.Dropdown(
                    choices=KENYA_COUNTIES,
                    value="Nairobi",
                    label="County"
                )

                run_btn = gr.Button(
                    "🔍 Run Pipeline",
                    variant="primary",
                    size="lg"
                )

            # ── Right Panel: Results Tabs ─────────────────
            with gr.Column(scale=2):
                with gr.Tabs():

                    # Tab 1 — Risk Assessment
                    with gr.Tab("🔴 Risk Assessment"):
                        risk_output = gr.Markdown()
                        shap_plot1  = gr.Plot(
                            label="SHAP Feature Importance"
                        )

                    # Tab 2 — Facility Map
                    with gr.Tab("🏥 Nearest Facilities"):
                        facility_output = gr.Markdown()

                    # Tab 3 — Recommendation
                    with gr.Tab("📋 Referral Plan"):
                        rec_output = gr.Markdown()

                    # Tab 4 — Compliance
                    with gr.Tab("⚠️ Compliance Prediction"):
                        compliance_output = gr.Markdown()
                        shap_plot4 = gr.Plot(
                            label="SHAP Barrier Importance"
                        )

                    # Tab 5 — Full Report
                    with gr.Tab("📄 Full Report"):
                        report_output = gr.Markdown()

        # Wire button to pipeline
        run_btn.click(
            fn=run_cancerpath,
            inputs=[
                age, wealth_index, location, education,
                hiv_tested, hiv_status, facility_visit,
                self_reported, county
            ],
            outputs=[
                risk_output, shap_plot1,
                facility_output,
                rec_output,
                compliance_output, shap_plot4,
                report_output
            ]
        )

        # Footer
        gr.Markdown("""
---
*CancerPath-Africa — AI in Healthcare Bootcamp 2026 Capstone Project*  
*Kenya DHS 2022 | Kenya Health Facilities (Open Africa) | WHO GLOBOCAN 2022*
""")

    return demo

# ── Launch ───────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )