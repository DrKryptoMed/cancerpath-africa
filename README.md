---
title: CancerPath-Africa
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
---

# 🏥 CancerPath-Africa

**An Explainable Multi-Agent AI Pipeline for Cervical Cancer Screening
Outreach Prioritisation and Referral Navigation in Kenya**

*AI in Healthcare Bootcamp 2026 — Capstone Project*

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.36-orange)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Clinical Problem](#clinical-problem)
- [System Architecture](#system-architecture)
- [Agent Descriptions](#agent-descriptions)
- [Explainability Framework](#explainability-framework)
- [Datasets](#datasets)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Stack](#technical-stack)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Overview

CancerPath-Africa is a modular multi-agent orchestrated AI pipeline
designed to support community health workers in Kenya with cervical
cancer screening outreach prioritisation and referral navigation.

The system integrates classical machine learning, LLM-powered
reasoning, geospatial facility mapping, and explainable AI into a
single end-to-end pipeline — deployable in low-resource Sub-Saharan
African healthcare settings.

---

## Clinical Problem

Cervical cancer is the most prevalent cancer among Kenyan women yet
is almost entirely preventable with early screening. Three critical
gaps drive late-stage presentation:

| Gap | Statistic |
|-----|-----------|
| Screening non-uptake | 86% of Kenyan women never screened (DHS 2022) |
| Referral non-completion | 72.8% face significant distance barriers |
| Oncologist shortage | < 1 specialist per million people in SSA |

CancerPath-Africa targets all three gaps through AI-assisted
community health worker support.

---

## System Architecture
Patient / Health Worker Input
↓
┌─────────────────────────────────┐
│  ORCHESTRATOR (orchestrator.py) │
│  Central pipeline controller    │
└─────────────────────────────────┘
↓
┌─────────────────────────────────┐
│  AGENT 1 — Screening Predictor  │
│  Logistic Regression + SHAP     │
│  Threshold: 0.30                │
└─────────────────────────────────┘
↓
Risk Level Output
(HIGH / MODERATE / LOW)
↓
┌─────────────────────────────────┐
│  AGENT 2 — Resource Mapper      │
│  Geospatial facility matching   │
│  County centroid distance calc  │
└─────────────────────────────────┘
↓
Nearest Facilities
↓
┌─────────────────────────────────┐
│  AGENT 3 — Referral Navigator   │
│  LLaMA 3.3-70B via Groq API     │
│  Chain-of-thought prompting     │
│  Offline template fallback      │
└─────────────────────────────────┘
↓
Referral Action Plan
↓
┌─────────────────────────────────┐
│  AGENT 4 — Compliance Predictor │
│  Logistic Regression + SHAP     │
│  Threshold: 0.45                │
│  Counterfactual interventions   │
└─────────────────────────────────┘
↓
┌─────────────────────────────────┐
│  UNIFIED PATIENT SUMMARY        │
│  + Audit Trail Log              │
└─────────────────────────────────┘
---

## Agent Descriptions

### 🔴 Agent 1 — Cervical Cancer Screening Non-Uptake Predictor

Predicts whether a woman is unlikely to have undergone cervical
cancer screening, enabling prioritised community outreach.

- **Model:** Logistic Regression (selected over SVM, XGBoost, RF
  via 12-combination ablation study)
- **Target:** Screening non-uptake (1=never screened, 0=screened)
- **Threshold:** 0.30 (optimised to reduce false negatives by 71%)
- **Performance:** AUC-ROC 0.7573 | F1 0.8960 | Recall 0.9005
- **XAI:** SHAP LinearExplainer — global + individual explanations
- **Top features:** Age, HIV testing history, Education level

### 🏥 Agent 2 — Geospatial Resource Mapper

Identifies nearest oncology-capable health facilities based on
patient location and risk level. Rule-based geospatial module —
not ML.

- **Data:** 331 geocoded oncology-capable facilities from 10,505
  Kenya Health Facilities (Open Africa)
- **Method:** Geodesic distance calculation via geopy
- **Facility tiers:** 4-tier capability classification
- **Fallback:** County centroid estimation when GPS unavailable

### 📋 Agent 3 — LLM Referral Navigator

Generates plain-language referral action plans synthesising
Agent 1 and Agent 2 outputs via chain-of-thought prompting.

- **LLM:** LLaMA 3.3-70B via Groq API (free tier)
- **Ablation:** Tested across zero-shot, few-shot,
  chain-of-thought, and structured output prompting
- **Fallback:** Template-based recommendations for offline use
- **Output:** 5-step structured recommendation (urgency,
  facility, guidance, uncertainties, health worker actions)

### ⚠️ Agent 4 — Referral Compliance Barrier Predictor

Predicts whether a referred patient will face significant barriers
to completing their referral, with personalised intervention
recommendations.

- **Model:** Logistic Regression (12-combination ablation)
- **Target:** Distance barrier to referral compliance
- **Threshold:** 0.45 (all metrics improved simultaneously)
- **Performance:** AUC-ROC 0.7804 | F1 0.8355 | Recall 0.8270
- **XAI:** SHAP + rule-based counterfactual interventions
- **Top feature:** Minutes to facility (SHAP: 0.62) — dominant
  predictor of compliance barriers

---

## Explainability Framework

| Agent | Method | Output |
|-------|--------|--------|
| Agent 1 | SHAP LinearExplainer | Feature importance bar + waterfall |
| Agent 2 | Transparent decision criteria | Facility scoring rationale |
| Agent 3 | Chain-of-thought prompting | Step-by-step reasoning display |
| Agent 4 | SHAP + counterfactuals | Barrier ranking + interventions |

**Key XAI finding:** Unknown HIV status and never having tested
for HIV are the strongest predictors of screening non-uptake —
confirming healthcare system disengagement as the primary barrier
to cervical cancer screening in Kenya.

---

## Datasets

| Dataset | Source | Use |
|---------|--------|-----|
| Kenya DHS 2022 | DHS Program | Agent 1 & 4 training |
| Kenya Health Facilities | Open Africa | Agent 2 facility database |
| WHO GLOBOCAN 2022 | IARC | Cancer burden context |
| Maternal Health Risk | Kaggle | Supplementary features |

---

## Key Findings

1. **86% screening non-uptake** among Kenyan women aged 15-49
   (DHS 2022)
2. **Threshold optimisation** reduced Agent 1 false negatives
   by 71% (996 → 288) at minimal precision cost
3. **Minutes to facility** is the dominant compliance barrier
   predictor (SHAP: 0.62) — 1.7x more influential than wealth
4. **Counterintuitive finding:** Urban women show higher
   screening non-uptake than rural women — warranting further
   investigation
5. **Agent 4 outperforms Agent 1** across all metrics
   (MCC: 0.41 vs 0.23) — compliance prediction is more
   tractable than screening behaviour prediction
6. **Resampling strategies** (SMOTE, ADASYN) showed no
   consistent improvement over class weights — consistent
   across both models

---

## Repository Structure
cancerpath-africa/
│
├── agents/
│   ├── agent1_risk_screener.py      ← Screening non-uptake predictor
│   ├── agent2_resource_mapper.py    ← Geospatial facility matcher
│   ├── agent3_recommendation.py     ← LLM referral navigator
│   ├── agent4_compliance.py         ← Compliance barrier predictor
│   └── orchestrator.py             ← Pipeline controller
│
├── config/
│   ├── config.py                   ← Project configuration
│   └── feature_config.json         ← Feature definitions
│
├── data/
│   ├── raw/                        ← Source datasets (not tracked)
│   ├── processed/                  ← Cleaned model-ready data
│   └── external/                   ← WHO, GLOBOCAN data
│
├── models/
│   ├── saved/                      ← Trained model pkl files
│   └── ablation/                   ← Ablation study results
│
├── notebooks/
│   ├── 01_eda_kenya_facilities.ipynb
│   ├── 02_eda_dhs_kenya.ipynb
│   ├── 03_agent1_ablation.ipynb
│   └── 04_agent4_ablation.ipynb
│
├── outputs/
│   ├── figures/                    ← Publication-ready charts
│   ├── reports/                    ← Audit logs
│   └── mlflow/                     ← MLflow experiment tracking
│
├── utils/
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── explainability.py
│
├── app.py                          ← Gradio entry point
├── requirements.txt
├── README.md
└── .env.example                    ← Environment variable template
---

## Installation

```bash
# Clone repository
git clone https://huggingface.co/spaces/DrKryptoMed/cancerpath-africa
cd cancerpath-africa

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

---

## Usage

### Run Locally

```bash
python app.py
# Open http://localhost:7860
```

### Run Pipeline Programmatically

```python
from agents.orchestrator import run_pipeline

patient = {
    'v012': 22,          # age
    'v190': 1,           # wealth index (1=poorest)
    'rural': 1,          # rural location
    'v106': 0,           # no education
    'hiv_tested': 0,     # never tested for HIV
    'hiv_positive': 2,   # unknown HIV status
    'v394': 0,           # no recent facility visit
    'v483a': 60,         # minutes to facility
    'county': 'Kisumu',  # Kenya county
}

result = run_pipeline(patient)
print(result['summary'])
```

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| ML Models | Scikit-learn (Logistic Regression) |
| Ablation | XGBoost, SVM, Random Forest |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| Explainability | SHAP |
| LLM | LLaMA 3.3-70B (Groq API) |
| Geospatial | geopy |
| Frontend | Gradio |
| Deployment | Hugging Face Spaces |

---

## Limitations

- Agent 1 relies on proxy variables for screening history —
  DHS is a population survey, not a clinical registry
- Agent 3 LLM recommendations not validated by oncologists
  at scale
- Compliance model trained on distance as proxy for
  non-compliance — not actual referral outcome data
- System validated on Kenya data — transferability to other
  SSA countries requires local dataset adaptation
- Agent 2 distances estimated from county centroids when GPS
  unavailable — not exact patient-level distances

---

## Citation

```bibtex
@misc{cancerpath2026,
  title={CancerPath-Africa: An Explainable Multi-Agent AI Pipeline
         for Cervical Cancer Screening Outreach Prioritisation
         and Referral Navigation in Kenya},
  author={Ibrahim A. Mikail},
  year={2026},
  Deployment={Hugging Face Spaces},
  url={https://huggingface.co/spaces/DrKryptoMed/cancerpath-africa}
}
```

---

## Acknowledgements

- AI in Healthcare Bootcamp 2026
- DHS Program (Kenya 2022 survey data)
- Open Africa (Kenya Health Facilities dataset)
- WHO GLOBOCAN 2022

---

*Built for AI in Healthcare Bootcamp 2026 Capstone Project*
*Kenya DHS 2022 | Kenya Health Facilities | WHO GLOBOCAN 2022*
