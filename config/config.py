import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW        = BASE_DIR / "data" / "raw"
DATA_PROCESSED  = BASE_DIR / "data" / "processed"
DATA_EXTERNAL   = BASE_DIR / "data" / "external"
MODELS_SAVED    = BASE_DIR / "models" / "saved"
MODELS_ABLATION = BASE_DIR / "models" / "ablation"
OUTPUTS_FIGURES = BASE_DIR / "outputs" / "figures"
OUTPUTS_REPORTS = BASE_DIR / "outputs" / "reports"
OUTPUTS_MLFLOW  = BASE_DIR / "outputs" / "mlflow"

# ─────────────────────────────────────────
# DATASET FILENAMES
# ─────────────────────────────────────────
KENYA_FACILITIES_FILE   = DATA_RAW / "kenya_health_facilities.xls"
MATERNAL_HEALTH_FILE    = DATA_RAW / "maternal_health_risk_data.csv"
GLOBOCAN_FILE           = DATA_RAW / "cancer_incidence_risk_kenya.csv"

# DHS files — update once approved and downloaded
DHS_INDIVIDUAL   = DATA_RAW / "dhs_kenya_individual.csv"
DHS_HOUSEHOLD    = DATA_RAW / "dhs_kenya_household.csv"
DHS_GPS          = DATA_RAW / "dhs_kenya_gps.csv"