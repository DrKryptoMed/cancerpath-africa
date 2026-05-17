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

# DHS files
DHS_INDIVIDUAL        = DATA_RAW / "KEIR8CFL.DTA"
DHS_HOUSEHOLD         = DATA_RAW / "KEHR8CFL.DTA"
DHS_HOUSEHOLD_MEMBER  = DATA_RAW / "KEPR8CFL.DTA"
DHS_GPS_SHAPEFILE     = DATA_RAW / "gps" / "KEGE8AFL.shp"
DHS_GEOSPATIAL_COV    = DATA_RAW / "KEGC8AFL.CSV"

# ─────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

# ─────────────────────────────────────────
# MLFLOW SETTINGS
# ─────────────────────────────────────────
MLFLOW_TRACKING_URI        = "file:///" + str(OUTPUTS_MLFLOW).replace("\\", "/")
MLFLOW_EXPERIMENT_AGENT1   = "Agent1_RiskScreener"
MLFLOW_EXPERIMENT_AGENT4   = "Agent4_CompliancePredictor"