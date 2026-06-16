"""
config.py — Central configuration for ParkSentinel.
All paths, constants, and tunable parameters live here.
Import this in every other module: from src.config import *
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_RAW        = ROOT_DIR / "data" / "raw"
DATA_PROCESSED  = ROOT_DIR / "data" / "processed"
MODELS_DIR      = ROOT_DIR / "models"
LOGS_DIR        = ROOT_DIR / "logs"

# Create dirs if missing
for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_CSV         = Path(os.getenv("DATA_PATH",
                    DATA_RAW / "jan_to_may_police_violation_anonymized.csv"))
CLEAN_PARQUET   = Path(os.getenv("CLEAN_DATA_PATH",
                    DATA_PROCESSED / "violations_clean_day1.parquet"))
H3_PARQUET      = DATA_PROCESSED / "h3_priority_scores.parquet"
ZONE_PARQUET    = DATA_PROCESSED / "zone_profiles.parquet"

# ── API Keys ───────────────────────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")

# ── Bengaluru Bounding Box ─────────────────────────────────────────────────
BLR_LAT_MIN, BLR_LAT_MAX = 12.7, 13.2
BLR_LON_MIN, BLR_LON_MAX = 77.4, 77.9
BLR_CENTER = (12.97, 77.59)

# ── H3 Resolution ─────────────────────────────────────────────────────────
# Resolution 8 ≈ 460m hex diameter — good for enforcement zones
H3_RESOLUTION = 8

# ── Severity Weights (offence_code → congestion impact) ───────────────────
SEVERITY_WEIGHTS = {
    109: 5,   # PARKING OPPOSITE TO ANOTHER PARKED VEHICLE
    108: 5,   # DOUBLE PARKING
    107: 4,   # PARKING IN A MAIN ROAD
    111: 4,   # PARKING NEAR BUSTOP/SCHOOL/HOSPITAL/ETC
    104: 3,   # PARKING NEAR ROAD CROSSING
    112: 2,   # WRONG PARKING
    113: 1,   # NO PARKING
}
DEFAULT_SEVERITY = 1

# ── Vehicle Congestion Weights ─────────────────────────────────────────────
VEHICLE_WEIGHT = {
    "BUS": 5, "TANKER": 5, "TRUCK": 5, "LORRY": 5,
    "MAXI-CAB": 4, "MAXI CAB": 4,
    "PASSENGER AUTO": 3, "AUTO": 3,
    "CAR": 2, "TAXI": 2, "CAB": 2,
    "MOTOR CYCLE": 1, "SCOOTER": 1, "BICYCLE": 1,
}
DEFAULT_VEH_WEIGHT = 2

# ── Road Rank (inferred from address string) ───────────────────────────────
ROAD_RANK = {
    "MAIN ROAD": 4, "OUTER RING ROAD": 4,
    "RING ROAD": 4, "HIGHWAY": 4,
    "CROSS ROAD": 3, "CROSS": 3,
    "ROAD": 2, "AVENUE": 2, "STREET": 2,
    "LAYOUT": 1, "COLONY": 1, "NAGAR": 1,
}
DEFAULT_ROAD_RANK = 1

# ── Priority Score Weights ─────────────────────────────────────────────────
# Must sum to 1.0
PRIORITY_WEIGHTS = {
    "violation_frequency": 0.30,
    "severity_score":      0.25,
    "junction_proximity":  0.20,
    "peak_hour_ratio":     0.15,
    "recurrence_rate":     0.10,
}

# ── Peak Hours ─────────────────────────────────────────────────────────────
MORNING_PEAK = range(7, 11)    # 7 AM – 10 AM
EVENING_PEAK = range(17, 22)   # 5 PM – 9 PM

# ── GenAI Model ───────────────────────────────────────────────────────────
GEMINI_MODEL    = "gemini-1.5-flash"
GEMINI_TEMP     = 0.3          # low = factual, consistent
MAX_TOKENS      = 1024

# ── Dashboard ─────────────────────────────────────────────────────────────
APP_TITLE       = "ParkSentinel — Bengaluru Parking Intelligence"
APP_ICON        = "🚦"
TOP_N_ZONES     = 10           # zones shown in enforcement queue
