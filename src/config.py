import os

# ====================
# URLs & Endpoints
# ====================
WORLD_AQUATICS_BASE = "https://www.worldaquatics.com"
WORLD_RECORDS_URL = f"{WORLD_AQUATICS_BASE}/swimming/records"
RANKINGS_URL = f"{WORLD_AQUATICS_BASE}/swimming/rankings"

# SwimCloud - Comprehensive meet results
SWIMCLOUD_BASE = "https://www.swimcloud.com"
MEET_URL_TEMPLATE = f"{SWIMCLOUD_BASE}/results/{{meet_id}}"
EVENT_URL_TEMPLATE = f"{SWIMCLOUD_BASE}/results/{{meet_id}}/event/{{event_id}}"

# USA Swimming Times Database
USA_SWIMMING_BASE = "https://www.usaswimming.org"
TIMES_DATABASE_URL = f"{USA_SWIMMING_BASE}/times/event-rank-search"

# Alternative Sources
SWIMSWAM_RESULTS_URL = "https://swimswam.com/meet-results/"  # For meet coverage
OMEGA_TIMING_URL = "https://www.omegatiming.com/"  # For live results

# Backup Sources
FINA_ARCHIVE_URL = "https://archives.fina.org/database"

# ====================
# File Paths
# ====================
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

MEETS_RAW_SUBDIR = os.path.join(RAW_DATA_DIR, "meets")
RECORDS_RAW_SUBDIR = os.path.join(RAW_DATA_DIR, "records")
REGS_RAW_SUBDIR = os.path.join(RAW_DATA_DIR, "registrations")

MEETS_PROC_SUBDIR = os.path.join(PROCESSED_DATA_DIR, "meets")
RECORDS_PROC_SUBDIR = os.path.join(PROCESSED_DATA_DIR, "records")
REGS_PROC_SUBDIR = os.path.join(PROCESSED_DATA_DIR, "registrations")

FEATURES_OUT_PATH = os.path.join(PROCESSED_DATA_DIR, "features.parquet")
MODEL_OUTPUT_DIR = "models"

# ====================
# Scraping Settings
# ====================
USER_AGENT = "SwimRecordBot/1.0 (hglecates@gmail.com)"
REQUEST_PAUSE = 1.0          # seconds between requests
MAX_RETRIES = 3              # retry attempts on failure
TIMEOUT = 10                 # seconds before request timeout

# ====================
# Data Normalization Constants
# ====================
EVENT_NAME_MAP = {
    "100 Free": "100 m Freestyle",
    "100 Metre Freestyle": "100 m Freestyle",
    # Add more mappings as needed
}

TIME_FORMATS = [
    "%M:%S.%f",  # mm:ss.sss
    "%S.%f",     # ss.sss
]

# ====================
# Feature-Engineering Parameters
# ====================
ELITE_THRESHOLD_PCT = 0.98  # swimmers within 98% of record
RECORD_AGE_BUCKETS = [0, 1, 3, 5, 10, 100]
MEET_TIERS = {
    "Olympics": ["TOKYO2020", "PARIS2024"],
    "Worlds": ["Fukuoka2023", "Budapest2017"],
    # Extend with actual meet IDs
}

# ====================
# Modeling Hyperparameters
# ====================
CV_FOLDS = 5
RANDOM_STATE = 42

LR_DEFAULT_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
}

RF_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
}

# ====================
# Logging & Miscellaneous
# ====================
LOG_LEVEL = "INFO"
DATE_FORMAT = "%Y-%m-%d"
TIMEZONE = "UTC"  # or "America/New_York" if needed
