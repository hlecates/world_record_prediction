import os

# ====================
# URLs & Endpoints
# ====================
WORLD_AQUATICS_BASE = "https://www.worldaquatics.com"
WORLD_RECORDS_URL = f"{WORLD_AQUATICS_BASE}/swimming/records"

# USA Swimming - Primary data source
USA_SWIMMING_BASE = "https://www.usaswimming.org"

# ====================
# File Paths (Streamlined)
# ====================
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

MEET_PDFS_DIR = os.path.join(RAW_DATA_DIR, "meet_pdfs")
PARSED_RESULTS_DIR = os.path.join(PROCESSED_DATA_DIR, "parsed_results")
FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, "features")
MODEL_OUTPUT_DIR = "models"
REPORTS_DIR = "outputs"

# ====================
# Scraping Settings
# ====================
USER_AGENT = "SwimRecordBot/1.0 (hglecates@gmail.com)"
REQUEST_PAUSE = 1.0          
MAX_RETRIES = 3              
TIMEOUT = 10                 

# ====================
# Data Processing Constants
# ====================
# Event normalization
VALID_STROKES = ['Freestyle', 'Backstroke', 'Breaststroke', 'Butterfly', 'IM']
VALID_DISTANCES = [50, 100, 200, 400, 800, 1500]
VALID_GENDERS = ['Men', 'Women']

# Time format patterns
TIME_FORMATS = [
    r'^\d{1,2}:\d{2}\.\d{2}$',  # MM:SS.ss or M:SS.ss
    r'^\d{1,2}\.\d{2}$'         # SS.ss
]

# ====================
# Feature Engineering Parameters
# ====================
ELITE_THRESHOLD_PCT = 0.98  # swimmers within 98% of record
RECORD_AGE_BUCKETS = [0, 1, 3, 5, 10, 100]  # years since record

# Meet tier classification
MEET_TIER_KEYWORDS = {
    'olympic': ['olympic', 'olympics'],
    'world': ['world', 'international', 'fina'],
    'national': ['national', 'usa', 'nationals'],
    'trials': ['trials', 'trial'],
    'regional': ['regional', 'sectional'],
    'local': ['invitational', 'classic', 'open']
}

# ====================
# ML Model Parameters
# ====================
CV_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Default hyperparameters
LINEAR_PARAMS = {
    "fit_intercept": True,
    "normalize": False
}

RIDGE_PARAMS = {
    "alpha": 1.0,
    "fit_intercept": True
}

RF_REGRESSION_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

RF_CLASSIFICATION_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": RANDOM_STATE
}

LOGISTIC_PARAMS = {
    "random_state": RANDOM_STATE,
    "max_iter": 1000
}

# ====================
# Target Variables
# ====================
REGRESSION_TARGETS = [
    'winner_vs_world_record',    # Time difference vs WR
    'gap_1st_2nd',              # Gap between 1st and 2nd place
    'gap_2nd_3rd',              # Gap between 2nd and 3rd place
    'std_final_time',           # Field competitiveness (spread of times)
    'avg_seed_vs_final'         # Average improvement from seed to final
]

CLASSIFICATION_TARGETS = [
    'top_seed_won',             # Did the top seed win?
]

# ====================
# Feature Sets
# ====================
BASE_FEATURES = [
    'distance',
    'world_record_time',
    'american_record_time',
    'field_size',
    'avg_seed_vs_final',
    'std_seed_vs_final',
    'event_difficulty'
]

CATEGORICAL_FEATURES = [
    'stroke_category',
    'distance_category', 
    'meet_type',
    'gender'
]

MEET_LEVEL_FEATURES = [
    'meet_avg_field_size',
    'meet_avg_competitiveness',
    'meet_avg_gap'
]

TIME_FEATURES = [
    'meet_year',
    'latest_record_year',
    'years_since_record',
    'record_age_avg'
]

# ====================
# Evaluation Metrics
# ====================
REGRESSION_METRICS = ['mse', 'mae', 'r2']
CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1']

# Performance thresholds
MIN_R2_SCORE = 0.3          # Minimum acceptable R² for regression
MIN_ACCURACY = 0.6          # Minimum acceptable accuracy for classification
MIN_SAMPLES_FOR_TRAINING = 50  # Minimum samples needed to train a model

# ====================
# Logging & Output
# ====================
LOG_LEVEL = "DEBUG"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMEZONE = "UTC"

# Output file names
PARSED_DATA_FILE = "all_meets_parsed.csv"
VALIDATED_DATA_FILE = "validated_data.csv"
FEATURES_FILE = "features.csv"
MODEL_METRICS_FILE = "model_metrics.json"
FEATURE_IMPORTANCE_FILE = "feature_importance.json"
EVALUATION_REPORT_FILE = "evaluation_report.md"

# ====================
# Data Quality Parameters
# ====================
# Age validation
MIN_SWIMMER_AGE = 8
MAX_SWIMMER_AGE = 80

# Time validation (in seconds)
MIN_REASONABLE_TIME = {
    50: 15.0,      # 50m events
    100: 35.0,     # 100m events  
    200: 80.0,     # 200m events
    400: 180.0,    # 400m events
    800: 400.0,    # 800m events
    1500: 750.0    # 1500m events
}

MAX_REASONABLE_TIME = {
    50: 120.0,
    100: 300.0,
    200: 600.0,
    400: 1200.0,
    800: 2400.0,
    1500: 4500.0
}

# Record validation
RECORD_TYPES = ['World', 'American', 'U.S. Open', 'Meet']
VALID_DATE_FORMATS = ['%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d']

# ====================
# Development Settings
# ====================
DEBUG_MODE = False
SAMPLE_SIZE = None  # Set to integer to sample data for testing
VERBOSE_LOGGING = True
SAVE_INTERMEDIATE_FILES = True