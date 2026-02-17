# =========================
# DATA CONFIGURATION
# =========================

N_SAMPLES = 400
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================
# KNN CONFIGURATION
# =========================

K_RANGE_START = 1
K_RANGE_END = 16

# =========================
# FEATURE CONFIGURATION
# =========================

FEATURE_COLUMNS = ["Age", "Annual_Income", "Spending_Score"]
TARGET_COLUMN = "Purchase"

# =========================
# OUTPUT PATHS
# =========================

DATASET_SAVE_PATH = "outputs/dataset/customer_data.csv"
PLOT_SAVE_PATH = "outputs/plots/accuracy_vs_k.png"
REPORT_SAVE_PATH = "outputs/reports/model_report.txt"
