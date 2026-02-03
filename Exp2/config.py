import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# Dataset Configuration (Student Pass/Fail Scenario)
DATASET_PATH = os.path.join(DATA_DIR, 'student_data.csv')
N_SAMPLES = 10
FEATURES = ['Study_Hours', 'Prev_Exam_Score', 'Attendance_Pct']
TARGET = 'Pass'
RANDOM_STATE = 42

# Model Configuration
MODEL_PARAMS = {
    'criterion': 'gini',
    'max_depth': 3,
    'random_state': RANDOM_STATE
}

VAL_SIZE = 0.2
TEST_SIZE = 0.2
