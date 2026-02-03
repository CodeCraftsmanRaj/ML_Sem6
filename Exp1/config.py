# config.py
import numpy as np
import os

# Global Settings
RANDOM_SEED = 42
DATA_SIZE = 10
TEST_SPLIT = 0.2

# Dataset Settings
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FORCE_REGENERATE = True # Set to True to regenerate datasets

# Simple Linear Regression Settings (y = b0 + b1*x)
SLR_INTERCEPT = 10  # beta_0
SLR_SLOPE = 2.5     # beta_1
NOISE_LEVEL = 2.0   # Standard deviation of error

# Multiple Linear Regression Settings (y = b0 + b1*x1 + b2*x2 + ... + bn*xn)
# Number of features is determined by the length of MLR_COEFFS
MLR_COEFFS = [3.0, 1.5, 2.0]  # beta_1, beta_2, beta_3, beta_4, beta_5