import numpy as np
import pandas as pd
import os
from config import RANDOM_SEED, SLR_SLOPE, SLR_INTERCEPT, NOISE_LEVEL, MLR_COEFFS, DATA_DIR, FORCE_REGENERATE

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

SLR_DATA_FILE = os.path.join(DATA_DIR, "slr_dataset.csv")
MLR_DATA_FILE = os.path.join(DATA_DIR, "mlr_dataset.csv")


def _save_dataset(X, y, filepath):
    """Save dataset to CSV file."""
    if X.shape[1] == 1:
        df = pd.DataFrame({"X": X.flatten(), "y": y.flatten()})
    else:
        df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
        df["y"] = y.flatten()
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to: {filepath}")


def _load_dataset(filepath):
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    y = df["y"].values.reshape(-1, 1)
    X_cols = [col for col in df.columns if col != "y"]
    X = df[X_cols].values
    print(f"Dataset loaded from: {filepath}")
    return X, y


def generate_simple_data(n, force_regenerate=None):
    """Generates data for Simple Linear Regression.
    
    Args:
        n: Number of samples
        force_regenerate: If True, regenerate even if saved data exists.
                         If None, uses FORCE_REGENERATE from config.
    """
    regenerate = force_regenerate if force_regenerate is not None else FORCE_REGENERATE
    
    # Check if saved dataset exists and should be used
    if os.path.exists(SLR_DATA_FILE) and not regenerate:
        return _load_dataset(SLR_DATA_FILE)
    
    # Generate new data
    np.random.seed(RANDOM_SEED)
    X = 10 * np.random.rand(n, 1) # Random values 0-10
    noise = np.random.normal(0, NOISE_LEVEL, (n, 1))
    # Equation: y = beta_0 + beta_1 * X + epsilon
    y = SLR_INTERCEPT + (SLR_SLOPE * X) + noise
    
    # Save the generated dataset
    _save_dataset(X, y, SLR_DATA_FILE)
    
    return X, y


def generate_multiple_data(n, force_regenerate=None):
    """Generates data for Multiple Linear Regression (2 Features).
    
    Args:
        n: Number of samples
        force_regenerate: If True, regenerate even if saved data exists.
                         If None, uses FORCE_REGENERATE from config.
    """
    regenerate = force_regenerate if force_regenerate is not None else FORCE_REGENERATE
    
    # Check if saved dataset exists and should be used
    if os.path.exists(MLR_DATA_FILE) and not regenerate:
        return _load_dataset(MLR_DATA_FILE)
    
    # Generate new data
    np.random.seed(RANDOM_SEED)
    X = 10 * np.random.rand(n, 2) # 2 columns of features
    noise = np.random.normal(0, NOISE_LEVEL, (n, 1))
    # Equation: y = beta_0 + beta_1*x1 + beta_2*x2 + epsilon
    y = SLR_INTERCEPT + (MLR_COEFFS[0] * X[:, 0:1]) + (MLR_COEFFS[1] * X[:, 1:2]) + noise
    
    # Save the generated dataset
    _save_dataset(X, y, MLR_DATA_FILE)
    
    return X, y