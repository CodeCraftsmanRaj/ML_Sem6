import numpy as np
import pandas as pd
import os
from config import RANDOM_SEED, SLR_SLOPE, SLR_INTERCEPT, NOISE_LEVEL, MLR_COEFFS, DATA_DIR, FORCE_REGENERATE

os.makedirs(DATA_DIR, exist_ok=True)

SLR_DATA_FILE = os.path.join(DATA_DIR, "slr_dataset.csv")
MLR_DATA_FILE = os.path.join(DATA_DIR, "mlr_dataset.csv")


def _save_dataset(X, y, filepath):
    if X.shape[1] == 1:
        df = pd.DataFrame({"X": X.flatten(), "y": y.flatten()})
    else:
        df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
        df["y"] = y.flatten()
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to: {filepath}")


def _load_dataset(filepath):
    df = pd.read_csv(filepath)
    y = df["y"].values.reshape(-1, 1)
    X_cols = [col for col in df.columns if col != "y"]
    X = df[X_cols].values
    print(f"Dataset loaded from: {filepath}")
    return X, y


def generate_simple_data(n, force_regenerate=None):
    regenerate = force_regenerate if force_regenerate is not None else FORCE_REGENERATE
    
    if os.path.exists(SLR_DATA_FILE) and not regenerate:
        return _load_dataset(SLR_DATA_FILE)
    
    np.random.seed(RANDOM_SEED)
    X = 10 * np.random.rand(n, 1)
    noise = np.random.normal(0, NOISE_LEVEL, (n, 1))
    y = SLR_INTERCEPT + (SLR_SLOPE * X) + noise
    
    _save_dataset(X, y, SLR_DATA_FILE)
    
    return X, y


def generate_multiple_data(n, force_regenerate=None):
    regenerate = force_regenerate if force_regenerate is not None else FORCE_REGENERATE
    
    if os.path.exists(MLR_DATA_FILE) and not regenerate:
        return _load_dataset(MLR_DATA_FILE)
    
    # Number of features is determined by the length of MLR_COEFFS
    num_features = len(MLR_COEFFS)
    
    # Generate new data
    np.random.seed(RANDOM_SEED)
    X = 10 * np.random.rand(n, num_features)  # n rows, num_features columns
    noise = np.random.normal(0, NOISE_LEVEL, (n, 1))
    
    # Equation: y = beta_0 + beta_1*x1 + beta_2*x2 + ... + beta_n*xn + epsilon
    y = SLR_INTERCEPT + noise
    for i, coeff in enumerate(MLR_COEFFS):
        y = y + coeff * X[:, i:i+1]
    
    # Save the generated dataset
    _save_dataset(X, y, MLR_DATA_FILE)
    
    return X, y