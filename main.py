import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from config import DATA_SIZE, TEST_SPLIT
from src.data_generator import generate_simple_data, generate_multiple_data
from src.manual_algo import ManualSimpleLinearRegression, calculate_manual_rmse

def run_simple_regression():
    print("="*50)
    print("1. SIMPLE LINEAR REGRESSION (SLR)")
    print("="*50)

    # 1. Generate Data
    X, y = generate_simple_data(DATA_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42)

    # --- APPROACH A: SKLEARN (The "Code" Way) ---
    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)
    y_pred_train_sk = sk_model.predict(X_train)
    y_pred_test_sk = sk_model.predict(X_test)

    # --- APPROACH B: MANUAL (The "Pen & Paper" Way simulated) ---
    man_model = ManualSimpleLinearRegression()
    man_model.fit(X_train, y_train)
    y_pred_train_man = man_model.predict(X_train)
    y_pred_test_man = man_model.predict(X_test)

    # --- EVALUATION ---
    print(f"Dataset Size: {DATA_SIZE} | Train: {len(X_train)} | Test: {len(X_test)}")
    print("-" * 30)
    print(f"Manual Coefficients:  Intercept={man_model.beta_0:.4f}, Slope={man_model.beta_1:.4f}")
    print(f"Sklearn Coefficients: Intercept={sk_model.intercept_[0]:.4f}, Slope={sk_model.coef_[0][0]:.4f}")
    print("-" * 30)
    
    # RMSE Calculation
    rmse_train_sk = np.sqrt(mean_squared_error(y_train, y_pred_train_sk))
    rmse_test_sk = np.sqrt(mean_squared_error(y_test, y_pred_test_sk))
    
    rmse_train_man = calculate_manual_rmse(y_train, y_pred_train_man)
    rmse_test_man = calculate_manual_rmse(y_test, y_pred_test_man)

    # Accuracy (R2 Score)
    r2_train = r2_score(y_train, y_pred_train_sk)
    r2_test = r2_score(y_test, y_pred_test_sk)

    print(f"Train RMSE (Code):   {rmse_train_sk:.6f} | Manual: {rmse_train_man:.6f}")
    print(f"Test  RMSE (Code):   {rmse_test_sk:.6f} | Manual: {rmse_test_man:.6f}")
    print(f"Train Accuracy (R2): {r2_train:.4f}")
    print(f"Test  Accuracy (R2): {r2_test:.4f}")

def run_multiple_regression():
    print("\n" + "="*50)
    print("2. MULTIPLE LINEAR REGRESSION (MLR)")
    print("="*50)
    
    # 1. Generate Data (2 features)
    X, y = generate_multiple_data(DATA_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42)
    
    # 2. Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 3. Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 4. Evaluate
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept:    {model.intercept_}")
    print(f"Train RMSE:   {rmse_train:.4f} | Test RMSE: {rmse_test:.4f}")
    print(f"Train Acc (R2):{r2_train:.4f} | Test Acc (R2):{r2_test:.4f}")

if __name__ == "__main__":
    run_simple_regression()
    run_multiple_regression()