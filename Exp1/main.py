import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from config import DATA_SIZE, TEST_SPLIT, MLR_COEFFS
from src.data_generator import generate_simple_data, generate_multiple_data
from src.plotting import plot_slr_scatter, plot_mlr_scatter

def run_simple_regression():
    print("="*50)
    print("1. SIMPLE LINEAR REGRESSION (SLR)")
    print("="*50)

    # 1. Generate Data
    X, y = generate_simple_data(DATA_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42)

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluation
    print(f"Dataset Size: {DATA_SIZE} | Train: {len(X_train)} | Test: {len(X_test)}")
    print("-" * 30)
    print(f"Coefficients: Intercept={model.intercept_[0]:.4f}, Slope={model.coef_[0][0]:.4f}")
    print("-" * 30)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"Train RMSE: {rmse_train:.4f} | Test RMSE: {rmse_test:.4f}")
    print(f"Train Accuracy (R2): {r2_train:.4f}")
    print(f"Test  Accuracy (R2): {r2_test:.4f}")
    
    # Plot scatter plot with regression line
    plot_slr_scatter(X, y, model)

def run_multiple_regression():
    print("\n" + "="*50)
    print(f"2. MULTIPLE LINEAR REGRESSION (MLR) - {len(MLR_COEFFS)} Features")
    print("="*50)
    
    # 1. Generate Data (n features based on MLR_COEFFS length)
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
    
    # Plot scatter plots for each feature
    plot_mlr_scatter(X, y, model)


if __name__ == "__main__":
    run_simple_regression()
    run_multiple_regression()