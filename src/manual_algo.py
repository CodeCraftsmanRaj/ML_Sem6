import numpy as np

class ManualSimpleLinearRegression:
    """
    Implements the math exactly as shown in the provided Slides 8 & 11.
    """
    def __init__(self):
        self.beta_0 = 0
        self.beta_1 = 0

    def fit(self, X, y):
        # Flatten input arrays for easier math
        x_flat = X.flatten()
        y_flat = y.flatten()
        n = len(x_flat)

        # 1. Calculate Statistics (Slide 11)
        sum_x = np.sum(x_flat)
        sum_y = np.sum(y_flat)
        sum_xy = np.sum(x_flat * y_flat)
        sum_x_sq = np.sum(x_flat ** 2)

        # Calculate Means
        x_bar = sum_x / n
        y_bar = sum_y / n

        # 2. Calculate Sxx and Sxy (Slide 8/11)
        # Sxx = Sum(x^2) - (Sum(x)^2 / n)
        Sxx = sum_x_sq - (sum_x**2 / n)
        
        # Sxy = Sum(xy) - (Sum(x)*Sum(y) / n)
        Sxy = sum_xy - ((sum_x * sum_y) / n)

        # 3. Calculate Estimators (Slide 8)
        self.beta_1 = Sxy / Sxx
        self.beta_0 = y_bar - (self.beta_1 * x_bar)

    def predict(self, X):
        return self.beta_0 + (self.beta_1 * X)

def calculate_manual_rmse(y_true, y_pred):
    """
    RMSE = Sqrt( Sum(residual^2) / n )
    Note: Machine Learning uses 'n' in denominator. 
    Statistics (Slide 19) uses 'n-2' for unbiased variance.
    We will use 'n' to match standard ML RMSE.
    """
    n = len(y_true)
    residuals = y_true - y_pred
    mse = np.sum(residuals**2) / n
    return np.sqrt(mse)