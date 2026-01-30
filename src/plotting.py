import numpy as np
import matplotlib.pyplot as plt


def plot_slr_scatter(X, y, model):
    """Plot scatter plot for Simple Linear Regression with regression line."""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of actual data
    plt.scatter(X, y, color='blue', alpha=0.6, label='Actual Data')
    
    # Regression line
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression Line')
    
    plt.xlabel('X (Feature)')
    plt.ylabel('y (Target)')
    plt.title('Simple Linear Regression - Scatter Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/slr_scatter_plot.png', dpi=150)
    plt.show()
    print("SLR scatter plot saved to: data/slr_scatter_plot.png")


def plot_mlr_scatter(X, y, model):
    """Plot scatter plots for each feature in Multiple Linear Regression."""
    num_features = X.shape[1]
    
    # Create subplots for each feature
    cols = min(3, num_features)
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    if num_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(num_features):
        ax = axes[i]
        ax.scatter(X[:, i], y, color='blue', alpha=0.6)
        ax.set_xlabel(f'X{i+1}')
        ax.set_ylabel('y')
        ax.set_title(f'Feature X{i+1} vs y')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(num_features, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'Multiple Linear Regression - {num_features} Features', fontsize=14)
    plt.tight_layout()
    plt.savefig('data/mlr_scatter_plot.png', dpi=150)
    plt.show()
    print("MLR scatter plot saved to: data/mlr_scatter_plot.png")
