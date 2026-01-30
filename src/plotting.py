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
    """Plot 3D scatter plot for MLR (Visualizing first 2 features vs Target)."""
    num_features = X.shape[1]
    
    if num_features < 2:
        print("Need at least 2 features for 3D plot. Falling back to 2D.")
        plot_slr_scatter(X, y, model)
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract features and flatten y
    x1 = X[:, 0]
    x2 = X[:, 1]
    y_flat = y.flatten()
    
    # Plot actual data points
    # If 3rd feature exists, use it for color, else use y
    if num_features >= 3:
        c = X[:, 2]
        c_label = "Feature X3"
    else:
        c = y_flat
        c_label = "Target y"
        
    scatter = ax.scatter(x1, x2, y_flat, c=c, cmap='viridis', s=50, alpha=0.8)
    
    # Create a meshgrid for the regression plane
    x1_range = np.linspace(x1.min(), x1.max(), 20)
    x2_range = np.linspace(x2.min(), x2.max(), 20)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    
    # Predict y values for the plane
    # We need to fill other features (X3, etc.) with their mean to visualize the plane of X1, X2
    plane_X = np.zeros((xx1.ravel().shape[0], num_features))
    plane_X[:, 0] = xx1.ravel()
    plane_X[:, 1] = xx2.ravel()
    
    # If there are more features, set them to their mean value for the prediction plane
    for i in range(2, num_features):
        plane_X[:, i] = X[:, i].mean()
        
    y_pred_plane = model.predict(plane_X)
    yy = y_pred_plane.reshape(xx1.shape)
    
    # Plot the regression plane
    ax.plot_surface(xx1, xx2, yy, alpha=0.3, color='orange')
    
    ax.set_xlabel('Feature X1')
    ax.set_ylabel('Feature X2')
    ax.set_zlabel('Target y')
    ax.set_title(f'MLR 3D Visualization\n(Plane at mean of other features)')
    
    # Add colorbar
    plt.colorbar(scatter, label=c_label, pad=0.1)
    
    plt.tight_layout()
    plt.savefig('data/mlr_3d_plot.png', dpi=150)
    plt.show()
    print("MLR 3D plot saved to: data/mlr_3d_plot.png")
