import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from src.utils import load_config, load_model
import os

def evaluate_model():
    cfg = load_config()
    
    # Load Data and Model
    df = pd.read_csv(cfg['data']['output_path'])
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Re-create split to ensure we test on unseen data
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=cfg['data']['test_size'], random_state=cfg['experiment']['random_seed']
    )
    
    model = load_model(cfg['model']['save_path'])
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualization (Only works well for 2 Features)
    if cfg['data']['n_features'] == 2:
        print("Plotting decision boundary...")
        os.makedirs(os.path.dirname(cfg['visualization']['plot_path']), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot decision boundary
        DecisionBoundaryDisplay.from_estimator(
            model,
            X,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            plot_method="pcolormesh",
            shading="auto",
            alpha=0.6,
            ax=ax
        )
        
        # Plot actual data points
        scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k", s=50)
        ax.set_title(f"SVM Decision Boundary (Kernel: {cfg['model']['kernel']})")
        
        plt.savefig(cfg['visualization']['plot_path'])
        print(f"Plot saved to {cfg['visualization']['plot_path']}")
    else:
        print("Skipping decision boundary plot (n_features != 2).")

if __name__ == "__main__":
    evaluate_model()