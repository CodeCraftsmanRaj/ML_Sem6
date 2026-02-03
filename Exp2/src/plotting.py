from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import os
import config

def save_tree_plot(clf, feature_names):
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=['Fail', 'Pass'])
    plt.title("Decision Tree Visualization")
    
    save_path = os.path.join(config.PLOTS_DIR, 'decision_tree.png')
    plt.savefig(save_path)
    print(f"\nTree plot saved to: {save_path}")
    plt.close()
