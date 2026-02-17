import os
import matplotlib.pyplot as plt

def create_directories():
    os.makedirs("outputs/dataset", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

def save_dataset(df, path):
    df.to_csv(path, index=False)

def save_plot(k_values, accuracies, path):
    plt.figure()
    plt.plot(k_values, accuracies)
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.title("K vs Accuracy")
    plt.savefig(path)
    plt.close()

def save_report(report_text, path):
    with open(path, "w") as f:
        f.write(report_text)
