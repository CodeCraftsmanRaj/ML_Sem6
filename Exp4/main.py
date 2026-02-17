from config import *
from data.generate_dataset import generate_data
from utils.preprocessing import preprocess_data
from models.knn import KNN
from utils.metrics import evaluate_model
from utils.saver import create_directories, save_dataset, save_plot, save_report

def main():

    # Create output folders
    create_directories()

    # Generate dataset
    df = generate_data()

    # Save dataset
    save_dataset(df, DATASET_SAVE_PATH)
    print("Dataset saved at:", DATASET_SAVE_PATH)

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df)

    k_values = range(K_RANGE_START, K_RANGE_END)
    accuracies = []

    best_accuracy = 0
    best_k = None
    report_text = "===== KNN Model Evaluation Report =====\n\n"

    for k in k_values:

        model = KNN(k=k)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc, cm, precision, recall, f1 = evaluate_model(y_test, y_pred)

        accuracies.append(acc)

        report_text += f"K = {k}\n"
        report_text += f"Accuracy: {round(acc,4)}\n"
        report_text += f"Confusion Matrix:\n{cm}\n"
        report_text += f"Precision: {round(precision,4)}\n"
        report_text += f"Recall: {round(recall,4)}\n"
        report_text += f"F1 Score: {round(f1,4)}\n"
        report_text += "-"*40 + "\n"

        if acc > best_accuracy:
            best_accuracy = acc
            best_k = k

    # Save plot
    save_plot(k_values, accuracies, PLOT_SAVE_PATH)
    print("Plot saved at:", PLOT_SAVE_PATH)

    # Add best model summary
    report_text += f"\nOptimal K: {best_k}\n"
    report_text += f"Best Accuracy: {round(best_accuracy,4)}\n"

    # Save report
    save_report(report_text, REPORT_SAVE_PATH)
    print("Report saved at:", REPORT_SAVE_PATH)

    print("\nOptimal K:", best_k)
    print("Best Accuracy:", round(best_accuracy,4))


if __name__ == "__main__":
    main()
