import matplotlib.pyplot as plt
from data.generate_dataset import generate_data
from utils.preprocessing import preprocess_data
from models.knn import KNN
from utils.metrics import evaluate_model

df = generate_data()

X_train, X_test, y_train, y_test = preprocess_data(df)

k_values = range(1, 16)
accuracies = []

for k in k_values:
    model = KNN(k=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc, cm, precision, recall, f1 = evaluate_model(y_test, y_pred)

    accuracies.append(acc)

    print(f"\nK = {k}")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

# Plot accuracy vs K
plt.plot(k_values, accuracies)
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy")
plt.show()

optimal_k = k_values[accuracies.index(max(accuracies))]
print("\nOptimal K:", optimal_k)
