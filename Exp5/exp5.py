# =====================================================
# WEATHER PREDICTION - USING REAL DATASET
# RF + Bagging + Clean Confusion Matrix + PDF Report
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# PDF
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Preformatted
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

# =====================================================
# Output Folder
# =====================================================

output_dir = "weather_experiment_outputs"
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# 1️⃣ Load YOUR Dataset
# =====================================================

data_path = "data/weather_forecast_data.csv"
df = pd.read_csv(data_path)

# Convert Rain column
df["Rain"] = df["Rain"].map({"rain": 1, "no rain": 0})

# Save cleaned dataset copy
df.to_csv(f"{output_dir}/cleaned_weather_dataset.csv", index=False)

# =====================================================
# 2️⃣ Split Data
# =====================================================

X = df.drop("Rain", axis=1)
y = df["Rain"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 3️⃣ Function to Plot CLEAN Confusion Matrix
# =====================================================

def plot_confusion_matrix(cm, model_name, filename):
    plt.figure(figsize=(6,5))
    plt.imshow(cm)

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0,1], ["No Rain", "Rain"])
    plt.yticks([0,1], ["No Rain", "Rain"])

    # Write numbers inside boxes
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j],
                     ha="center",
                     va="center",
                     fontsize=14)

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}")
    plt.close()

# =====================================================
# 4️⃣ RANDOM FOREST
# =====================================================

rf_model = RandomForestClassifier(n_estimators=2, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_report = classification_report(
    y_test,
    rf_pred,
    target_names=["No Rain", "Rain"]
)

rf_cm = confusion_matrix(y_test, rf_pred)

plot_confusion_matrix(rf_cm, "Random Forest", "rf_confusion_matrix.png")

# Feature Importance
plt.figure(figsize=(6,4))
plt.bar(X.columns, rf_model.feature_importances_)
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/rf_feature_importance.png")
plt.close()

# ROC
rf_probs = rf_model.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_rf, tpr_rf)
plt.title("Random Forest ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig(f"{output_dir}/rf_roc_curve.png")
plt.close()

with open(f"{output_dir}/rf_classification_report.txt", "w") as f:
    f.write(f"Accuracy: {rf_accuracy}\n\n")
    f.write(rf_report)

# =====================================================
# 5️⃣ BAGGING
# =====================================================

bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=2,
    bootstrap=True,
    random_state=42
)

bag_model.fit(X_train, y_train)

bag_pred = bag_model.predict(X_test)
bag_accuracy = accuracy_score(y_test, bag_pred)
bag_report = classification_report(
    y_test,
    bag_pred,
    target_names=["No Rain", "Rain"]
)

bag_cm = confusion_matrix(y_test, bag_pred)

plot_confusion_matrix(bag_cm, "Bagging", "bagging_confusion_matrix.png")

# ROC
bag_probs = bag_model.predict_proba(X_test)[:,1]
fpr_bag, tpr_bag, _ = roc_curve(y_test, bag_probs)
roc_auc_bag = auc(fpr_bag, tpr_bag)

plt.figure()
plt.plot(fpr_bag, tpr_bag)
plt.title("Bagging ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig(f"{output_dir}/bagging_roc_curve.png")
plt.close()

with open(f"{output_dir}/bagging_classification_report.txt", "w") as f:
    f.write(f"Accuracy: {bag_accuracy}\n\n")
    f.write(bag_report)

# =====================================================
# 6️⃣ SUMMARY
# =====================================================

with open(f"{output_dir}/summary.txt", "w") as f:
    f.write("WEATHER PREDICTION USING ENSEMBLE METHODS\n")
    f.write("----------------------------------------\n")
    f.write(f"Random Forest Accuracy: {rf_accuracy}\n")
    f.write(f"Random Forest AUC: {roc_auc_rf}\n\n")
    f.write(f"Bagging Accuracy: {bag_accuracy}\n")
    f.write(f"Bagging AUC: {roc_auc_bag}\n")

# =====================================================
# 7️⃣ FINAL PDF
# =====================================================

pdf_path = f"{output_dir}/Final_Report.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
elements = []
styles = getSampleStyleSheet()

elements.append(Paragraph("<b>WEATHER PREDICTION USING ENSEMBLE METHODS</b>", styles["Title"]))
elements.append(Spacer(1, 0.3 * inch))

elements.append(Paragraph("Random Forest Results", styles["Heading2"]))
elements.append(Paragraph(f"Accuracy: {rf_accuracy}", styles["Normal"]))
elements.append(Paragraph(f"AUC Score: {roc_auc_rf}", styles["Normal"]))
elements.append(Spacer(1, 0.2 * inch))

elements.append(Paragraph("Bagging Results", styles["Heading2"]))
elements.append(Paragraph(f"Accuracy: {bag_accuracy}", styles["Normal"]))
elements.append(Paragraph(f"AUC Score: {roc_auc_bag}", styles["Normal"]))
elements.append(Spacer(1, 0.3 * inch))

elements.append(Paragraph("Random Forest Classification Report", styles["Heading3"]))
elements.append(Preformatted(rf_report, styles["Code"]))
elements.append(Spacer(1, 0.3 * inch))

elements.append(Paragraph("Bagging Classification Report", styles["Heading3"]))
elements.append(Preformatted(bag_report, styles["Code"]))
elements.append(Spacer(1, 0.3 * inch))

elements.append(Image(f"{output_dir}/rf_confusion_matrix.png", width=4*inch, height=4*inch))
elements.append(Spacer(1, 0.3 * inch))

elements.append(Image(f"{output_dir}/bagging_confusion_matrix.png", width=4*inch, height=4*inch))

doc.build(elements)

print("✅ All outputs saved in:", output_dir)
print("✅ Final PDF Generated:", pdf_path)