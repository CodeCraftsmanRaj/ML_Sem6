# =========================================
# FAST K-MEANS (HIGH SILHOUETTE - OPTIMIZED)
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# =========================================
# OUTPUT DIRECTORY
# =========================================
OUTPUT_DIR = "kmeans_fast_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================
# 1. LOAD DATASET
# =========================================
print("Loading dataset...")

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# =========================================
# 2. SELECT TOP FEATURES (FAST 🔥)
# =========================================
top_features = [
    'mean radius', 'mean texture', 'mean perimeter',
    'mean area', 'mean concave points',
    'worst radius', 'worst perimeter',
    'worst area', 'worst concave points', 'worst texture'
]

df = df[top_features]

# =========================================
# 3. SCALE DATA
# =========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

feature_names = df.columns

# =========================================
# 4. SEARCH BEST FEATURE PAIR + K
# =========================================
print("Searching best feature pair (optimized)...")

best_score = -1
best_features = None
best_k = None
best_X = None

count = 0

for f1, f2 in itertools.combinations(range(len(feature_names)), 2):
    count += 1

    if count % 10 == 0:
        print(f"Processed {count} combinations...")

    X_subset = X_scaled[:, [f1, f2]]

    for k in range(2, 4):  # reduced K range
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_subset)

        score = silhouette_score(X_subset, labels)

        if score > best_score:
            best_score = score
            best_features = (feature_names[f1], feature_names[f2])
            best_k = k
            best_X = X_subset

print("\n===== BEST CONFIGURATION =====")
print("Best Features:", best_features)
print("Best K:", best_k)
print("Best Silhouette Score:", best_score)

# =========================================
# 5. FINAL MODEL
# =========================================
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(best_X)

# =========================================
# 6. METRICS
# =========================================
sil = silhouette_score(best_X, labels)
db = davies_bouldin_score(best_X, labels)
inertia = kmeans.inertia_

metrics = {
    "Features": str(best_features),
    "K": best_k,
    "Inertia": inertia,
    "Silhouette Score": sil,
    "Davies-Bouldin Index": db
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(f"{OUTPUT_DIR}/metrics.csv", index=False)

# =========================================
# 7. CLUSTER PLOT
# =========================================
plt.figure()
plt.scatter(best_X[:, 0], best_X[:, 1], c=labels)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    marker='X'
)
plt.title(f"Best Clustering ({best_features[0]} vs {best_features[1]})")
plt.xlabel(best_features[0])
plt.ylabel(best_features[1])
plt.savefig(f"{OUTPUT_DIR}/clusters.png")
plt.close()

# =========================================
# 8. ELBOW METHOD
# =========================================
inertia_vals = []
K_range = range(1, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(best_X)
    inertia_vals.append(km.inertia_)

plt.figure()
plt.plot(K_range, inertia_vals, marker='o')
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/elbow.png")
plt.close()

# =========================================
# 9. SILHOUETTE VS K
# =========================================
sil_scores = []
K_test = range(2, 8)

for k in K_test:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(best_X)
    sil_scores.append(silhouette_score(best_X, labels_k))

plt.figure()
plt.plot(K_test, sil_scores, marker='o')
plt.title("Silhouette Score vs K")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/silhouette_vs_k.png")
plt.close()

# =========================================
# 10. SAVE DATA
# =========================================
final_df = pd.DataFrame(best_X, columns=best_features)
final_df["Cluster"] = labels
final_df.to_csv(f"{OUTPUT_DIR}/clustered_data.csv", index=False)

history_df = pd.DataFrame({
    "K": list(K_test),
    "Silhouette": sil_scores
})
history_df.to_csv(f"{OUTPUT_DIR}/training_history.csv", index=False)

# =========================================
# 11. PRINT RESULTS
# =========================================
print("\n===== FINAL RESULTS =====")
print(metrics_df)

print("\nAll outputs saved in:", OUTPUT_DIR)