# =========================================
# EXPERIMENT: K-MEANS CLUSTERING PIPELINE
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, davies_bouldin_score

# =========================================
# CREATE OUTPUT DIRECTORY
# =========================================
OUTPUT_DIR = "kmeans_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================
# 1. SYNTHETIC DATA GENERATION
# =========================================
print("Generating synthetic dataset...")

X, y_true = make_blobs(
    n_samples=1000,
    centers=4,
    cluster_std=1.5,
    random_state=42
)

df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["True_Label"] = y_true

df.to_csv(f"{OUTPUT_DIR}/dataset.csv", index=False)

# =========================================
# 2. ELBOW METHOD
# =========================================
print("Running Elbow Method...")

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (WCSS)")
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/elbow_plot.png")
plt.close()

# =========================================
# 3. TRAIN FINAL MODEL
# =========================================
K_OPTIMAL = 4  # Based on elbow (you can adjust)

print(f"Training KMeans with K={K_OPTIMAL}...")

kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

df["Cluster"] = labels

# =========================================
# 4. METRICS
# =========================================
print("Calculating metrics...")

sil_score = silhouette_score(X, labels)
db_score = davies_bouldin_score(X, labels)
inertia_final = kmeans.inertia_

metrics = {
    "K": K_OPTIMAL,
    "Inertia (WCSS)": inertia_final,
    "Silhouette Score": sil_score,
    "Davies-Bouldin Index": db_score
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(f"{OUTPUT_DIR}/metrics.csv", index=False)

# =========================================
# 5. CLUSTER VISUALIZATION
# =========================================
print("Generating plots...")

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    marker='X'
)
plt.title("K-Means Clustering")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.savefig(f"{OUTPUT_DIR}/clusters.png")
plt.close()

# =========================================
# 6. SAVE FINAL DATASET WITH CLUSTERS
# =========================================
df.to_csv(f"{OUTPUT_DIR}/clustered_data.csv", index=False)

# =========================================
# 7. SAVE TRAINING HISTORY (INERTIA PER K)
# =========================================
history_df = pd.DataFrame({
    "K": list(K_range),
    "Inertia": inertia
})

history_df.to_csv(f"{OUTPUT_DIR}/training_history.csv", index=False)

# =========================================
# 8. PRINT SUMMARY
# =========================================
print("\n===== FINAL RESULTS =====")
print(metrics_df)

print("\nAll outputs saved in folder:", OUTPUT_DIR)