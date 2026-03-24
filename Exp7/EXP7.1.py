# =========================================
# HIGH SILHOUETTE K-MEANS EXPERIMENT
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# =========================================
# OUTPUT DIRECTORY
# =========================================
OUTPUT_DIR = "kmeans_high_score_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================
# 1. GENERATE HIGHLY SEPARABLE DATA 🔥
# =========================================
print("Generating highly separable dataset...")

X, y_true = make_blobs(
    n_samples=1500,
    centers=[[-10, -10], [0, 0], [10, 10], [20, -5]],  # far apart centers
    cluster_std=0.3,   # 🔥 VERY LOW = tight clusters
    random_state=42
)

df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["True_Label"] = y_true

df.to_csv(f"{OUTPUT_DIR}/dataset.csv", index=False)

# =========================================
# 2. ELBOW METHOD
# =========================================
inertia = []
K_range = range(1, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/elbow_plot.png")
plt.close()

# =========================================
# 3. TRAIN FINAL MODEL (TRUE K = 4)
# =========================================
K_OPTIMAL = 4

kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

df["Cluster"] = labels

# =========================================
# 4. METRICS
# =========================================
sil = silhouette_score(X, labels)
db = davies_bouldin_score(X, labels)
inertia_final = kmeans.inertia_

metrics = {
    "K": K_OPTIMAL,
    "Inertia": inertia_final,
    "Silhouette Score": sil,
    "Davies-Bouldin Index": db
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(f"{OUTPUT_DIR}/metrics.csv", index=False)

# =========================================
# 5. VISUALIZATION
# =========================================
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    marker='X'
)
plt.title("High Separation Clusters")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.savefig(f"{OUTPUT_DIR}/clusters.png")
plt.close()

# =========================================
# 6. SAVE HISTORY
# =========================================
history_df = pd.DataFrame({
    "K": list(K_range),
    "Inertia": inertia
})
history_df.to_csv(f"{OUTPUT_DIR}/training_history.csv", index=False)

# =========================================
# 7. PRINT RESULTS
# =========================================
print("\n===== FINAL RESULTS =====")
print(metrics_df)

print("\nAll outputs saved in:", OUTPUT_DIR)