# ==============================
# EXPERIMENT 10: LDA COMPLETE
# ==============================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

# ==============================
# LOAD DATA
# ==============================
digits = load_digits()
X = digits.data
y = digits.target

print("Shape of data:", X.shape)

# ==============================
# SPLIT + SCALE
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# BASELINE MODEL
# ==============================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
acc_before = accuracy_score(y_test, y_pred)

print("\nAccuracy BEFORE LDA:", acc_before)

# ==============================
# CONFUSION MATRIX BEFORE
# ==============================
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Before LDA)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# MANUAL LDA (FIXED)
# ==============================
mean_overall = np.mean(X_train, axis=0)
classes = np.unique(y_train)

# Class means
mean_vectors = {c: np.mean(X_train[y_train == c], axis=0) for c in classes}

# Sw
Sw = np.zeros((X_train.shape[1], X_train.shape[1]))
for c in classes:
    X_c = X_train[y_train == c]
    mean_c = mean_vectors[c].reshape(-1,1)
    for x in X_c:
        x = x.reshape(-1,1)
        Sw += (x - mean_c).dot((x - mean_c).T)

# Sb
Sb = np.zeros_like(Sw)
for c in classes:
    n_c = X_train[y_train == c].shape[0]
    mean_c = mean_vectors[c].reshape(-1,1)
    mean_o = mean_overall.reshape(-1,1)
    Sb += n_c * (mean_c - mean_o).dot((mean_c - mean_o).T)

# Eigen (FIX)
eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))
eigvals = np.real(eigvals)
eigvecs = np.real(eigvecs)

# Sort
eig_pairs = sorted([(eigvals[i], eigvecs[:,i]) for i in range(len(eigvals))],
                   key=lambda x: abs(x[0]), reverse=True)

# Projection
k = len(classes) - 1
W = np.hstack([eig_pairs[i][1].reshape(-1,1) for i in range(k)])

# Transform
X_train_lda = X_train.dot(W)
X_test_lda = X_test.dot(W)

print("\nShape BEFORE:", X_train.shape)
print("Shape AFTER LDA:", X_train_lda.shape)

# ==============================
# MODEL AFTER LDA
# ==============================
knn.fit(X_train_lda, y_train)
y_pred_lda = knn.predict(X_test_lda)

acc_after = accuracy_score(y_test, y_pred_lda)
print("\nAccuracy AFTER LDA:", acc_after)

# ==============================
# CONFUSION MATRIX AFTER
# ==============================
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_lda), annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix (After LDA)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 3D → 2D VISUALIZATION
# ==============================

# STEP 1: Reduce to 3D using PCA (simulate high-dim → 3D)
pca = PCA(n_components=3)
X_3d = pca.fit_transform(X_train)

# STEP 2: Apply LDA to reduce 3D → 2D
lda_2 = LinearDiscriminantAnalysis(n_components=2)
X_2d = lda_2.fit_transform(X_3d, y_train)

# ==============================
# PLOT 3D
# ==============================
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2],
                     c=y_train, cmap='tab10', s=10)

ax.set_title("3D Representation (Before LDA)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.show()

# ==============================
# PLOT 2D AFTER LDA
# ==============================
plt.figure(figsize=(8,6))

for c in classes:
    plt.scatter(X_2d[y_train == c, 0],
                X_2d[y_train == c, 1],
                label=str(c), s=10)

plt.title("2D Representation (After LDA)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.grid()
plt.show()