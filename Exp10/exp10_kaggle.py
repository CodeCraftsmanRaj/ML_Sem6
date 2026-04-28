import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("/home/raj_99/Projects/Sem6_Labs/ML/Exp10/Stars.csv")

print("Dataset Head:\n", df.head())

# =========================
# FEATURES & LABELS
# =========================
X = df[['Temperature', 'L', 'R', 'A_M']].values
y = df['Type'].values

print("\nShape of X:", X.shape)

# =========================
# STANDARDIZE
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# CLASSIFICATION WITHOUT LDA
# =========================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("\nAccuracy WITHOUT LDA:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix WITHOUT LDA")
plt.show()

# =========================
# MANUAL LDA IMPLEMENTATION
# =========================

# Mean vectors
mean_vectors = []
for cl in np.unique(y_train):
    mean_vectors.append(np.mean(X_train[y_train == cl], axis=0))

print("\nComputed class-wise mean vectors")

# Overall mean
mean_overall = np.mean(X_train, axis=0)
print("Overall Mean shape:", mean_overall.shape)

# =========================
# WITHIN-CLASS SCATTER (Sw)
# =========================
Sw = np.zeros((X_train.shape[1], X_train.shape[1]))

for cl, mv in zip(np.unique(y_train), mean_vectors):
    class_scatter = np.zeros((X_train.shape[1], X_train.shape[1]))
    for row in X_train[y_train == cl]:
        row, mv = row.reshape(-1,1), mv.reshape(-1,1)
        class_scatter += (row - mv).dot((row - mv).T)
    Sw += class_scatter

print("Sw shape:", Sw.shape)

# =========================
# BETWEEN-CLASS SCATTER (Sb)
# =========================
Sb = np.zeros((X_train.shape[1], X_train.shape[1]))

for i, mean_vec in enumerate(mean_vectors):
    n = X_train[y_train == i+0].shape[0]
    mean_vec = mean_vec.reshape(-1,1)
    mean_overall_vec = mean_overall.reshape(-1,1)
    Sb += n * (mean_vec - mean_overall_vec).dot((mean_vec - mean_overall_vec).T)

print("Sb shape:", Sb.shape)

# =========================
# EIGEN VALUES & VECTORS
# =========================
# FIX: use pseudo-inverse (avoids singular matrix)
eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))

# Sort eigenvalues
eig_pairs = [(np.abs(eigvals[i]), eigvecs[:, i]) for i in range(len(eigvals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# =========================
# SELECT TOP COMPONENTS (3D)
# =========================
W = np.hstack((eig_pairs[0][1].reshape(-1,1),
               eig_pairs[1][1].reshape(-1,1),
               eig_pairs[2][1].reshape(-1,1)))

# Transform data
X_train_lda = X_train.dot(W)
X_test_lda = X_test.dot(W)

print("\nReduced shape (3D):", X_train_lda.shape)

# =========================
# CLASSIFICATION AFTER LDA
# =========================
knn.fit(X_train_lda, y_train)
y_pred_lda = knn.predict(X_test_lda)

print("\nAccuracy WITH LDA:", accuracy_score(y_test, y_pred_lda))

cm = confusion_matrix(y_test, y_pred_lda)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix WITH LDA")
plt.show()

# =========================
# VISUALIZATION
# =========================

# 3D Plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for label in np.unique(y_train):
    ax.scatter(
        X_train_lda[y_train == label, 0],
        X_train_lda[y_train == label, 1],
        X_train_lda[y_train == label, 2],
        label=f"Class {label}"
    )

ax.set_title("3D LDA Projection")
ax.legend()
plt.show()

# =========================
# 2D REDUCTION (PCA)
# =========================
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train_lda)

plt.figure()
for label in np.unique(y_train):
    plt.scatter(
        X_train_2d[y_train == label, 0],
        X_train_2d[y_train == label, 1],
        label=f"Class {label}"
    )

plt.title("2D Projection (After LDA → PCA)")
plt.legend()
plt.show()