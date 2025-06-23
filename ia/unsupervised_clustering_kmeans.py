import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generar datos sintéticos
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Ajustar KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# Visualización
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centros')
plt.title('Clustering KMeans (no supervisado)')
plt.legend()
plt.show() 