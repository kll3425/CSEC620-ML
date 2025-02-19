"""
Hyperparameter Tuning
For each algorithm you will need to perform some hyperparameter tuning. The critical
hyperparameters for each algorithm are:
● k-Means
    ○ Number of k clusters to build.
    ○ Distance threshold value t.
● DBSCAN
    ○ The epsilon value used to determine neighbors.
    ○ The minimum number of neighbors to be labeled as a core point.
● The number of dimensions to reduce your features to (if at all) - for both of the clustering
algorithms.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed

# Load data function
def load_npy_data():
    training_data_normal = np.load('./assignment_02/training_normal.npy')
    testing_data_attack = np.load('./assignment_02/testing_attack.npy')
    testing_data_normal = np.load('./assignment_02/testing_normal.npy')
    return training_data_normal, testing_data_attack, testing_data_normal

# Function to apply PCA
def apply_pca(X, pca_dim):
    if pca_dim is not None and pca_dim < X.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=42)
        return pca.fit_transform(X)
    return X

# Optimized k-Means tuning using parallel processing
def evaluate_kmeans(X, k, t, pca_dim):
    X_transformed = apply_pca(X, pca_dim)
    
    if k <= 1:
        return None  # Skip invalid k values
    
    kmeans = KMeans(n_clusters=k, tol=t, random_state=42)
    labels = kmeans.fit_predict(X_transformed)
    
    if len(set(labels)) > 1:
        score = silhouette_score(X_transformed, labels)
        return {'k': k, 't': t, 'pca_dim': pca_dim, 'score': score}
    return None

def tune_kmeans(X, k_values, t_values, pca_dimensions=None):
    if pca_dimensions is None:
        pca_dimensions = [None]

    results = Parallel(n_jobs=-1)(delayed(evaluate_kmeans)(X, k, t, pca_dim)
                                  for pca_dim in pca_dimensions
                                  for k in k_values
                                  for t in t_values)
    
    results = [r for r in results if r is not None]
    return max(results, key=lambda x: x['score'], default={'k': None, 't': None, 'pca_dim': None, 'score': -1})

# Optimized DBSCAN tuning using parallel processing
def evaluate_dbscan(X, eps, min_samples, pca_dim):
    X_transformed = apply_pca(X, pca_dim)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_transformed)
    
    unique_labels = set(labels) - {-1}
    if len(unique_labels) > 1:
        score = silhouette_score(X_transformed, labels)
        return {'eps': eps, 'min_samples': min_samples, 'pca_dim': pca_dim, 'score': score}
    return None

def tune_dbscan(X, eps_values, min_samples_values, pca_dimensions=None):
    if pca_dimensions is None:
        pca_dimensions = [None]

    results = Parallel(n_jobs=-1)(delayed(evaluate_dbscan)(X, eps, min_samples, pca_dim)
                                  for pca_dim in pca_dimensions
                                  for eps in eps_values
                                  for min_samples in min_samples_values)
    
    results = [r for r in results if r is not None]
    return max(results, key=lambda x: x['score'], default={'eps': None, 'min_samples': None, 'pca_dim': None, 'score': -1})

if __name__ == "__main__":
    training_data_normal, _, _ = load_npy_data()
    X = training_data_normal[:1000]  # Use a subset of data for faster tuning

    k_values = [2, 3, 4, 5]
    t_values = [1e-3, 1e-4]
    eps_values = [0.1, 0.3, 0.5]
    min_samples_values = [3, 5]
    pca_dimensions = [None, 2]

    best_kmeans_params = tune_kmeans(X, k_values, t_values, pca_dimensions)
    print(f"Best k-Means Parameters: {best_kmeans_params}")

    best_dbscan_params = tune_dbscan(X, eps_values, min_samples_values, pca_dimensions)
    print(f"Best DBSCAN Parameters: {best_dbscan_params}")