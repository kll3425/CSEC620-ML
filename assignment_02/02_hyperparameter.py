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
from joblib import Parallel, delayed  # for parallel processing

# load data from .npy files
def load_npy_data():
    """
    Loads training and testing data from numpy (.npy) files.
    
    Returns:
    - training_data_normal: Normal training dataset
    - testing_data_attack: Attack dataset for testing
    - testing_data_normal: Normal dataset for testing
    """
    training_data_normal = np.load('./assignment_02/training_normal.npy')
    testing_data_attack = np.load('./assignment_02/testing_attack.npy')
    testing_data_normal = np.load('./assignment_02/testing_normal.npy')
    return training_data_normal, testing_data_attack, testing_data_normal

# apply PCA dimensionality reduction
def apply_pca(data, pca_dim):
    """
    Applies PCA to reduce feature dimensions if specified.

    Parameters:
    - data: Input data (numpy array)
    - pca_dim: Number of principal components to reduce to (None means no reduction)

    Returns:
    - Transformed data with reduced dimensions if PCA is applied, otherwise original data.
    """
    if pca_dim is not None and pca_dim < data.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=42)
        return pca.fit_transform(data)
    return data  # return original data if PCA is not applied

# evaluate a single k-Means configuration
def evaluate_kmeans(data, k, t, pca_dim):
    """
    Evaluates k-Means clustering for a specific combination of hyperparameters.

    Parameters:
    - data: Input data
    - k: Number of clusters
    - t: Tolerance value for convergence
    - pca_dim: Number of dimensions to reduce to using PCA (None means no PCA)

    Returns:
    - Dictionary containing the parameters and silhouette score, or None if invalid.
    """
    X_transformed = apply_pca(data, pca_dim)  # apply PCA if necessary

    if k <= 1:
        return None  # skip invalid cluster counts

    # initialize k-Means with specified hyperparameters
    kmeans = KMeans(n_clusters=k, tol=t, random_state=42)
    labels = kmeans.fit_predict(X_transformed)  # fit and get cluster labels

    # compute silhouette score only if multiple clusters exist
    if len(set(labels)) > 1:
        score = silhouette_score(X_transformed, labels)
        return {'k': k, 't': t, 'pca_dim': pca_dim, 'score': score}
    
    return None  # return None if clustering isn't meaningful

# function to tune k-Means hyperparameters in parallel
def tune_kmeans(data, k_values, t_values, pca_dimensions=None):
    """
    Tunes k-Means hyperparameters using parallel processing.

    Parameters:
    - data: Input data
    - k_values: List of k values to test
    - t_values: List of tolerance values to test
    - pca_dimensions: List of PCA dimensions to test (None means no PCA)

    Returns:
    - Best hyperparameter combination based on silhouette score.

    AI Usage:
        Chat GPT Prompt:
            Optimize the following code using parallel processing:
            def tune_kmeans(X, k_values, t_values, pca_dimensions):
                best_params = {'k': None, 't': None, 'pca_dim': None, 'score': -1}
                
                for pca_dim in pca_dimensions:
                    X_transformed = X if pca_dim is None else PCA(n_components=pca_dim).fit_transform(X)
                    
                    for k in k_values:
                        for t in t_values:
                            kmeans = KMeans(n_clusters=k, tol=t, random_state=42)
                            labels = kmeans.fit_predict(X_transformed)
                            score = silhouette_score(X_transformed, labels)
                            
                            if score > best_params['score']:
                                best_params = {'k': k, 't': t, 'pca_dim': pca_dim, 'score': score}
                
                return best_params
    """
    if pca_dimensions is None:
        pca_dimensions = [None]

    # use parallel processing to evaluate multiple configurations
    results = Parallel(n_jobs=-1)(delayed(evaluate_kmeans)(data, k, t, pca_dim)
                                  for pca_dim in pca_dimensions
                                  for k in k_values
                                  for t in t_values)
    
    # remove invalid cases and return the best configuration
    results = [r for r in results if r is not None]
    return max(results, key=lambda x: x['score'], default={'k': None, 't': None, 'pca_dim': None, 'score': -1})

# evaluate a single DBSCAN configuration
def evaluate_dbscan(data, eps, min_samples, pca_dim):
    """
    Evaluates DBSCAN clustering for a specific combination of hyperparameters.

    Parameters:
    - data: Input data
    - eps: Epsilon value (radius of neighborhood)
    - min_samples: Minimum number of neighbors to be a core point
    - pca_dim: Number of dimensions to reduce to using PCA

    Returns:
    - Dictionary containing the parameters and silhouette score, or None if clustering fails.
    """
    X_transformed = apply_pca(data, pca_dim)  # apply PCA if necessary

    # init DBSCAN with specified hyperparameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_transformed)

    # ensure at least two distinct clusters, ignoring noise (-1)
    unique_labels = set(labels) - {-1}
    if len(unique_labels) > 1:
        score = silhouette_score(X_transformed, labels)
        return {'eps': eps, 'min_samples': min_samples, 'pca_dim': pca_dim, 'score': score}
    
    return None  # return None if clustering isn't meaningful

# function to tune DBSCAN hyperparameters in parallel
def tune_dbscan(data, eps_values, min_samples_values, pca_dimensions=None):
    """
    Tunes DBSCAN hyperparameters using parallel processing.

    Parameters:
    - data: Input data
    - eps_values: List of epsilon values to test
    - min_samples_values: List of min_samples values to test
    - pca_dimensions: List of PCA dimensions to test (None means no PCA)

    Returns:
    - Best hyperparameter combination based on silhouette score.

    AI Usage:
        Chat GPT Prompt:
            Optimize the following code using parallel processing:
            def tune_dbscan(X, eps_values, min_samples_values, pca_dimensions):
                best_params = {'eps': None, 'min_samples': None, 'pca_dim': None, 'score': -1}

                for pca_dim in pca_dimensions:
                    X_transformed = X if pca_dim is None else PCA(n_components=pca_dim).fit_transform(X)
                    
                    for eps in eps_values:
                        for min_samples in min_samples_values:
                            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                            labels = dbscan.fit_predict(X_transformed)
                            
                            if len(set(labels)) > 1:
                                score = silhouette_score(X_transformed, labels)
                                
                                if score > best_params['score']:
                                    best_params = {'eps': eps, 'min_samples': min_samples, 'pca_dim': pca_dim, 'score': score}
                
                return best_params
    """
    if pca_dimensions is None:
        pca_dimensions = [None]  # ensure it's iterable

    # use parallel processing to evaluate multiple configurations
    results = Parallel(n_jobs=-1)(delayed(evaluate_dbscan)(data, eps, min_samples, pca_dim)
                                  for pca_dim in pca_dimensions
                                  for eps in eps_values
                                  for min_samples in min_samples_values)
    
    # remove invalid cases and return the best configuration
    results = [r for r in results if r is not None]
    return max(results, key=lambda x: x['score'], default={'eps': None, 'min_samples': None, 'pca_dim': None, 'score': -1})

if __name__ == "__main__":
    training_data_normal, testing_data_attack, testing_data_normal = load_npy_data()
    data = training_data_normal[:1000] # smaller subset for memory allocation error and faster tuning

    # hyperparameter ranges
    k_values = [2, 3, 4, 5]  #pPossible k values for k-Means
    t_values = [1e-3, 1e-4]  # tolerance values for k-Means
    eps_values = [0.1, 0.3, 0.5]  # epsilon values for DBSCAN
    min_samples_values = [3, 5]  # minimum number of neighbors for DBSCAN
    pca_dimensions = [None, 2]  # PCA dimensions to test

    # hyperparameter tuning for k-Means
    best_kmeans_params = tune_kmeans(data, k_values, t_values, pca_dimensions)
    print("\n Best k-Means Parameters")
    print(f"  - Optimal Clusters (k): {best_kmeans_params['k']}")
    print(f"  - Convergence Tolerance (t): {best_kmeans_params['t']:.6f}")
    print(f"  - PCA Dimensions: {best_kmeans_params['pca_dim']}")
    print(f"  - Silhouette Score: {best_kmeans_params['score']:.4f}")

    # hyperparameter tuning for DBSCAN
    best_dbscan_params = tune_dbscan(data, eps_values, min_samples_values, pca_dimensions)
    print("\n Best DBSCAN Parameters")
    print(f"  - Epsilon (eps): {best_dbscan_params['eps']}")
    print(f"  - Min Samples: {best_dbscan_params['min_samples']}")
    print(f"  - PCA Dimensions: {best_dbscan_params['pca_dim']}")
    print(f"  - Silhouette Score: {best_dbscan_params['score']:.4f}\n")