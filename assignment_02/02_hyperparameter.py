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

# Load the data from the .npy files (path may need to be changed)
def load_npy_data():
    training_data_normal = np.load('./assignment_02/training_normal.npy')
    testing_data_attack = np.load('./assignment_02/testing_attack.npy')
    testing_data_normal = np.load('./assignment_02/testing_normal.npy')
    return training_data_normal, testing_data_attack, testing_data_normal

# Function to tune k-Means hyperparameters
def tune_kmeans(X, k_values, t_values, pca_dimensions=None):
    """
    Tunes k-Means clustering by testing different values of k (clusters),
    tolerance (t), and optional PCA dimensionality reduction.
    
    Parameters:
    - X: numpy array, input data.
    - k_values: list of int, values of k to test.
    - t_values: list of float, tolerance values for convergence.
    - pca_dimensions: list of int or None, number of PCA components to reduce data to.
    
    Returns:
    - best_params: dict, optimal values for k, t, and pca_dim based on silhouette score.
    """
    best_params = {'k': None, 't': None, 'pca_dim': None, 'score': -1}  # Store best parameters
    
    for pca_dim in pca_dimensions:
        X_transformed = X  # Start with original data
        
        # Apply PCA if required
        if pca_dim is not None and pca_dim < X.shape[1]:
            pca = PCA(n_components=pca_dim)
            X_transformed = pca.fit_transform(X)
        
        # Iterate over different values of k and tolerance
        for k in k_values:
            for t in t_values:
                kmeans = KMeans(n_clusters=k, tol=t, random_state=42)  # Initialize k-Means
                labels = kmeans.fit_predict(X_transformed)  # Fit and predict cluster labels
                score = silhouette_score(X_transformed, labels)  # Compute silhouette score
                
                # Update best parameters if current configuration is better
                if score > best_params['score']:
                    best_params.update({'k': k, 't': t, 'pca_dim': pca_dim, 'score': score})
    
    return best_params

# Function to tune DBSCAN hyperparameters
def tune_dbscan(X, eps_values, min_samples_values, pca_dimensions=None):
    """
    Tunes DBSCAN clustering by testing different values of epsilon (eps),
    minimum samples (min_samples), and optional PCA dimensionality reduction.
    
    Parameters:
    - X: numpy array, input data.
    - eps_values: list of float, epsilon values to test.
    - min_samples_values: list of int, min_samples values to test.
    - pca_dimensions: list of int or None, number of PCA components to reduce data to.
    
    Returns:
    - best_params: dict, optimal values for eps, min_samples, and pca_dim based on silhouette score.
    """
    best_params = {'eps': None, 'min_samples': None, 'pca_dim': None, 'score': -1}  # Store best parameters
    
    for pca_dim in pca_dimensions:
        X_transformed = X  # Start with original data
        
        # Apply PCA if required
        if pca_dim is not None and pca_dim < X.shape[1]:
            pca = PCA(n_components=pca_dim)
            X_transformed = pca.fit_transform(X)
        
        # Iterate over different values of eps and min_samples
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Initialize DBSCAN
                labels = dbscan.fit_predict(X_transformed)  # Fit and predict cluster labels
                
                # Compute silhouette score only if there are multiple clusters
                if len(set(labels)) > 1:
                    score = silhouette_score(X_transformed, labels)
                    
                    # Update best parameters if current configuration is better
                    if score > best_params['score']:
                        best_params.update({'eps': eps, 'min_samples': min_samples, 'pca_dim': pca_dim, 'score': score})
    
    return best_params

# Main execution block
if __name__ == "__main__":
    # Get the data
    training_data_normal, testing_data_attack, testing_data_normal = load_npy_data()
    X = training_data_normal
    
    # Define hyperparameter ranges to test
    k_values = [2, 3, 4, 5, 6]  # Number of clusters
    t_values = [1e-3, 1e-4, 1e-5]  # Tolerance values
    eps_values = [0.1, 0.3, 0.5, 0.7]  # Epsilon values for DBSCAN
    min_samples_values = [3, 5, 10]  # Minimum samples for DBSCAN
    pca_dimensions = [None, 2, 3]  # PCA dimensions to test
    
    # Perform hyperparameter tuning for k-Means
    best_kmeans_params = tune_kmeans(X, k_values, t_values, pca_dimensions)
    print(f"Best k-Means Parameters: {best_kmeans_params}")
    
    # Perform hyperparameter tuning for DBSCAN
    best_dbscan_params = tune_dbscan(X, eps_values, min_samples_values, pca_dimensions)
    print(f"Best DBSCAN Parameters: {best_dbscan_params}")