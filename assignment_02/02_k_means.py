"""
(3.1) k-Means
● Use k-Means clustering to identify the centroids of the clusters in the normal traffic
training set.
● Select a distance threshold value, t.
● For each test sample, find the cluster centroid to which the sample is closest using a
distance function (e.g. Euclidean distance). If the distance is less than your threshold
value t, then classify the sample as normal. If it is greater than your threshold value t,
then classify the sample as anomalous.
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the data from the .npy files (path may need to be changed)
def load_npy_data():
    training_data_normal = np.load('./CSEC620-ML/assignment_02/training_normal.npy')
    testing_data_attack = np.load('./CSEC620-ML/assignment_02/testing_attack.npy')
    testing_data_normal = np.load('./CSEC620-ML/assignment_02/testing_normal.npy')
    return training_data_normal, testing_data_attack, testing_data_normal

# Show the size of the data
def show_data_shapes(training_data_normal, testing_data_attack, testing_data_normal):
    print("Training data normal shape: ", training_data_normal.shape)
    print("Testing data attack shape: ", testing_data_attack.shape)
    print("Testing data normal shape: ", testing_data_normal.shape)

# Get accuracy, TPR, FPR, and F1 score
def get_performance_metric(predicted_labels, actual_labels):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == 1 and actual_labels[i] == 1:
            TP += 1
        elif predicted_labels[i] == 1 and actual_labels[i] == 0:
            FP += 1
        elif predicted_labels[i] == 0 and actual_labels[i] == 0:
            TN += 1
        elif predicted_labels[i] == 0 and actual_labels[i] == 1:
            FN += 1
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    TPR = TP / (FN + TP)
    FPR = FP / (TN + FP)
    F1 = 2 * TP / (2 * TP + FP + FN)
    return accuracy, TPR, FPR, F1

def init_k_clusters(training_data, k=1):
    # Randomly select k points from the training data as the initial cluster centroids
    cluster_centroids = []
    cluster_indexes = []
    num_clusters_appended = 0
    while num_clusters_appended < k:
        random_index = np.random.randint(0, len(training_data))
        if random_index not in cluster_indexes:
            cluster_centroids.append(training_data[random_index])
            cluster_indexes.append(random_index)
            num_clusters_appended += 1
    return cluster_centroids

def cluster_data(cluster_centroids, data):
    cluster_points = [[] for i in range(len(cluster_centroids))]
    for point in data:
        min_distance = np.linalg.norm(point - cluster_centroids[0])
        closest_cluster = 0
        for i in range(1, len(cluster_centroids)):
            distance = np.linalg.norm(point - cluster_centroids[i])
            if distance < min_distance:
                min_distance = distance
                closest_cluster = i
        cluster_points[closest_cluster].append(point)
    return cluster_points

def cluster_data_classify(cluster_centroids, data, threshold=0.5):
    classify_points = [[] for i in range(2)]
    point_and_prediction = [[] for i in range(2)]
    for point in data:
        min_distance = np.linalg.norm(point - cluster_centroids[0])
        closest_cluster = 0
        for i in range(1, len(cluster_centroids)):
            distance = np.linalg.norm(point - cluster_centroids[i])
            if distance < min_distance:
                min_distance = distance
                closest_cluster = i
        # Apply threshold, put in classify_points[0] if distance is less than threshold, else put in classify_points[1]
        # Index 0 is normal, index 1 is attack
        if min_distance < threshold:
            classify_points[0].append(point)
            point_and_prediction[0].append(point)
            point_and_prediction[1].append(0)
        else:
            classify_points[1].append(point)
            point_and_prediction[0].append(point)
            point_and_prediction[1].append(1)
    return classify_points, point_and_prediction

def k_means(training_data, k=1, max_iterations=100):
    # Initialize the cluster centroids
    cluster_centroids = init_k_clusters(training_data, k)
    # Iterate until convergence (centroids no longer change), or until max_iterations is reached
    for iteration in range(max_iterations):
        # Copy previous cluster centroids for comparison
        previous_cluster_centroids = cluster_centroids.copy()
        # Cluster the data
        cluster_points = cluster_data(cluster_centroids, training_data)
        # Update the cluster centroids
        for i in range(len(cluster_centroids)):
            cluster_centroids[i] = np.mean(cluster_points[i], axis=0)
        # Check for convergence
        if np.array_equal(previous_cluster_centroids, cluster_centroids):
            break
    return cluster_centroids

def run_k_means_training(training_data, k=4, num_dimensions=2):
    # Use PCA to reduce the dimensionality of the data
    pca = PCA(n_components=2)
    training_data_normal_projected = pca.fit_transform(training_data)

    # Perform k-Means clustering on the projected data
    cluster_centroids = k_means(training_data_normal_projected, k)

    # Print the cluster centroids
    print("Cluster Centroids:")
    for i in range(len(cluster_centroids)):
        print(f"Cluster {i+1}: {cluster_centroids[i]}")

    # Cluster the training data
    cluster_points = cluster_data(cluster_centroids, training_data_normal_projected)

    """ ChatGPT assistance was used to generate the code to plot the clustered data """
    # Plot the clustered data
    for i in range(len(cluster_points)):
        # Convert the cluster points to a numpy array
        points = np.array(cluster_points[i])
        # Plot the cluster points
        plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i+1}')
        # Plot the cluster centroids
        plt.scatter(cluster_centroids[i][0], cluster_centroids[i][1], marker='x', color='black')
    
    # Set labels
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('k-Means Clustering Of Data On First Two Principal Components')
    plt.legend()

    # Show the plot
    plt.show()

def run_k_means_classifier(training_data, testing_data, testing_labels, threshold=0.5, k=4, num_dimensions=2):
    # Use PCA to reduce the dimensionality of the data to the specified number of dimension
    pca = PCA(n_components=num_dimensions)

    # Fit the PCA model to the training data and project the training data onto n principal components
    training_data_normal_projected = pca.fit_transform(training_data)

    # Project the testing data to the same number of dimensions as the training data
    testing_data_projected = pca.transform(testing_data)

    # Perform k-Means clustering on the projected training data
    cluster_centroids = k_means(training_data_normal_projected, k)

    # Print the cluster centroids
    print("Cluster Centroids:")
    for i in range(len(cluster_centroids)):
        print(f"Cluster {i+1}: {cluster_centroids[i]}")

    # Cluster the testing data using the updated cluster centroids
    predicted_points, point_and_prediction = cluster_data_classify(cluster_centroids, testing_data_projected, threshold)

    # Create subplots to show the clustered data and the testing labels
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Create labels for the data points
    labels = ['Normal', 'Attack']
    colors = ['blue', 'red']

    # Plot the predicted data
    for i in range(len(predicted_points)):
        # Plot the predicted classifications for testing points
        axs[0].scatter(np.array(predicted_points[i])[:, 0], np.array(predicted_points[i])[:, 1], label=labels[i], color=colors[i])
    
    # Set labels for clustered data plot
    axs[0].set_xlabel('First Principal Component')
    axs[0].set_ylabel('Second Principal Component')
    axs[0].set_title('k-Means Classifier Predictions')
    axs[0].legend()

    # Plot the testing data with the actual labels
    labeled_points = [testing_data_projected[np.where(testing_labels == 0)], testing_data_projected[np.where(testing_labels == 1)]]
    for i in range(2):
        axs[1].scatter(labeled_points[i][:, 0], labeled_points[i][:, 1], label=labels[i], color=colors[i])
    
    # Set labels for testing data plot
    axs[1].set_xlabel('First Principal Component')
    axs[1].set_ylabel('Second Principal Component')
    axs[1].set_title('Labeled Testing Data')
    axs[1].legend()

    # Get performance metrics
    accuracy, TPR, FPR, F1 = get_performance_metric(point_and_prediction[1], testing_labels)
    print("Accuracy: ", accuracy)
    print("TPR: ", TPR)
    print("FPR: ", FPR)
    print("F1 Score: ", F1)

    # Show the plot
    plt.show()

def main():
    # Get the data
    training_data_normal, testing_data_attack, testing_data_normal = load_npy_data()
    # show_data_shapes(training_data_normal, testing_data_attack, testing_data_normal)

    # Run k-Means clustering on the training data
    # run_k_means_training(training_data_normal, k=4)

    """
    # TODO: DELETE
    # Visualize the projected testing data
    # Generate PCA model on training data
    pca = PCA(n_components=2)
    training_data_normal_projected = pca.fit_transform(training_data_normal)
    # Project attack data to PCA
    testing_data_attack_projected = pca.transform(testing_data_attack)
    # Show the projected data
    plt.scatter(testing_data_attack_projected[:, 0], testing_data_attack_projected[:, 1], label='Attack')
    plt.show()
    """

    # Concatenate the testing data and generate labels (1 for attack, 0 for normal)
    testing_data = np.concatenate((testing_data_attack, testing_data_normal))
    testing_labels = np.concatenate((np.ones(len(testing_data_attack)), np.zeros(len(testing_data_normal))))

    # Run k-Means clustering 1 time
    run_k_means_classifier(training_data_normal, testing_data, testing_labels, threshold=0.1, k=4)
    
if __name__ == "__main__":
    main()