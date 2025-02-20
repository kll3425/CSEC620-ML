import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# TODO: Delete - below libraries of sklearn are just to VERIFY performance.
from sklearn.cluster import DBSCAN
# from sklearn.datasets.samples_generator import make_blobs

# Load the data from the .npy files (path may need to be changed)
def load_npy_data():
    training_data_normal = np.load('./CSEC620-ML/assignment_02/training_normal.npy')
    testing_data_attack = np.load('./CSEC620-ML/assignment_02/testing_attack.npy')
    testing_data_normal = np.load('./CSEC620-ML/assignment_02/testing_normal.npy')
    return training_data_normal, testing_data_attack, testing_data_normal

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
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    TPR = TP / (FN + TP) if (FN + TP) > 0 else 0
    FPR = FP / (TN + FP) if (TN + FP) > 0 else 0
    F1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    return accuracy, TPR, FPR, F1

def region_query(data, point_index, epsilon):
    distances = np.linalg.norm(data - data[point_index], axis=1)
    return np.where(distances <= epsilon)[0]

""" ChatGPT was used to optimize the loops in the original fit function """
def dbscan_fit(training_data, epsilon=0.1, min_pts=4):
    n = len(training_data)
    cluster_labels = np.full(n, -1)  # -1 means noise, will be updated with cluster numbers
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for point_index in range(n):
        if visited[point_index]:
            continue
        
        visited[point_index] = True
        neighbors = region_query(training_data, point_index, epsilon)

        if len(neighbors) < min_pts:
            cluster_labels[point_index] = -1  # Noise
            continue
        
        # Start a new cluster
        cluster_labels[point_index] = cluster_id
        queue = set(neighbors.tolist())  # Using set to avoid duplicate insertions

        while queue:
            neighbor_index = queue.pop()
            if not visited[neighbor_index]:
                visited[neighbor_index] = True
                new_neighbors = region_query(training_data, neighbor_index, epsilon)
                if len(new_neighbors) >= min_pts:
                    queue.update(new_neighbors.tolist())

            # Assign the neighbor to the current cluster if it's unclassified
            if cluster_labels[neighbor_index] == -1:
                cluster_labels[neighbor_index] = cluster_id

        cluster_id += 1  # Move to the next cluster

    return cluster_labels

def near_core_point(core_points, test_point, epsilon):
    # Return if the test point is within epsilon of a core point
    distances = np.linalg.norm(core_points - test_point, axis=1)
    return np.any(distances <= epsilon)

def run_dbscan(training_data, testing_data, testing_labels, epsilon=0.05, num_neighbors=4):
    # Use PCA to reduce the dimensionality of the data to the specified number of dimension
    pca = PCA(n_components=2)

    # Fit the PCA model to the training data and project the training data onto n principal components
    training_data_projected = pca.fit_transform(training_data)

    # Project the testing data to the same number of dimensions as the training data
    testing_data_projected = pca.transform(testing_data)

    # Fit DBScan on training data
    cluster_labels = dbscan_fit(training_data_projected, epsilon, num_neighbors)

    # Classify testing points based on core points
    core_points = training_data_projected[cluster_labels != -1]

    # Classify the testing points
    predicted_labels = np.array([0 if near_core_point(core_points, test_point, epsilon) else 1 for test_point in testing_data_projected])

    # Get the performance metrics
    accuracy, TPR, FPR, F1 = get_performance_metric(predicted_labels, testing_labels)

    # Print the performance metrics
    print("Accuracy: ", accuracy)
    print("True Positive Rate: ", TPR)
    print("False Positive Rate: ", FPR)
    print("F1 Score: ", F1)

    # Create subplots to show the predicted labels and the actual labels
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Create labels for the data points
    labels = ['Normal', 'Attack']
    colors = ['blue', 'red']

    # Plot the testing data with the predicted labels
    predicted_points = [testing_data_projected[np.where(predicted_labels == 0)], testing_data_projected[np.where(predicted_labels == 1)]]
    for i in range(2):
        axs[0].scatter(predicted_points[i][:, 0], predicted_points[i][:, 1], label=labels[i], color=colors[i])

    # Plot the testing data with the actual labels
    actual_points = [testing_data_projected[np.where(testing_labels == 0)], testing_data_projected[np.where(testing_labels == 1)]]
    for i in range(2):
        axs[1].scatter(actual_points[i][:, 0], actual_points[i][:, 1], label=labels[i], color=colors[i])

    # Set the titles and labels for the subplots
    axs[0].set_title('Predicted Labels')
    axs[0].set_xlabel('Principal Component 1')
    axs[0].set_ylabel('Principal Component 2')
    axs[0].legend()

    axs[1].set_title('Actual Labels')
    axs[1].set_xlabel('Principal Component 1')
    axs[1].set_ylabel('Principal Component 2')
    axs[1].legend()

    # Show the plots
    plt.show()

def main():
    # Get the data
    training_data_normal, testing_data_attack, testing_data_normal = load_npy_data()

    # Concatenate the testing data and generate labels (1 for attack, 0 for normal)
    testing_data = np.concatenate((testing_data_attack[:500], testing_data_normal[:500]))
    testing_labels = np.concatenate((np.ones(len(testing_data_attack[:500])), np.zeros(len(testing_data_normal[:500]))))

    # DBScan is only run on a subset of the data as the algorithm is slow
    run_dbscan(training_data_normal[:1000], testing_data, testing_labels, epsilon=0.015, num_neighbors=3)

if __name__ == "__main__":
    main()