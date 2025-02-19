import numpy as np
import matplotlib.pyplot as plt
#Below libraries of sklearn are just to VERIFY performance.
from sklearn.cluster import DBSCAN
#from sklearn.datasets.samples_generator import make_blobs

# Load the data from the .npy files (path may need to be changed)
def load_npy_data():
    training_data_normal = np.load('training_normal.npy')
    testing_data_attack = np.load('testing_attack.npy')
    testing_data_normal = np.load('testing_normal.npy')
    return training_data_normal, testing_data_attack, testing_data_normal

# Get accuracy, TPR, FPR, and F1 score
def get_performance_metric(predicted_labels, actual_labels):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == 'attack' and actual_labels[i] == 'attack':
            TP += 1
        elif predicted_labels[i] == 'attack' and actual_labels[i] == 'normal':
            FP += 1
        elif predicted_labels[i] == 'normal' and actual_labels[i] == 'normal':
            TN += 1
        elif predicted_labels[i] == 'normal' and actual_labels[i] == 'attack':
            FN += 1
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    TPR = TP / (FN + TP)
    FPR = FP / (TN + FP)
    F1 = 2 * TP / (2 * TP + FP + FN)
    return accuracy, TPR, FPR, F1

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def region_query(data, point_idx, epsilon):
    neighbors = []
    for i in range(len(data)):
        if euclidean_distance(data[point_idx], data[i]) <= epsilon:
            neighbors.append(i)
    return neighbors

def dbscan(data, epsilon, min_points):
    labels = [-1] * len(data)  # -1 means unclassified
    cluster_id = 0
    core_points = set()
    
    for i in range(len(data)):
        if labels[i] != -1:
            continue
        
        neighbors = region_query(data, i, epsilon)
        
        if len(neighbors) < min_points:
            labels[i] = -2  # Mark as noise
        else:
            core_points.add(tuple(data[i]))
            cluster_id += 1
            labels[i] = cluster_id
            expand_cluster(data, labels, neighbors, cluster_id, epsilon, min_points, core_points)
    
    return labels, core_points

def expand_cluster(data, labels, neighbors, cluster_id, epsilon, min_points, core_points):
    i = 0
    while i < len(neighbors):
        point_idx = neighbors[i]
        if labels[point_idx] == -2:  # Previously marked as noise, now part of cluster
            labels[point_idx] = cluster_id
        if labels[point_idx] == -1:  # Unclassified
            labels[point_idx] = cluster_id
            new_neighbors = region_query(data, point_idx, epsilon)
            if len(new_neighbors) >= min_points:
                core_points.add(tuple(data[point_idx]))
                neighbors += new_neighbors
        i += 1

def classify_samples(test_data, core_points, epsilon):
    classifications = []
    for sample in test_data:
        is_normal = any(euclidean_distance(sample, np.array(core_point)) <= epsilon for core_point in core_points)
        classifications.append("Normal" if is_normal else "Anomalous")
    return classifications

# def main():
    
#     training_data_normal, testing_data_attack, testing_data_normal = load_npy_data()

#     # Concatenate the testing data and generate labels (1 for attack, 0 for normal)
#     testing_data = np.concatenate((testing_data_attack, testing_data_normal))
#     testing_labels = np.concatenate((np.ones(len(testing_data_attack)), np.zeros(len(testing_data_normal))))

#     labels, core_points = dbscan(training_data, epsilon, min_points)
#     classifications = classify_samples(testing_data, core_points, epsilon)

#     dbscan(training_data_normal, testing_data, testing_labels, threshold=0.1, k=4)

#     print("Core Points:", core_points)
#     print("Classifications:", classifications)
    
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                             random_state=0)

# X = StandardScaler().fit_transform(X)    
    
# plt.scatter(X[:,0], X[:,1]) #Our original data - unclustered

# own_labels = own_dbscan(X,0.3,10)

# plt.scatter(X[:,0], X[:,1], marker='o', s=14, c=own_labels, cmap='rainbow')

# db = DBSCAN(eps=0.3, min_samples=10).fit(X)
# sklearn_labels = db.labels_


# plt.scatter(X[:,0], X[:,1], marker='o', s=14, c=sklearn_labels, cmap='rainbow')    
    
# if __name__ == "__main__":
#     main()