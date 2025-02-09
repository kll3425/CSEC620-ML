import numpy as np
from sklearn.decomposition import PCA

# Load the data from the .npy files (path may need to be changed)
def load_npy_data():
    training_data_normal = np.load('CSEC620-ML/assignment_02/training_normal.npy')
    testing_data_attack = np.load('CSEC620-ML/assignment_02/testing_attack.npy')
    testing_data_normal = np.load('CSEC620-ML/assignment_02/testing_normal.npy')
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

def main():
    training_data_normal, testing_data_attack, testing_data_normal = load_npy_data()
    show_data_shapes(training_data_normal, testing_data_attack, testing_data_normal)

if __name__ == "__main__":
    main()