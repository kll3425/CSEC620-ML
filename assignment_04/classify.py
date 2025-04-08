#!/usr/bin/env python3

import json
import argparse
import os
import numpy as np
import tqdm
import pandas as pd

# Supress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# seed value
# (ensures consistent dataset splitting between runs)
SEED = 0


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    def check_path(parser, x):
        if not os.path.exists(x):
            parser.error("That directory {} does not exist!".format(x))
        else:
            return x
    parser.add_argument('-r', '--root', type=lambda x: check_path(parser, x), 
                        help='The path to the root directory containing feature files.')
    parser.add_argument('-s', '--split', type=float, default=0.7, 
                        help='The percentage of samples to use for training.')

    return parser.parse_args()


def load_data(root, min_samples=20, max_samples=1000):
    """Load json feature files produced from feature extraction.

    The device label (MAC) is identified from the directory in which the feature file was found.
    Returns X and Y as separate multidimensional arrays.
    The instances in X contain only the first 6 features.
    The ports, domain, and cipher features are stored in separate arrays for easier process in stage 0.

    Parameters
    ----------
    root : str
           Path to the directory containing samples.
    min_samples : int
                  The number of samples each class must have at minimum (else it is pruned).
    max_samples : int
                  Stop loading samples for a class when this number is reached.

    Returns
    -------
    features_misc : numpy array
    features_ports : numpy array
    features_domains : numpy array
    features_ciphers : numpy array
    labels : numpy array
    """
    X = []
    X_p = []
    X_d = []
    X_c = []
    Y = []

    port_dict = dict()
    domain_set = set()
    cipher_set = set()

    # create paths and do instance count filtering
    fpaths = []
    fcounts = dict()
    for rt, dirs, files in os.walk(root):
        for fname in files:
            path = os.path.join(rt, fname)
            label = os.path.basename(os.path.dirname(path))
            name = os.path.basename(path)
            if name.startswith("features") and name.endswith(".json"):
                fpaths.append((path, label, name))
                fcounts[label] = 1 + fcounts.get(label, 0)

    # load samples
    processed_counts = {label:0 for label in fcounts.keys()}
    for fpath in tqdm.tqdm(fpaths):
        path = fpath[0]
        label = fpath[1]
        if fcounts[label] < min_samples:
            continue
        if processed_counts[label] >= max_samples:
            continue
        processed_counts[label] += 1
        with open(path, "r") as fp:
            features = json.load(fp)
            instance = [features["flow_volume"],
                        features["flow_duration"],
                        features["flow_rate"],
                        features["sleep_time"],
                        features["dns_interval"],
                        features["ntp_interval"]]
            X.append(instance)
            X_p.append(list(features["ports"]))
            X_d.append(list(features["domains"]))
            X_c.append(list(features["ciphers"]))
            Y.append(label)
            domain_set.update(list(features["domains"]))
            cipher_set.update(list(features["ciphers"]))
            for port in set(features["ports"]):
                port_dict[port] = 1 + port_dict.get(port, 0)

    # prune rarely seen ports
    port_set = set()
    for port in port_dict.keys():
        if port_dict[port] > 10:
            port_set.add(port)

    # map to wordbag
    print("Generating wordbags ... ")
    for i in tqdm.tqdm(range(len(Y))):
        X_p[i] = list(map(lambda x: X_p[i].count(x), port_set))
        X_d[i] = list(map(lambda x: X_d[i].count(x), domain_set))
        X_c[i] = list(map(lambda x: X_c[i].count(x), cipher_set))

    return np.array(X).astype(float), np.array(X_p), np.array(X_d), np.array(X_c), np.array(Y)


def classify_bayes(X_tr, Y_tr, X_ts, Y_ts):
    """
    Use a multinomial naive bayes classifier to analyze the 'bag of words' seen in the ports/domain/ciphers features.
    Returns the prediction results for the training and testing datasets as an array of tuples in which each row
      represents a data instance and each tuple is composed as the predicted class and the confidence of prediction.

    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels

    Returns
    -------
    C_tr : numpy array
           Prediction results for training samples.
    C_ts : numpy array
           Prediction results for testing samples.
    """
    classifier = MultinomialNB()
    classifier.fit(X_tr, Y_tr)

    # produce class and confidence for training samples
    C_tr = classifier.predict_proba(X_tr)
    C_tr = [(np.argmax(instance), max(instance)) for instance in C_tr]

    # produce class and confidence for testing samples
    C_ts = classifier.predict_proba(X_ts)
    C_ts = [(np.argmax(instance), max(instance)) for instance in C_ts]

    return C_tr, C_ts


def do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts):
    """
    Perform stage 0 of the classification procedure:
        process each multinomial feature using naive bayes
        return the class prediction and confidence score for each instance feature

    Parameters
    ----------
    Xp_tr : numpy array
           Array containing training (port) samples.
    Xp_ts : numpy array
           Array containing testing (port) samples.
    Xd_tr : numpy array
           Array containing training (port) samples.
    Xd_ts : numpy array
           Array containing testing (port) samples.
    Xc_tr : numpy array
           Array containing training (port) samples.
    Xc_ts : numpy array
           Array containing testing (port) samples.
    Y_tr : numpy array
           Array containing training labels.
    Y_ts : numpy array
           Array containing testing labels

    Returns
    -------
    resp_tr : numpy array
              Prediction results for training (port) samples.
    resp_ts : numpy array
              Prediction results for testing (port) samples.
    resd_tr : numpy array
              Prediction results for training (domains) samples.
    resd_ts : numpy array
              Prediction results for testing (domains) samples.
    resc_tr : numpy array
              Prediction results for training (cipher suites) samples.
    resc_ts : numpy array
              Prediction results for testing (cipher suites) samples.
    """
    # perform multinomial classification on bag of ports
    resp_tr, resp_ts = classify_bayes(Xp_tr, Y_tr, Xp_ts, Y_ts)

    # perform multinomial classification on domain names
    resd_tr, resd_ts = classify_bayes(Xd_tr, Y_tr, Xd_ts, Y_ts)

    # perform multinomial classification on cipher suites
    resc_tr, resc_ts = classify_bayes(Xc_tr, Y_tr, Xc_ts, Y_ts)

    return resp_tr, resp_ts, resd_tr, resd_ts, resc_tr, resc_ts


def do_stage_1(X_tr, X_ts, Y_tr, Y_ts):
    """
    Perform stage 1 of the classification procedure:
        train a random forest classifier using the NB prediction probabilities

    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels

    Returns
    -------
    rf_pred : numpy array
              Final predictions from the Random Forest on the testing dataset.
    dt_pred : numpy array
              Final predictions from the Decision Tree on the testing dataset.
    """

    print("\n--- Decision Tree ---")
    
    def multiclass_accuracy(y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        return correct / total
    
    print("Training Decision Tree...")

    # Decision tree with hyperparameter tuning (testing different max_depth and min_node values)
    def tune_decision_tree(X_tr, X_ts, Y_tr, Y_ts, max_depth_vals=[5, 10], min_node_vals=[2, 5]):
        best_acc = 0
        best_preds = None
        for max_depth in max_depth_vals:
            for min_node in min_node_vals:
                dt_preds, _, _ = decision_tree(X_tr, X_ts, Y_tr, Y_ts, max_depth=max_depth, min_node=min_node, all_classes=np.unique(Y_tr))
                dt_acc = multiclass_accuracy(Y_ts, dt_preds)
                print(f"Decision Tree Accuracy (max_depth={max_depth}, min_node={min_node}): {dt_acc:.4f}")
                if dt_acc > best_acc:
                    best_acc = dt_acc
                    best_preds = dt_preds
        return best_preds
    
    dt_preds = tune_decision_tree(X_tr, X_ts, Y_tr, Y_ts)
    
    print("\nTraining Random Forest...")
    
    # Random Forest with hyperparameter tuning (testing different n_trees, max_depth, min_node)
    def tune_random_forest(X_tr, Y_tr, X_ts, Y_ts, n_trees_vals=[10, 50], max_depth_vals=[5, 10], min_node_vals=[2, 5], data_frac_vals=[0.7, 0.8], feature_subcount_vals=[5, 10]):
        best_acc = 0
        best_preds = None
        for n_trees in n_trees_vals:
            for max_depth in max_depth_vals:
                for min_node in min_node_vals:
                    for data_frac in data_frac_vals:
                        for feature_subcount in feature_subcount_vals:
                            rf_preds = random_forest(X_tr, X_ts, Y_tr, Y_ts, n_trees=n_trees, data_frac=data_frac, feature_subcount=feature_subcount, max_depth=max_depth, min_node=min_node, all_classes=np.unique(Y_tr))
                            rf_acc = multiclass_accuracy(Y_ts, rf_preds)
                            print(f"Random Forest Accuracy (n_trees={n_trees}, max_depth={max_depth}, min_node={min_node}, data_frac={data_frac}, feature_subcount={feature_subcount}): {rf_acc:.4f}")
                            if rf_acc > best_acc:
                                best_acc = rf_acc
                                best_preds = rf_preds
        return best_preds
    
    rf_preds = tune_random_forest(X_tr, Y_tr, X_ts, Y_ts)
    
    # Return the final tuned predictions
    return rf_preds, dt_preds


def gini_impurity(data_points, all_classes):
    """
    Parameters
    ----------
    data_points : numpy array
           Array containing training labels.
    all_classes : numpy array
           Array containing all labels.

    Returns
    -------
    GINI impurity score of a single node.
    """

    # Loop thorough each sample and accumulate the count per class 
    sample_counts = {}
    for each_class in all_classes:
       # Initialize each class to 0
       sample_counts[each_class] = 0
    for sample in data_points:
       # Increment sample count for specific class
       sample_counts[sample] += 1

    # Calculate the probability of each class
    class_probabilities = []
    for each_class in all_classes:
       # Count the number of samples of a single class and divide it by the total number of samples in the node
       # Store the probabilities as the square of the value
       class_probabilities.append(np.square(sample_counts[each_class] / len(data_points)))

    # Calculate the GINI impurity of the node by summing the square probabilities, then subtracting it from 1
    return 1 - np.sum(class_probabilities)

def weighted_gini_impurity(list_of_splits, all_classes):
    """
    Parameters
    ----------
    list_of_splits : list of numpy arrays
           List containing arrays containing training labels.
    all_classes : numpy array
           Array containing all labels.

    Returns
    -------
    Weighted GINI impurity score for a list of splits from the same parent node.
    """

    # Return the sum of the product of a node's GINI and percent of total data points out of the total in the parent node
    num_data_points = sum(len(split) for split in list_of_splits)
    return np.sum((len(split) / num_data_points) * gini_impurity(split, all_classes) for split in list_of_splits)

def find_best_split(feature_values, labels, all_classes):
    # Zip pairs of feature values and labels, then sort pairs by the feature values
    data = sorted(zip(feature_values, labels), key=lambda x: x[0])
    # Store best split value for the specific feature
    lowest_gini = float('inf')
    best_split = None
    empty_split = False
    # Loop through the dataset, splitting between each adjacent pair of points and calculating the GINI
    for i in range(1, len(feature_values)):
        # If the feature_value of the next sample is the same as the current, skip the redundant calculation
        if data[i - 1][0] == data[i][0]:
           continue
        # Create split dataset labels, ignoring feature values
        left_split = [label for _, label in data[:i]]
        right_split = [label for _, label in data[i:]]
        # Compute weighted GINI of split nodes
        weighted_gini = weighted_gini_impurity([left_split, right_split], all_classes)
        # Update best split value if calculated GINI score is lower
        if weighted_gini < lowest_gini:
           lowest_gini = weighted_gini
           best_split = (data[i - 1][0] + data[i][0]) / 2
           # Check for empty split
           if len(left_split) == 0 or len(right_split) == 0:
               empty_split = True
           else:
               empty_split = False
    return best_split, lowest_gini, empty_split

def decision_tree(X_tr, X_ts, Y_tr, Y_ts, max_depth, min_node, all_classes, ts_indices=None):
    """
    Predict the result of the sample using a random decision tree

    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels
    max_depth : int
           Maximum depth of the tree.
    min_node : int
           Minimum number of samples in a node to split.
    all_classes : numpy array
           Array containing all labels.
    ts_indices : numpy array, optional
           Array containing original indices of the testing samples and predictions.

    Returns
    -------
    Predictions, labels, and indices of the test samples.
    """
    # Initialize the indices of the test samples if not provided
    if ts_indices is None:
        ts_indices = np.arange(len(X_ts))
    # Transpose the array of samples to obtain an array of features
    features = X_tr.T
    # Check if node meets ending criteria
    if max_depth == 0 or len(X_tr) < min_node:
        # Return most popular class in remaining data for all test points and test labels along with the indices
        return [np.argmax(np.bincount(Y_tr)) for i in range(len(X_ts))], Y_ts, ts_indices
    # Store the split and lowest GINI of each feature to determine best feature to split on
    best_feature_split, lowest_feature_gini, empty_split = find_best_split(features[0], Y_tr, all_classes)
    best_feature = 0
    for i in range(1, len(features)):
        best_split, lowest_gini, empty_split_curr = find_best_split(features[i], Y_tr, all_classes)
        if lowest_gini < lowest_feature_gini:
            best_feature_split = best_split
            lowest_feature_gini = lowest_gini
            best_feature = i
            empty_split = empty_split_curr
    # Calculate GINI of current node
    gini = gini_impurity(Y_tr, all_classes)
    # Check additional node ending criteria
    if lowest_feature_gini > gini or empty_split:
        # Return most popular class in remaining data for all test points and test labels along with the indices
        return [np.argmax(np.bincount(Y_tr)) for i in range(len(X_ts))], Y_ts, ts_indices
    # Create boolean masks to split dataset
    left_train_mask = X_tr[:, best_feature] <= best_feature_split
    left_test_mask = X_ts[:, best_feature] <= best_feature_split
    right_train_mask = X_tr[:, best_feature] > best_feature_split
    right_test_mask = X_ts[:, best_feature] > best_feature_split
    # Extra check to ensure that the split is not empty
    if len(left_train_mask) == 0 or len(right_train_mask) == 0:
        # Return most popular class in remaining data for all test points and test labels along with the indices
        return [np.argmax(np.bincount(Y_tr)) for i in range(len(X_ts))], Y_ts, ts_indices
    # Predict test samples using recursive splits
    left_predictions, left_labels, left_indexes = decision_tree(
        X_tr[left_train_mask],
        X_ts[left_test_mask], 
        Y_tr[left_train_mask], 
        Y_ts[left_test_mask], 
        max_depth - 1, 
        min_node, 
        all_classes,
        ts_indices[left_test_mask]
    )
    right_predictions, right_labels, right_indexes = decision_tree(
        X_tr[right_train_mask],
        X_ts[right_test_mask], 
        Y_tr[right_train_mask], 
        Y_ts[right_test_mask], 
        max_depth - 1, 
        min_node, 
        all_classes,
        ts_indices[right_test_mask]
    )
    return np.concatenate((left_predictions, right_predictions)), np.concatenate((left_labels, right_labels)), np.concatenate((left_indexes, right_indexes))

def random_forest(train_samples, test_samples, train_labels, test_labels, n_trees, data_frac, feature_subcount, max_depth, min_node, all_classes):
    """
    Predict the result of the sample using a random forest classifier

    Parameters
    ----------
    train_samples : numpy array
           Array containing training samples.
    train_labels : numpy array
           Array containing training labels.
    test_samples : numpy array
           Array containing testing samples.
    test_labels : numpy array
           Array containing testing labels
    n_trees : int
           Number of trees to use per random forest.
    data_frac : int
           Percentage of total data per tree/subset of data.
    feature_subcount : int
           Number of features to use per tree.
    max_depth : int
           Maximum depth of each tree.
    min_node : int
           Minimum number of samples in a node to split for each tree.
    all_classes : numpy array
           Array containing all labels.
    ts_indices : numpy array, optional
           Array containing original indices of the testing samples and predictions.

    Returns
    -------
    Predictions of the test samples.
    """
    all_predictions = []
    # Build n trees
    for i in range(n_trees):
        # Create a subset of data using random sampling until the subset if data_frac percent of the total data
        subset_samples = []
        subset_labels = []
        for i in range(int(len(train_samples) * data_frac)):
            rand_index = np.random.randint(0, len(train_samples))
            subset_samples.append(train_samples[rand_index])
            subset_labels.append(train_labels[rand_index])
        subset_samples = np.array(subset_samples)
        subset_labels = np.array(subset_labels)
        # Randomly select feature_subcount features to use for the tree
        feature_indices = np.random.choice(len(train_samples[0]), feature_subcount, replace=False)
        # Create a new decision tree using the subset of data and features
        predictions, labels, indexes = decision_tree(subset_samples[:, feature_indices], test_samples[:, feature_indices], subset_labels, test_labels, max_depth, min_node, all_classes)
        # Store the predictions of the tree using the original feature indices
        sorted_predictions = np.zeros(len(predictions), dtype=int)
        for i in range(len(predictions)):
            sorted_predictions[i] = int(predictions[np.where(indexes == i)[0]])
        all_predictions.append(sorted_predictions)
    # Transpose predictions to sort by sample, then count most common prediction per sample and return the list of predictions
    all_predictions = np.array(all_predictions, dtype=int).T
    return np.array([np.bincount(sample).argmax() for sample in all_predictions])

def class_by_class_confusion_matrix(predictions, labels, all_classes):
       # Create a confusion matrix for each class.
       confusion_matrix = np.zeros((len(all_classes), len(all_classes)), dtype=int)
       for i in range(len(predictions)):
              confusion_matrix[labels[i]][predictions[i]] += 1
       # Save the confusion matrix to "confusion_matrix.csv" using pandas
       df = pd.DataFrame(confusion_matrix, index=all_classes, columns=all_classes)
       df.to_csv("confusion_matrix.csv", index=True, header=True)

def main(args):
    """
    Perform main logic of program
    """
    # Specify path or use args.root
    path = ".\\assignment_04\\iot_data"
    argsroot = path if path != "" else args.root

    # load dataset
    print("Loading dataset ... ")
    # X, X_p, X_d, X_c, Y = load_data(args.root)
    X, X_p, X_d, X_c, Y = load_data(argsroot)

    # encode labels
    print("Encoding labels ... ")
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)

    print("Dataset statistics:")
    print("\t Classes: {}".format(len(le.classes_)))
    print("\t Samples: {}".format(len(Y)))
    print("\t Dimensions: ", X.shape, X_p.shape, X_d.shape, X_c.shape)

    # shuffle
    print("Shuffling dataset using seed {} ... ".format(SEED))
    s = np.arange(Y.shape[0])
    np.random.seed(SEED)
    np.random.shuffle(s)
    X, X_p, X_d, X_c, Y = X[s], X_p[s], X_d[s], X_c[s], Y[s]

    # split
    print("Splitting dataset using train:test ratio of {}:{} ... ".format(int(args.split*10), int((1-args.split)*10)))
    cut = int(len(Y) * args.split)
    X_tr, Xp_tr, Xd_tr, Xc_tr, Y_tr = X[cut:], X_p[cut:], X_d[cut:], X_c[cut:], Y[cut:]
    X_ts, Xp_ts, Xd_ts, Xc_ts, Y_ts = X[:cut], X_p[:cut], X_d[:cut], X_c[:cut], Y[:cut]

    # perform stage 0
    print("Performing Stage 0 classification ... ")
    p_tr, p_ts, d_tr, d_ts, c_tr, c_ts = \
        do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts)

    # build stage 1 dataset using stage 0 results
    # NB predictions are concatenated to the quantitative attributes processed from the flows
    X_tr_full = np.hstack((X_tr, p_tr, d_tr, c_tr))
    X_ts_full = np.hstack((X_ts, p_ts, d_ts, c_ts))

    # perform final classification
    print("Performing Stage 1 classification ... ")
    pred = do_stage_1(X_tr_full, X_ts_full, Y_tr, Y_ts)

    # print classification report
    print(classification_report(Y_ts, pred, target_names=le.classes_))


if __name__ == "__main__":
    # parse cmdline args
    args = parse_args()
    main(args)