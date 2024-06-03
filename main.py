from classifier import KNNClassifier
from validator  import Validator
import numpy as np

def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, 1:]  # Assuming the first column is the label and the rest are features
    Y = data[:, 0]   # Class labels are in the first columnass labels are in the first column
    return X, Y

def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized


def main():
    file_path = 'small-test-dataset-1.txt'
    X, Y = load_data(file_path)
    X = normalize_features(X)
    knn = KNNClassifier(k=3)

    validator = Validator(knn)
    feature_subset = [2, 4, 6] #We ignore 1st column so feature 3 should be in column 2
    accuracy = validator.validate(X, Y, feature_subset)
    print(f"Accuracy: {accuracy:.3f}")


main()