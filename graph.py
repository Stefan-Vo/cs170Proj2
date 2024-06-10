import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from classifier import KNNClassifier
from validator import Validator

def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, 1:]  # Assuming the first column is the label and the rest are features
    Y = data[:, 0]   # Class labels are in the first column
    return X, Y

def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized

def leave_one_out_validator(X, Y, feature_subset):
    if not feature_subset:  # No features selected
        # Return the accuracy of a baseline classifier (e.g., predicting the majority class)
        majority_class = np.bincount(Y.astype(int)).argmax()
        predictions = np.full(Y.shape, majority_class)
        accuracy = np.mean(predictions == Y)
        return accuracy * 100  # Return accuracy as a percentage

    knn = KNNClassifier(k=3)
    validator = Validator(knn)
    accuracy = validator.validate(X, Y, feature_subset)
    return accuracy

def forwardElimination(X, Y):
    print("Forward Elimination")
    num_features = X.shape[1]
    features = [{i} for i in range(num_features)]
    size = len(features)
    origSize = size
    full_range = set(range(num_features))

    bestAcc = 0
    bestFeature = set()

    while size > 0:
        change = False
        newFeatures = []
        for feature in features:
            feature_subset = list(feature)
            currentAcc = leave_one_out_validator(X, Y, feature_subset)
            print(f"        Using feature(s) {[f+1 for f in feature]} accuracy is {currentAcc:.3f}%")
            if currentAcc >= bestAcc:
                change = True
                bestAcc = currentAcc
                bestFeature = feature
        print("")
        if change:
            print(f"Feature set {[f+1 for f in bestFeature]} was best, accuracy is {bestAcc:.3f}%\n")
        else:
            print("(Warning, Accuracy has decreased!)")
            features = bestFeature
            break
        if len(bestFeature) == origSize:
            features = bestFeature
            break
        missing = full_range - bestFeature
        for number in missing:
            new_test = bestFeature.copy()
            new_test.add(number)
            newFeatures.append(new_test)
        features = newFeatures
        size -= 1
    print(f"Finished search!! The best feature subset is {[f+1 for f in bestFeature]}, which has an accuracy of {bestAcc:.3f}%")
    return list(bestFeature)

def backwardsElimination(X, Y):
    print("Backward Elimination")
    num_features = X.shape[1]
    features = set(range(num_features))
    features = [features]
    origSize = num_features
    bestAcc = 0
    bestFeature = set()
    while num_features > 0:
        change = False
        for feature in features:
            feature_subset = list(feature)
            currentAcc = leave_one_out_validator(X, Y, feature_subset)
            print(f"        Using feature(s) {[f+1 for f in feature]} accuracy is {currentAcc:.3f}%")
            if currentAcc > bestAcc:
                change = True
                bestAcc = currentAcc
                bestFeature = feature
        print("")
        if change:
            print(f"Feature set {[f+1 for f in bestFeature]} was best, accuracy is {bestAcc:.3f}%\n")
        else:
            print("(Warning, Accuracy has decreased!)")
            features = bestFeature
            break
        if len(bestFeature) == 1:
            features = bestFeature
            break
        features = [set(combo) for combo in itertools.combinations(bestFeature, len(bestFeature) - 1)]
        num_features -= 1
    print(f"Finished search!! The best feature subset is {[f+1 for f in bestFeature]}, which has an accuracy of {bestAcc:.3f}%")
    return list(bestFeature)

def plot_features(X, Y, feature_subset):
    if len(feature_subset) == 1:
        plt.scatter(X[:, feature_subset[0]], Y)
        plt.xlabel(f'Feature {feature_subset[0] + 1}')
        plt.ylabel('Class Label')
        plt.title('Feature vs Class Label')
    elif len(feature_subset) == 2:
        plt.scatter(X[:, feature_subset[0]], X[:, feature_subset[1]], c=Y)
        plt.xlabel(f'Feature {feature_subset[0] + 1}')
        plt.ylabel(f'Feature {feature_subset[1] + 1}')
        plt.title('Feature Scatter Plot')
    else:
        # Using PCA to reduce to 2 dimensions for visualization
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X[:, feature_subset])
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of Selected Features')
    plt.show()

def main():
    print("Welcome to Group 8's Feature Selection Algorithm\n")

    num = int(input("Please enter the total number of features: "))

    print("\nType the number of the algorithm you want to run.")
    print("1: Forward Selection")
    print("2: Backward Elimination")

    alg = input()

    file_path = 'small-test-dataset-1.txt'
    X, Y = load_data(file_path)
    X = normalize_features(X)

    # Print accuracy with no features
    no_feature_accuracy = leave_one_out_validator(X, Y, [])
    print(f"\nUsing no features: {no_feature_accuracy:.3f}%\n")

    if alg == '1':
        best_features = forwardElimination(X, Y)
    elif alg == '2':
        best_features = backwardsElimination(X, Y)
    
    if best_features:
        plot_features(X, Y, best_features)

# Call the main function
main()