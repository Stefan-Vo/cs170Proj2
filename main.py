import classifier
import validator 

def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, [3, 5, 7]]  # Using features 3, 5, and 7
    Y = data[:, 0]   # Class labels are in the first column
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

    accuracy = correct_predictions / float(total_samples)
    print(f"Accuracy: {accuracy:.3f}")