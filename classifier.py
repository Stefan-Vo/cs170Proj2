import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.Y_train = None

    def Train(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, x):
        distances = [self.euclidean_distance(x, xt) for xt in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

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

file_path = 'small-test-dataset-1.txt'
X, Y = load_data(file_path)

X = normalize_features(X)

knn = KNNClassifier(k=3)

#leave-one-out validation    
correct_predictions = 0
total_samples = len(X)
for i in range(total_samples):
    X_train = np.delete(X, i, axis=0)  
    Y_train = np.delete(Y, i)
    X_test = X[i].reshape(1, -1)  
    Y_test = Y[i]
    
    knn.Train(X_train, Y_train)
    prediction = knn.predict(X_test[0])
    if prediction == Y_test:
        correct_predictions += 1

accuracy = correct_predictions / float(total_samples)
print(f"Accuracy: {accuracy:.3f}")