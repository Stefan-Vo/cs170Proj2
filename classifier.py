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





