import numpy as np

class Validator:
    def __init__(self, classifier):
        self.classifier = classifier

    def leave_one_out_validation(self, X, Y):
        correct_predictions = 0
        total_samples = len(X)

        for i in range(total_samples):
            X_train = np.delete(X, i, axis=0)
            Y_train = np.delete(Y, i)
            X_test = X[i].reshape(1, -1)
            Y_test = Y[i]

            self.classifier.Train(X_train, Y_train)
            prediction = self.classifier.predict(X_test[0])

            if prediction == Y_test:
                correct_predictions += 1

        accuracy = correct_predictions / float(total_samples)
        return accuracy

    def validate(self, X, Y, feature_subset):
        X_subset = X[:, feature_subset]
        return self.leave_one_out_validation(X_subset, Y)