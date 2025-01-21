import numpy as np

class KNNClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        """Compute the Euclidean distance between two points for KNN"""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """
        Predict the labels for the input data.
        """
        X = np.array(X)
        predictions = []

        for x in X:
            # Compute distances from x to all training points
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

            # Get the indices of the k nearest neighbors
            nn_indices = np.argsort(distances)[:self.n_neighbors]

            # Get the labels of the nearest neighbors
            nn_labels = self.y_train[nn_indices]

            # Determine the most common label among the nearest neighbors
            unique, counts = np.unique(nn_labels, return_counts=True)
            majority_label = unique[np.argmax(counts)]

            predictions.append(majority_label)

        return np.array(predictions)

# Example usage:
# if __name__ == "__main__":
#     # Create some dummy data
#     X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]])
#     y_train = np.array([0, 0, 0, 1, 1, 1])
#
#     X_test = np.array([[2, 3], [7, 8]])
#
#     knn = KNNClassifier(n_neighbors=3)
#     knn.fit(X_train, y_train)
#     predictions = knn.predict(X_test)
#     print("Predictions:", predictions)
