"""This module contains classes and functions for implementng
Decision Tree Classifier ML model."""


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Node:
    """Class for creating nodes of the final Decision Tree Classifier.
    """

    def __init__(self, X, y, gini, value=None):
        self.X = X
        self.y = y
        self.gini = gini
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.value = value


class MyDecisionTreeClassifier:
    """Class for creating a Decision Tree Classifier.
    """

    def __init__(self, max_depth) -> None:
        self.map_depth = max_depth

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Basically a wrapper for recursive function that creates and appends
        nodes to the tree. Trains the model.

        Args:
            X (np.ndarray): Sample values.
            y (np.ndarray): Target values.
        """
        self.root = self.build_tree(X, y)

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively create and append nodes to the Decision Tree Classifier
        with all necessary info.

        Args:
            X (np.ndarray): Sample values of current node.
            y (np.ndarray): Parent values of current node.
            depth (int, optional): Depth of current node. Defaults to 0.

        Returns:
            Node: Root node of the tree, with all architecture.
        """
        node = Node(
            X, y, self.gini(y),
            value=np.argmax([np.sum(y == i) for i in range(len(set(y)))])
        )

        if depth < self.map_depth:
            feature_index, threshold_value = self.split_data(X, y)

            if feature_index is not None:
                left_indexes = X[:, feature_index] <= threshold_value
                X_left, y_left = X[left_indexes], y[left_indexes]
                X_right, y_right = X[~left_indexes], y[~left_indexes]

                node.feature_index = feature_index
                node.threshold = threshold_value
                node.left = self.build_tree(X_left, y_left, depth+1)
                node.right = self.build_tree(X_right, y_right, depth+1)

        return node

    def split_data(self, X: np.ndarray, y: np.ndarray) -> tuple[int, int]:
        """Choose the best split. Return best feature and thresholder values.

        Args:
            X (np.ndarray): Sample values.
            y (np.ndarray): Target values.

        Returns:
            tuple[int, int]: Feature value and thresholder of the split.
        """
        if y.size < 2:
            return None, None

        gini_best = self.gini(y)
        feature_best, threshold_best = None, None

        num_features = X.shape[1]
        for feature_index in range(num_features):
            values = X[:, feature_index]
            thresholds = [(values[i-1]+values[i]) /
                          2 for i in range(1, len(values))]

            for threshold in thresholds:
                left_indexes = X[:, feature_index] <= threshold
                y_left = y[left_indexes]
                y_right = y[~left_indexes]

                gini_left = self.gini(y_left)
                gini_right = self.gini(y_right)
                gini_curr = (y_left.size/y.size)*gini_left + \
                    (y_right.size/y.size)*gini_right

                if gini_curr < gini_best:
                    gini_best = gini_curr
                    feature_best = feature_index
                    threshold_best = threshold

        return feature_best, threshold_best

    def gini(self, classes: np.ndarray) -> float:
        """Return Gini Impurity index of the node.

        Args:
            classes (ndarray): Target values of the node.

        Returns:
            float: Gini Impurity index.
        """
        count_each_class = [np.sum(classes == c)
                            for c in range(len(set(classes)))]
        gini = 1 - sum([(n/classes.size)**2 for n in count_each_class])
        return gini

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted values of incoming samples.

        Args:
            X (np.ndarray): Sample values.

        Returns:
            np.ndarray: Predicted classes.
        """
        prediction = [self._predict(x) for x in X]
        return np.array(prediction)

    def _predict(self, sample: list) -> int:
        """Return predicted value of a particular sample.

        Args:
            sample (list): Sample to predict value.

        Returns:
            int: Class of the sample.
        """
        node = self.root
        while node.left or node.right:
            if sample[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


def find_mean_accuracy(num_of_iter: int = 100, base: str = 'wine') -> float:
    """Find and return mean accuracy of the model on different datasets.

    Args:
        num_of_iter (int, optional): Number of iterations. Defaults to 100.
        base (str, optional): Dataset to test on. Defaults to 'wine'.

    Returns:
        float: Mean accuracy.
    """
    if base == 'wine':
        base = datasets.load_wine()
    elif base == 'breast_cancer':
        base = datasets.load_breast_cancer()
    elif base == 'iris':
        base = datasets.load_iris()

    my_dtc = MyDecisionTreeClassifier(4)

    total = 0
    for _ in tqdm(range(num_of_iter)):
        X, y = base.data, base.target
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.20)

        my_dtc.fit(X, y)

        predictions = my_dtc.predict(X_test)
        total += sum(predictions == y_test) / len(y_test)

    return total/num_of_iter


if __name__ == "__main__":
    base = datasets.load_wine()
    my_dtc = MyDecisionTreeClassifier(4)

    X, y = base.data, base.target
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.20)
    my_dtc.fit(X, y)

    predictions = my_dtc.predict(X_test)
    # print(find_mean_accuracy())
