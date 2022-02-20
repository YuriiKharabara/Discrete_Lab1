import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Node:
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
    def __init__(self, max_depth) -> None:
        self.map_depth = max_depth

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        node = Node(
            X, y, self.gini(X, y),
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

    def split_data(self, X, y):
        if y.size < 2:
            return None, None

        gini_best = self.gini(X, y)
        feature_best, threshold_best = None, None

        num_features = X.shape[1]
        for feature_index in range(num_features):
            values = X[:, feature_index]
            thresholds = [(values[i-1]+values[i]) /
                          2 for i in range(1, len(values))]

            for threshold in thresholds:
                left_indexes = X[:, feature_index] <= threshold
                X_left, y_left = X[left_indexes], y[left_indexes]
                X_right, y_right = X[~left_indexes], y[~left_indexes]

                gini_left = self.gini(X_left, y_left)
                gini_right = self.gini(X_right, y_right)
                gini_curr = (y_left.size/y.size)*gini_left + \
                    (y_right.size/y.size)*gini_right

                if gini_curr < gini_best:
                    gini_best = gini_curr
                    feature_best = feature_index
                    threshold_best = threshold

        return feature_best, threshold_best

    def gini(self, groups, classes):
        count_each_class = [np.sum(classes == c)
                            for c in range(len(set(classes)))]
        gini = 1 - sum([(n/classes.size)**2 for n in count_each_class])
        return gini

    def predict(self, X):
        prediction = [self._predict(x) for x in X]
        return np.array(prediction)

    def _predict(self, sample):
        node = self.root
        while node.left:
            if sample[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


def find_mean_accuracy(num_of_iter: int = 100, base: str = 'wine'):
    if base == 'wine':
        base = datasets.load_wine()
    elif base == 'breast_cancer':
        base = datasets.load_breast_cancer()
    elif base == 'iris':
        base = datasets.load_iris()

    total = 0
    for _ in tqdm(range(num_of_iter)):
        X, y = base.data, base.target
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.20)

        my_dtc.fit(X, y)

        predictions = my_dtc.predict(X_test)
        total += sum(predictions == y_test) / len(y_test)

    return total/num_of_iter


if __name__ == "__main__":
    my_dtc = MyDecisionTreeClassifier(4)

    print(find_mean_accuracy(100))
