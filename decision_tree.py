import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from collections import Counter

def entropy(y):
    counts = np.bincount(y)
    probs = counts[np.nonzero(counts)] / len(y)
    return -np.sum(probs * np.log2(probs))

def information_gain(y, X_column, split_value):
    parent_entropy = entropy(y)
    
    left_mask = X_column <= split_value
    right_mask = X_column > split_value
    
    if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
        return 0
    
    n = len(y)
    n_left, n_right = len(y[left_mask]), len(y[right_mask])
    e_left, e_right = entropy(y[left_mask]), entropy(y[right_mask])
    
    child_entropy = (n_left/n)*e_left + (n_right/n)*e_right
    return parent_entropy - child_entropy

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X, y, depth=0, max_depth=3):
    if len(set(y)) == 1 or depth >= max_depth:
        return Node(value=Counter(y).most_common(1)[0][0])
    
    best_gain = 0
    best_feature = None
    best_threshold = None
    
    for feature in X.columns:
        values = X[feature].unique()
        for threshold in values:
            gain = information_gain(y, X[feature], threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    
    if best_gain == 0:
        return Node(value=Counter(y).most_common(1)[0][0])
    
    left_mask = X[best_feature] <= best_threshold
    right_mask = X[best_feature] > best_threshold
    
    left = build_tree(X[left_mask], y[left_mask], depth+1, max_depth)
    right = build_tree(X[right_mask], y[right_mask], depth+1, max_depth)
    
    return Node(best_feature, best_threshold, left, right)

def predict_one(x, node):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_one(x, node.left)
    else:
        return predict_one(x, node.right)

def predict(X, tree):
    return [predict_one(x, tree) for _, x in X.iterrows()]

if __name__ == "__main__":
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")

    data = X.copy()
    data['target'] = y

    tree = build_tree(X, y, max_depth=3)
    y_pred = predict(X, tree)

    accuracy = np.mean(y_pred == y)
    print(f"Acurácia: {accuracy:.2f}")
