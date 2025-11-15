import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

class KMeansClustering():
    def __init__(self, n_clusters=2, random_state=42, tol=0.00001, max_iter=200):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter

        random.seed(random_state)

    def fit(self, X):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_num = X[numeric_cols]

        centroids = pd.DataFrame([
            {col: np.random.uniform(X_num[col].min(), X_num[col].max()) for col in X_num.columns}
            for _ in range(self.n_clusters)
        ])

        error = 2 * self.tol
        iter = 0
        while error > self.tol and iter < self.max_iter:
            labels = []
            for _, obs in X_num.iterrows():
                distances = []
                for _, centroid in centroids.iterrows():
                    dist_centroid = euclidean_distance(centroid.values, obs.values)
                    distances.append(dist_centroid)
                label = np.argmin(distances)
                labels.append(label)

            new_centroids = []
            for c in range(self.n_clusters):
                c_obs = X_num.iloc[[i for i, l in enumerate(labels) if l == c]]
                if len(c_obs) == 0:
                    new_centroids.append(centroids.iloc[c])
                else:
                    new_centroids.append(c_obs.mean(axis=0))

            new_centroids_df = pd.DataFrame(new_centroids)
            error = np.linalg.norm(new_centroids_df.values - centroids.values)
            centroids = new_centroids_df
            iter += 1

        self.centroids = centroids
        self.labels = labels

    def predict(self, X):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_num = X[numeric_cols]

        labels = []
        for _, obs in X_num.iterrows():
            distances = []
            for _, centroid in self.centroids.iterrows():
                dist_centroid = euclidean_distance(centroid.values, obs.values)
                distances.append(dist_centroid)

            label = np.argmin(distances)
            labels.append(label)

        return labels

if __name__ == '__main__':
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names).iloc[:, [0, 1]]

    iris_df.info()

    model = KMeansClustering(n_clusters=3, random_state=42)

    model.fit(iris_df)

    labels = model.labels
    centroids = model.centroids

    colors = ['#ff0000', '#00ff00', '#0000ff']

    plt.figure()

    # Plot centroids
    plt.scatter(centroids.iloc[:, 0], centroids.iloc[:, 1], marker='x')

    for c in range(3):
        c_obs = iris_df.iloc[[i for i, l in enumerate(labels) if l == c]]
        plt.scatter(c_obs.iloc[:, 0], c_obs.iloc[:, 1], color=colors[c])

    plt.title("Clusters")
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")

    plt.show()
