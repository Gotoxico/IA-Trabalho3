import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

class Adaline():
    def __init__(self, n_features, learning_rate=0.0025, tol=1e-6, max_epochs=2000, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.weights = [random.uniform(0, 1) for _ in range(n_features + 1)]
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_epochs = max_epochs

    def summation(self, x):
        s = 0.0
        for i in range(len(x)):
            s += self.weights[i] * x[i]
        return s

    def calculate_mse(self, X, y):
        mse = 0.0
        for xi, yi in zip(X, y):
            u = self.summation(xi)
            mse += (yi - u) ** 2
        return mse / len(X)

    def train(self, X, y):
        epochs = 0
        prev_mse = float("inf")
        cur_mse = self.calculate_mse(X, y)

        while abs(prev_mse - cur_mse) > self.tol and epochs < self.max_epochs:
            prev_mse = cur_mse

            for xi, yi in zip(X, y):
                u = self.summation(xi)
                error = yi - u
                for j in range(len(xi)):
                    self.weights[j] += self.learning_rate * error * xi[j]

            cur_mse = self.calculate_mse(X, y)
            epochs += 1

        return epochs, cur_mse

    def predict_raw(self, X):
        return [self.summation(xi) for xi in X]

    def predict(self, X):
        return [1 if u >= 0 else -1 for u in self.predict_raw(X)]


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y_full = iris.target

    classes = (1, 2)
    mask = np.isin(y_full, classes)
    X = X[mask]
    y_int = y_full[mask]

    y = np.where(y_int == classes[0], -1, 1)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    ones = np.ones((X_std.shape[0], 1))
    X_with_bias = np.hstack([ones, X_std])

    training_inputs = X_with_bias.tolist()
    training_outputs = y.tolist()

    adaline = Adaline(n_features=4, learning_rate=0.0025, tol=1e-6, max_epochs=5000, seed=42)

    print(f"Vetor de pesos inicial: {adaline.weights}\n")

    epochs, final_mse = adaline.train(training_inputs, training_outputs)

    print(f"Vetor de pesos ajustados: {adaline.weights}")
    print(f"Épocas até parada: {epochs}")
    print(f"MSE final: {final_mse:.6f}")

    y_pred = adaline.predict(training_inputs)
    acc = (np.array(y_pred) == np.array(training_outputs)).mean()
    print(f"Acurácia: {acc*100:.2f}%")
