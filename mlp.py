import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def one_hot(y, num_classes):
    Y = np.zeros((y.size, num_classes))
    Y[np.arange(y.size), y] = 1.0
    return Y

def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).mean()

class MLP:
    def __init__(self, input_size, hidden_size, output_size,
                 learning_rate=0.1, epochs=5000, tol=1e-5, seed=42, verbose=True):
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol
        self.verbose = verbose

        self.W1 = rng.normal(0, 0.1, size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, 0.1, size=(hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

        self._last_loss = np.inf

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1.0 - a)

    @staticmethod
    def softmax(z):
        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shift)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def cross_entropy(y_true_onehot, y_pred_proba, eps=1e-12):
        y_pred_proba = np.clip(y_pred_proba, eps, 1.0 - eps)
        return -np.mean(np.sum(y_true_onehot * np.log(y_pred_proba), axis=1))

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = self.softmax(Z2)
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def backward(self, cache, y_true_onehot):
        X, A1, A2 = cache["X"], cache["A1"], cache["A2"]
        N = X.shape[0]
        dZ2 = (A2 - y_true_onehot) / N
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.sigmoid_derivative(A1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y_onehot):
        for epoch in range(1, self.epochs + 1):
            y_pred, cache = self.forward(X)
            loss = self.cross_entropy(y_onehot, y_pred)

            dW1, db1, dW2, db2 = self.backward(cache, y_onehot)
            self.update(dW1, db1, dW2, db2)

            if self.verbose and epoch % 500 == 0:
                print(f"Época {epoch:5d} | Loss: {loss:.6f}")

            if abs(self._last_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Parando cedo na época {epoch} (delta loss < {self.tol})")
                break
            self._last_loss = loss

        return epoch, loss

    def predict_proba(self, X):
        y_pred, _ = self.forward(X)
        return y_pred

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target
    num_classes = len(np.unique(y))

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_std, y, test_size=0.3, random_state=42, stratify=y
    )

    Y_train = one_hot(y_train, num_classes)
    Y_test = one_hot(y_test, num_classes)

    mlp = MLP(input_size=4, hidden_size=8, output_size=3,
              learning_rate=0.1, epochs=10000, tol=1e-6, seed=42, verbose=True)

    epochs, final_loss = mlp.train(X_train, Y_train)
    print(f"\nTreino finalizado em {epochs} épocas | Loss final: {final_loss:.6f}")

    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    print(f"Acurácia (treino): {acc_train*100:.2f}%")
    print(f"Acurácia (teste):  {acc_test*100:.2f}%")
