import numpy as np


class GradientDescent:
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
      
    def compute_loss(self, y, y_pred):
        epsilon = 1e-15  # to avoid log(0)
        m = len(y)
        loss = - (1/m) * np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        return loss
    
    def compute_accuracy(self, y, y_pred):
        predictions = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        return accuracy

    def __init__(self, learning_rate=0.0001, n_iters=10000, actuation=sigmoid):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.actuation = actuation

    def fit(self, X, y):
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        X = (X - X.mean()) / X.std() # Standardize features
        
        # Initialize parameters
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        self.losses, self.accuracies = [], []

        # Gradient descent
        for _ in range(self.n_iters):
            linear_pred = X @ self.weights + self.bias
            y_pred = self.actuation(linear_pred)
            dw = (1 / m) * (X.T @ (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            self.losses.append(self.compute_loss(y, y_pred))
            self.accuracies.append(self.compute_accuracy(y, y_pred))
            
        # Print final parameters
        print(f'Weights: {self.weights}, Bias: {self.bias}')

    def predict(self, X):
        return self.actuation(X @ self.weights + self.bias) >= 0.5
      
    def predict_proba(self, X):
        return self.actuation(X @ self.weights + self.bias)
    
    def compute_error(self, X, y):
        y_pred = self.actuation(X @ self.weights + self.bias)
        return y_pred - y
