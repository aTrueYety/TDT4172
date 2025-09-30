import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.0001, n_iters=10000):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # Initialize parameters
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias
            dw = (1/m) * (X.T @ (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        # Print final parameters
        print(f'Weights: {self.weights}, Bias: {self.bias}')
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        return X @ self.weights + self.bias

    def compute_error(self, X, y):
        """
        Computes the error.

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats

        Returns:
            A length m array of floats representing the error
        """
        y_pred = self.predict(X)
        error = y_pred - y
        return error



