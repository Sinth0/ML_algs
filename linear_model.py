# Practice example for a linear model using gradient descent.
# This is not optimised (esp. not performance wise), just my first best implementation of a linear model

import numpy as np
import matplotlib.pyplot as plt

class linear_model():
    
    def __init__(self, weights=None):
        self.weights = weights

    def fit(self, X, Y, lr, epochs):
        # initial weights = 0
        self.weights = np.zeros(X.shape[1] + 1)
        # add 1-column corresponding to intercept
        X_w = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        for i in range(epochs):
            Y_pred = X_w.dot(self.weights)
            # derivatives of MSE
            D_w0 = (-2/Y.shape[0]) * sum(Y - Y_pred)
            D_w = (-2/Y.shape[0]) * X.T.dot(Y - Y_pred)
            # update weights
            self.weights[0] -= lr * D_w0
            self.weights[1:] -= lr * D_w

    def predict(self, x):
        intercept = np.ones(x.shape[0])
        x_w = np.c_[intercept.T, x]
        return x_w.dot(self.weights)

# generate data
n_samples = 100
mu = np.array([5.0, 5.0, 5.0])
r = np.array([
        [  3.40, -2.75, -2.00],
        [ -2.75,  5.50,  1.50],
        [ -2.00,  1.50,  1.25]
    ])

rng = np.random.default_rng(seed=1)
data = rng.multivariate_normal(mu, r, size=n_samples)

X = data[:,:-1]
Y = data[:,-1]

# prediction values
x = np.array([[1,1],[4,5]])

lin_model = linear_model()
lin_model.fit(X,Y,0.0001,10000)
prediction = lin_model.predict(x)


print(f"Prediction for (1,1) and (4,5): {prediction}")