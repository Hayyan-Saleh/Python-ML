import numpy as np
import pandas as pd


def get_data():
    df = pd.read_csv("purchase.csv")
    data = df.values
    X = data[:, 0:-1]
    y = data[:, -1]
    return X, y


def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std


def sigmoid(X, w, b):
    z = np.dot(X, w) + b
    return 1 / (1 + np.exp(-z))


def cost_func(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb = sigmoid(X[i], w, b)
        cost += -(y[i]) * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
    return cost / m


def get_derivatives(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0
    for i in range(m):
        f_wb = sigmoid(X[i], w, b)
        err = f_wb - y[i]
        for j in range(n):
            dj_dw[j] += (err) * X[i, j]
        dj_db += err
    return dj_dw / m, dj_db / m


def gradient_descent(X, y, alpha, iterations=10000):
    history = []
    w = np.zeros(X.shape[1])
    b = 0.0

    print(f"{'Iteration':<12} {'Cost':<15}")
    print("-" * 27)

    for i in range(iterations):
        dj_dw, dj_db = get_derivatives(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % (iterations // 10) == 0:
            cost = cost_func(X, y, w, b)
            history.append((i, cost))
            print(f"{i:<12} {cost:<15.6f}")

    return w, b, history


def predict(X, w, b):
    return (sigmoid(X, w, b) >= 0.5).astype(int)


def accuracy(p, y):
    correct_predictions = np.sum(p == y)

    m = y.shape[0]
    acc = correct_predictions / m

    return acc
