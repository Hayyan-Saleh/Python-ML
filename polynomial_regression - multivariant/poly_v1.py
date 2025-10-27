import numpy as np
import pandas as pd


def get_data():
    df = pd.read_csv("rocket_data.csv")
    data = df.values
    X = data[:, :-1]

    x0_e2 = np.array([x**2 for x in data[:, 0]])
    x0_e3 = np.array([x**3 for x in data[:, 0]])

    x1_e2 = np.array([x**2 for x in data[:, 1]])
    x1_e3 = np.array([x**3 for x in data[:, 1]])
    x1_e4 = np.array([x**4 for x in data[:, 1]])

    # NEW: Interaction term x0 * x1
    x0_x1 = data[:, 0] * data[:, 1]
    poly_X = np.c_[X, x0_e2, x0_e3, x1_e2, x1_e3, x1_e4, x0_x1]

    y = df["altitude_m"].values

    return X, poly_X, y


def normalize(X):
    mean = X.mean() if X.ndim == 1 else X.mean(axis=0)
    std = X.std() if X.ndim == 1 else X.std(axis=0)
    norm_X = (X - mean) / std
    return norm_X, std, mean


def cost_function(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        cost += ((np.dot(w, x[i]) + b) - y[i]) ** 2
    return cost / (2 * m)


def get_derivatives(X, y, w, b):
    m = X.shape[0]
    n = X.shape[1]
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        err = np.dot(w, X[i]) + b
        for j in range(n):
            dj_dw[j] += (err - y[i]) * X[i][j]
        dj_db += err - y[i]
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(X, y, alpha, iterations=1000):
    w = np.zeros(X.shape[1])
    b = 0

    history = []

    i = 0
    while i < iterations:
        dj_dw, dj_db = get_derivatives(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i % (iterations / 10) == 0:
            history.append((cost_function(X, y, w, b), i))
        i += 1
    return w, b, history


def predict(test_point, x_mean, y_mean, x_std, y_std, w, b):
    test_poly = np.array(
        [
            test_point[0],
            test_point[1],
            test_point[0] ** 2,
            test_point[0] ** 3,
            test_point[1] ** 2,
            test_point[1] ** 3,
            test_point[1] ** 4,
            test_point[0] * test_point[1],
        ]
    )
    norm_pred = np.dot(w, (test_poly - x_mean) / x_std) + b
    pred = norm_pred * y_std + y_mean
    print(f"prediction for {test_point} is: {pred:.2f}")
    print("===============")
    print("weights are: ")
    print(w)
    print("===============")
