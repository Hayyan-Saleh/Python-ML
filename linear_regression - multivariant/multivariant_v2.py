import numpy as np
import pandas as pd


def get_data():
    df = pd.read_csv("prices.csv")
    data = df.values
    return data[:, :-1], df["price"].values


def normalize(X, option=0):
    match option:
        case 1:
            # min-max normaliztation
            min = X.min(axis=0)
            max = X.max(axis=0)
            normalized_X = (X - min) / (max - min)
            return normalized_X, min, max
        case 2:
            # mean normaliztation
            min = X.min(axis=0)
            max = X.max(axis=0)
            mean = X.mean(axis=0)
            normalized_X = (X - mean) / (max - min)
            return normalized_X, mean, min, max
        case 3:
            # z-score normalization
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            normalized_X = (X - mean) / (std)
            return normalized_X, mean, std
        case 0:
            return X


def cost_function(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        cost += ((np.dot(w, x[i]) + b) - y[i]) ** 2
    return cost / (2 * m)


def get_derivatives(x, y, w, b):
    m = x.shape[0]
    n = x.shape[1]
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        err = np.dot(w, x[i])
        for j in range(n):
            dj_dw[j] += ((err + b) - y[i]) * x[i][j]
        dj_db += (err + b) - y[i]
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, cost_function):
    w = w_in
    b = b_in

    history = []  # to store cost function changes

    i = 0
    while i < 1000:
        dj_dw, dj_db = get_derivatives(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i % 100 == 0:
            history.append((cost_function(x, y, w, b), i))
        i += 1
    return w, b, history


def predict(w, b, x, norm_x):
    print("===============")
    print(
        f"Predicting a house price for {x[0]} square feet, {x[1]} number of bedrooms, {x[2]} number of floors and an age of {x[3]}: {(np.dot(w, norm_x) + b):.2f} K $"
    )
    print("===============")
