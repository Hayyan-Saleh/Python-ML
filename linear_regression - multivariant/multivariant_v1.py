import numpy as np


def get_training_examples():
    x_training = np.array(
        [
            [2000, 4, 2, 12],
            [1300, 2, 1, 58],
            [700, 2, 1, 93],
            [900, 3, 2, 18],
            [1500, 3, 1, 30],
        ]
    )  # house sizes in  sqft, house bedrooms number, house floors number
    y_training = np.array([600, 360, 200, 300, 400])  # house prices in 1000 USD
    return x_training, y_training


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


def predict(w, b, x):
    print("===============")
    print(
        f"Predicting a house price for {x[0]} square feet, {x[1]} number of bedrooms and {x[2]} number of floors: {(np.dot(w, x) + b):.2f} K $"
    )
    print("===============")
