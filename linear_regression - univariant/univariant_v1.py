import numpy as np


def get_training_examples():
    x_training = np.array([2, 1.3, 0.7, 0.9, 1.5])  # house sizes in 1000 sqft
    y_training = np.array([600, 360, 200, 300, 400])  # house sizes in 1000 USD
    return x_training, y_training


def cost_function(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        cost += ((w * x[i] + b) - y[i]) ** 2
    return cost / (2 * m)


def get_derivatives(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        dj_dw += ((w * x[i] + b) - y[i]) * x[i]
        dj_db += (w * x[i] + b) - y[i]
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, cost_function):
    w = w_in
    b = b_in

    history = []  # to store cost function changes

    i = 3
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
        f"Predicting a house price for {x} thousand square feet: {(w * x + b):.2f} K $"
    )
    print("===============")
