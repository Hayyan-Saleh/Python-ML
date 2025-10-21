import numpy as np
import matplotlib.pyplot as plt


def plot(x, y, w, b, alpha, history):
    fig, ax = plt.subplots(1, 2)

    ax[0].scatter(x, y, color="red", marker="x", s=20)

    x_line = np.linspace(x.min(), x.max(), 2)
    y_line = [w * x + b for x in x_line]

    ax[0].plot(x_line, y_line, color="cyan", alpha=0.5, linewidth=3)

    ax[0].set_title("Actual Model (Predicting House Prices)")
    ax[0].set_xlabel("Size (in 1000 sqft)")
    ax[0].set_ylabel("Price (in 1000 USD)")

    costs, iters = zip(*history)
    ax[1].plot(iters, costs, color="pink", linewidth=5)

    ax[1].set_title(f"Cost Function (on alpha = {alpha})")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("J(w,b)")

    plt.show()
