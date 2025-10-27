import numpy as np
import matplotlib.pyplot as plt


def plot_features(X, y, pred_y, history, alpha):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    feature_names = ["Fuel (kg)", "Thrust angle (deg)"]
    for i in range(len(feature_names)):
        ax[i].scatter(X[:, i], y, marker="x", c="red", s=20, label="Actual")
        sorted_indices = np.argsort(X[:, i])
        ax[i].plot(
            X[sorted_indices, i],
            pred_y[sorted_indices],
            c="cyan",
            linewidth=2,
            label="Predicted",
        )
        ax[i].set_ylabel("Altitude (m)")
        ax[i].set_xlabel(feature_names[i])
        ax[i].legend()

    costs, iters = zip(*history)
    ax[2].plot(iters, costs, color="pink", linewidth=5)

    ax[2].set_title(f"Cost Function (on alpha = {alpha})")
    ax[2].set_xlabel("Iterations")
    ax[2].set_ylabel("J(w,b)")

    fig.tight_layout()
    plt.show()
