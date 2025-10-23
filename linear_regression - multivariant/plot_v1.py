import numpy as np
import matplotlib.pyplot as plt


def plot(x, y, w, b, alpha, history):
    fig, ax = plt.subplots(1, 5)

    feature_names = ["Size (1000 sqft)", "Bedrooms", "Floors", "Age"]

    # Plot each feature with regression line (first 3 subplots)
    for i in range(len(feature_names)):
        ax[i].scatter(x[:, i], y, color="red", marker="x", s=20)

        # Create line for this feature (holding others at their mean)
        x_line = np.linspace(x[:, i].min(), x[:, i].max(), 2)
        x_mean = np.mean(x, axis=0)

        y_line = []
        for val in x_line:
            x_temp = x_mean.copy()
            x_temp[i] = val
            y_line.append(np.dot(w, x_temp) + b)

        ax[i].plot(x_line, y_line, color="cyan", linewidth=2)
        ax[i].set_xlabel(feature_names[i])
        ax[i].set_ylabel("Price (1000 USD)")

    # Plot cost function (4th subplot)
    costs, iters = zip(*history)
    ax[3].plot(iters, costs, color="pink", linewidth=5)
    ax[3].set_title(f"Cost (α={alpha})")
    ax[3].set_xlabel("Iterations")
    ax[3].set_ylabel("J(w,b)")

    plt.tight_layout(rect=[0, 0.93, 1, 0.05])
    plt.show()
