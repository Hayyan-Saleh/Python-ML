import matplotlib.pyplot as plt
from matplotlib import style


def plot_features(X, norm_X, y):
    style.use("fivethirtyeight")
    fig, ax = plt.subplots(2, 4, figsize=(12, 4), sharey=True)

    feature_names = ["Size (1000 sqft)", "Bedrooms", "Floors", "Age"]

    ax[0][0].set_ylabel("Price (1000 USD)")
    # Plot each feature with regression line (first 3 subplots)
    for i in range(len(feature_names)):
        ax[0][i].scatter(X[:, i], y, color="red", marker="x", s=20)

    ax[1][0].set_ylabel("Price (1000 USD)")
    # Plot each feature with regression line (first 3 subplots)
    for i in range(len(feature_names)):
        ax[1][i].scatter(norm_X[:, i], y, color="red", marker="x", s=20)
        ax[1][i].set_xlabel(feature_names[i])

    plt.tight_layout()
    plt.show()


def plot_predictions(X, y, pred_y, history, alpha):
    style.use("ggplot")
    fig, ax = plt.subplots(
        1,
        5,
        figsize=(12, 4),
    )

    feature_names = ["Size (1000 sqft)", "Bedrooms", "Floors", "Age"]

    ax[0].set_ylabel("Price (1000 USD)")
    # Plot each feature with corresponded predictions respectively
    for i in range(len(feature_names)):
        ax[i].scatter(X[:, i], y, color="red", marker="x", s=20)
        ax[i].scatter(X[:, i], pred_y, color="cyan", alpha=0.8, s=10)

    # Plot cost function (5th subplot)
    costs, iters = zip(*history)
    ax[4].plot(iters, costs, color="pink", linewidth=5)
    ax[4].set_title(f"Cost (α={alpha})")
    ax[4].set_xlabel("Iterations")
    ax[4].set_ylabel("J(w,b)")

    plt.tight_layout()
    plt.show()
