import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_features(X, y):
    df = pd.DataFrame(
        X, columns=["Age", "Income", "Hours Online", "Previous Purchases"]
    )
    df["Will Buy"] = y
    sns.pairplot(df, hue="Will Buy", markers=["o", "s"], palette=["red", "blue"])
    plt.show()
