import logistic_regression as lr
from plot import plot_features

X, y = lr.get_data()
X_norm, mean, std = lr.normalize(X)

# plot_features(X_norm, y) # Commented out plotting for brevity

w, b, history = lr.gradient_descent(X_norm, y, 0.1)

p = lr.predict(X_norm, w, b)

accuracy = lr.accuracy(p, y) * 100

print(f"The percentage of fitting (Accuracy) is: {accuracy:.2f}%")
