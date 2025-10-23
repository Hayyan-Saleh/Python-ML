import numpy as np
import multivariant_v2 as v2
import plot_v2 as pv2

# np.set_printoptions(precision=2)

X_training, y_training = v2.get_data()

normalized_X, mean, std = v2.normalize(X_training, 3)

# ? plot to see range difference between original and noramlized features
pv2.plot_features(X_training, normalized_X, y_training)


w = np.zeros(normalized_X.shape[1])
b = 0
alpha = 0.3

w, b, history = v2.gradient_descent(
    normalized_X, y_training, w, b, alpha, v2.cost_function
)


# ? plot to see the prediction against actual training examples
pred_y = np.dot((normalized_X), w) + b

pv2.plot_predictions(X_training, y_training, pred_y, history, alpha)


print(f"least cost: {history[-1][0]:.2f} on alpha {alpha}")

prediction_input = np.array([1450, 2, 3, 15])
norm_pred_input = (prediction_input - mean) / std

v2.predict(
    w,
    b,
    prediction_input,
    norm_pred_input,
)

#! ========= v1 main code =========
# x_training, y_training = v1.get_training_examples()
# w = np.zeros(x_training.shape[1])
# b = 0
# alpha = 0.000001

# w, b, history = v1.gradient_descent(
#     x_training, y_training, w, b, alpha, v1.cost_function
# )

# print(f"least cost: {history[-1][0]:.2f} on alpha {alpha}")
# v1.predict(
#     w,
#     b,
#     [2000, 4, 2, 12],
# )


# plot(x_training, y_training, w, b, alpha, history)

#! cost function values on v1:
# 1- 344.10 on alpha 0.000001 (when trying to increase alpha the was overflow due to large range of values for feature 1 )

#! cost function values on v2

# ? (min-max normalization):

# 1- 4586.81 on alpha 0.001
# 2- 676.50 on alpha 0.003
# 3- 324.00 on alpha 0.01
# 4- 232.00 on alpha 0.03
# 5- 122.50 on alpha 0.1
# 6- 39.81 on alpha 0.3
# 7- 16.33 on alpha 1 (this means the fault range is only 4k )

# ? (mean normalization):
# 1- 132.39 on alpha 0.1
# 2- 41.29 on alpha 0.3
# 2- 16.32 on alpha 1

# ? (z-score normalization):
# 1- 25.25 on alpha 0.03
# 2- 16.20 on alpha 0.1
# 2- 16.19 on alpha 0.3
