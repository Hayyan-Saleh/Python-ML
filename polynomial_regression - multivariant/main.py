import numpy as np
import poly_v1 as v1
import plot_v1 as p_v1

X, poly_X, y = v1.get_data()  # non-linear data set

norm_poly_X, x_std, x_mean = v1.normalize(poly_X)
norm_y, y_std, y_mean = v1.normalize(y)

alpha = 0.1

w, b, history = v1.gradient_descent(norm_poly_X, norm_y, alpha, iterations=50000)

# Denormalize predictions for plotting
norm_pred_y = np.dot(norm_poly_X, w) + b
pred_y = norm_pred_y * y_std + y_mean


p_v1.plot_features(X, y, pred_y, history, alpha)

test_point = np.array([8.56, 16.77])
v1.predict(test_point, x_mean, y_mean, x_std, y_std, w, b)
