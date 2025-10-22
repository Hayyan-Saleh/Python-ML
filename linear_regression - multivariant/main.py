import numpy as np
import multivariant_v1 as v1
from plot import plot

x_training, y_training = v1.get_training_examples()
w = np.zeros(x_training.shape[1])
b = 0
alpha = 0.000001

w, b, history = v1.gradient_descent(
    x_training, y_training, w, b, alpha, v1.cost_function
)

print(f"least cost: {history[-1][0]:.2f} on alpha {alpha}")
v1.predict(w, b, [2000, 4, 2])


plot(x_training, y_training, w, b, alpha, history)

#! cost function values on v1:
# 1- 344.10 on alpha 0.000001 (when trying to increase alpha the was overflow due to large range of values for feature 1 )
