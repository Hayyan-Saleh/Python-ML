import univariant_v1 as v1
from plot import plot

x_training, y_training = v1.get_training_examples()
w = 0
b = 0
alpha = 0.3

w, b, history = v1.gradient_descent(
    x_training, y_training, w, b, alpha, v1.cost_function
)

print(f"least cost: {history[-1][0]:.2f}")
v1.predict(w, b, 0.6)


# plot(x_training, y_training, w, b, alpha, history)

#! cost function values on v1:
# 1- 1686 on alpha 0.001
# 2- 972.65 on alpha 0.003
# 3- 583.52 on alpha 0.01
# 4- 355.69 on alpha 0.03
# 5- 339.77 on alpha 0.1
# 6- 339.77 on alpha 0.3
