import numpy as np
import matplotlib.pyplot as plt

# Given data points
x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([1, 2, 3, 4, 5, 6, 7])

# Hypothesis function hθ(x) = θ0 + θ1 * x
# Assume θ0 = 0, so hθ(x) = θ1 * x

# Define cost function J(θ1)
def cost_function(theta1, x, y):
    m = len(y)  # Number of training examples
    predictions = theta1 * x
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Testing for different values of θ1
theta1_values = np.linspace(-0.5, 2.25, 20)  # Same as given in the image
cost_values = [cost_function(theta1, x, y) for theta1 in theta1_values]

# Plot cost function vs θ1 values
plt.figure(figsize=(8, 5))
plt.plot(theta1_values, cost_values, marker='o', linestyle='-')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'Cost $J(\theta_1)$')
plt.title('Cost Function vs Theta1')
plt.grid()
plt.show()
