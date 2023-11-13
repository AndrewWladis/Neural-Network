import math

import numpy as np
import matplotlib.pyplot as plt

# Regenerating the data points and adding Gaussian noise

# Define the line equation
def line(x, m, b):
    return m * x + b

# Parameters for the line
m = 2
b = 5

# Generate x-values
x = np.linspace(0, 10, 100)

# Compute y-values
y = line(x, m, b)

# Generate Gaussian noise
noise_std_dev = 2
noise = np.random.normal(0, noise_std_dev, x.shape)

# Add noise to y-values
y_noisy = y + noise


def gradient_descent(x, y, learning_rate=0.01, epochs=10000):
    # 1. Initialization
    m = np.random.randn()
    b = np.random.randn()

    N = len(x)

    # To store the history of loss values
    loss_history = []

    for epoch in range(epochs):
        # 2. Hypothesis
        y_pred = m * x + b

        # 3. Loss Function
        loss = (1 / N) * np.sum((y_pred - y) ** 2)
        loss_history.append(loss)

        # 4. Gradient Descent
        dm = -(2 / N) * np.sum(x * (y - y_pred))
        db = -(2 / N) * np.sum(y - y_pred)

        m = m - learning_rate * dm
        b = b - learning_rate * db

    return m, b, loss_history


# Parameters for gradient descent
learning_rate = 0.01
epochs = 1000

m_estimated, b_estimated, loss_history = gradient_descent(x, y_noisy, learning_rate, epochs)

# Run the gradient descent algorithm
m_estimated, b_estimated, loss_history = gradient_descent(x, y_noisy, learning_rate, epochs)

# Plotting the loss function convergence over iterations
plt.figure(figsize=(10,6))
plt.plot(loss_history)
plt.title('Convergence of Loss Function')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.show()

print(m_estimated, b_estimated)





# Oct 24
# y=mx+b
# configureable number of neurons
# 1st implement forward calculation
# put in an x and calculate a y
# numpy array of Weights, biases
#a = sigma(x*w1+b) w and b have arrays, but x is scalar
