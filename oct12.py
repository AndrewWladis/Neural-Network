import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 50, 100)
y = 2 * x + 1
mu = 0  # Mean of the Gaussian distribution
sigma = 10  # Standard deviation of the Gaussian distribution
noise = np.random.normal(mu, sigma, len(x))
noisy_y = y + noise # noise is error
# calculate cost from a single point
# use rand values for m and b
# cost(m, b) = (y - (mx + b))^2 as little as possible sum for every point

# 1. Calc cost for current m and b
# 2. Add a little bit to m (i.e. 10^-7)
# 3. Calculate cost for new m and b
    # dcost/dm = cost(m + new m, b) - cost(m, b)/10^-7

# 1. reset m, add a little bit to b (i.e. 10^-7)
# 2. Calculate cost for new m and b
    # dcost/db = cost(m, b + new b) - cost(m, b)/10^-7

plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Original Line", color='b')
plt.scatter(x, noisy_y, label="Noisy Line", color='r', alpha=0.7)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# plt.legend()
plt.title("Simple Line Plot with Gaussian Noise")
plt.grid(True)
plt.show()