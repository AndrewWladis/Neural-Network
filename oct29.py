import numpy as np
import matplotlib.pyplot as plt

plt.ion()

def update_plot(actual_Y, predicted_Y):
    plt.clf()
    plt.scatter(X, actual_Y, label='Actual')
    plt.scatter(X, predicted_Y, color='r', marker='x', label='Predicted')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.pause(0.01)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_parameters(input_dim, hidden_dim, output_dim):
    W1 = np.random.randn(hidden_dim, input_dim)
    b1 = np.zeros((hidden_dim, 1))
    W2 = np.random.randn(output_dim, hidden_dim)
    b2 = np.zeros((output_dim, 1))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def forward_propagation(X, parameters):
    Z1 = np.dot(parameters['W1'], X) + parameters['b1']
    A1 = sigmoid(Z1)
    Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
    A2 = Z2  # Linear activation for single-output
    return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

def backward_propagation(X, Y, cache, parameters, learning_rate):
    m = X.shape[1]

    dZ2 = cache['A2'] - Y
    dW2 = (1 / m) * np.dot(dZ2, cache['A1'].T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(parameters['W2'].T, dZ2) * sigmoid_derivative(cache['A1'])
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    parameters['W1'] -= learning_rate * dW1
    parameters['b1'] -= learning_rate * db1
    parameters['W2'] -= learning_rate * dW2
    parameters['b2'] -= learning_rate * db2

    return parameters


def train_neural_network(X, Y, hidden_dim, epochs, learning_rate):
    input_dim = X.shape[0]
    output_dim = Y.shape[0]

    parameters = initialize_parameters(input_dim, hidden_dim, output_dim)

    for epoch in range(epochs):
        cache = forward_propagation(X, parameters)
        parameters = backward_propagation(X, Y, cache, parameters, learning_rate)

        if epoch % 100 == 0:
            cost = np.mean(np.square(cache['A2'] - Y))
            print(f"Epoch {epoch}: Cost = {cost}")

            # Get predictions for current parameters
            predicted_Y = predict(X, parameters)
            update_plot(Y, predicted_Y)

    plt.ioff()  # Turn off interactive mode at the end
    plt.show()

    return parameters

def predict(X, parameters):
    cache = forward_propagation(X, parameters)
    return cache['A2']

# Example data
np.random.seed(42)
X = np.random.rand(1, 100) * 10 - 5
#Y = 2 * X + 1 + np.random.randn(1, 100)  # y = 2x + 1 + noise
Y = np.sin(X) + 1 + np.random.randn(1, 100) # y = sin(x) + 1 + noise

# Training the neural network
parameters = train_neural_network(X, Y, hidden_dim=16, epochs=10000, learning_rate=0.01)

# Predictions
predicted_Y = predict(X, parameters)

# Plotting
plt.scatter(X, Y, label='Actual')
plt.scatter(X, predicted_Y, color='r', marker='x', label='Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
