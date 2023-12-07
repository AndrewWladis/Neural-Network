import os
import struct
import numpy as np
from scipy.ndimage import rotate

# Function to read MNIST images
# Function to read MNIST images
def read_idx_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
        if magic != 0x803:
            raise ValueError("Invalid magic number for images file")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape((num_images, num_rows * num_cols))


# Function to read MNIST labels
def read_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 0x801:
            raise ValueError("Invalid magic number for labels file")
        return np.frombuffer(f.read(), dtype=np.int8)

# Function to one-hot encode labels
def one_hot_encode(labels, num_classes):
    encoded_labels = np.zeros((len(labels), num_classes))
    encoded_labels[np.arange(len(labels)), labels] = 1
    return encoded_labels

# Function to initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    biases_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.randn(hidden_size, output_size)
    biases_output = np.zeros((1, output_size))
    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output

# Function to perform forward pass
def forward_pass(inputs, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output):
    hidden_activations = sigmoid(np.dot(inputs, weights_input_hidden) + biases_hidden)
    output_activations = softmax(np.dot(hidden_activations, weights_hidden_output) + biases_output)
    return hidden_activations, output_activations

# Function to compute loss
def compute_loss(predictions, targets):
    return -np.sum(targets * np.log(predictions)) / len(predictions)

# Function to perform backward pass and update parameters
def backward_pass(inputs, hidden_activations, output_activations, targets,
                  weights_input_hidden, biases_hidden, weights_hidden_output, biases_output, learning_rate):
    output_loss = output_activations - targets
    hidden_loss = np.dot(output_loss, weights_hidden_output.T) * hidden_activations * (1 - hidden_activations)

    weights_hidden_output -= learning_rate * np.dot(hidden_activations.T, output_loss) / len(inputs)
    biases_output -= learning_rate * np.sum(output_loss, axis=0, keepdims=True) / len(inputs)
    weights_input_hidden -= learning_rate * np.dot(inputs.T, hidden_loss) / len(inputs)
    biases_hidden -= learning_rate * np.sum(hidden_loss, axis=0, keepdims=True) / len(inputs)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Softmax activation function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Function to train the neural network
# Function to calculate accuracy
def calculate_accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == labels)

# Function to read data in IDX format
def read_idx(filename_images, filename_labels):
    images = read_idx_images(filename_images)
    labels = read_idx_labels(filename_labels)
    return images, labels

# Function to save data in IDX format
def save_idx(images, labels, filename_images, filename_labels):
    save_idx(images, labels, filename_images)
    save_idx(images, labels, filename_labels)

# Function to train the neural network
def train_neural_network(train_images, train_labels, test_images, test_labels,
                         input_size, hidden_size, output_size, learning_rate, epochs, batch_size):
    # Initialize parameters
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    biases_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.randn(hidden_size, output_size)
    biases_output = np.zeros((1, output_size))

    # Training loop
    for epoch in range(epochs):
        total_correct = 0
        total_samples = 0
        for i in range(0, len(train_images), batch_size):
            inputs = train_images[i:i + batch_size]
            targets = np.eye(output_size)[train_labels[i:i + batch_size]]

            # Forward pass
            hidden_activations = 1 / (1 + np.exp(-np.dot(inputs, weights_input_hidden) - biases_hidden))
            output_activations = np.exp(np.dot(hidden_activations, weights_hidden_output) + biases_output)
            output_activations /= np.sum(output_activations, axis=1, keepdims=True)

            # Backward pass and update parameters
            output_loss = output_activations - targets
            hidden_loss = np.dot(output_loss, weights_hidden_output.T) * hidden_activations * (1 - hidden_activations)

            weights_hidden_output -= learning_rate * np.dot(hidden_activations.T, output_loss) / len(inputs)
            biases_output -= learning_rate * np.sum(output_loss, axis=0, keepdims=True) / len(inputs)
            weights_input_hidden -= learning_rate * np.dot(inputs.T, hidden_loss) / len(inputs)
            biases_hidden -= learning_rate * np.sum(hidden_loss, axis=0, keepdims=True) / len(inputs)

            # Count correctly guessed numbers
            total_correct += np.sum(np.argmax(output_activations, axis=1) == train_labels[i:i + batch_size])
            total_samples += len(inputs)

        # Calculate accuracy
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {accuracy:.2%}")

    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output

# Function to evaluate the neural network
def evaluate_neural_network(images, labels, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output):
    hidden_activations = 1 / (1 + np.exp(-np.dot(images, weights_input_hidden) - biases_hidden))
    output_activations = np.exp(np.dot(hidden_activations, weights_hidden_output) + biases_output)
    output_activations /= np.sum(output_activations, axis=1, keepdims=True)

    accuracy = np.mean(np.argmax(output_activations, axis=1) == labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Load original and rotated MNIST data from IDX files
data_path = "archive"  # Replace with the actual full path to your MNIST data

original_train_images, original_train_labels = read_idx(
    os.path.join(data_path, "original_train_data.idx"),
    os.path.join(data_path, "original_train_labels.idx")
)
rotated_train_images, rotated_train_labels = read_idx(
    os.path.join(data_path, "rotated_train_data.idx"),
    os.path.join(data_path, "rotated_train_labels.idx")
)
test_images, test_labels = read_idx(
    os.path.join(data_path, "t10k-images.idx3-ubyte"),
    os.path.join(data_path, "t10k-labels.idx1-ubyte")
)

# Normalize pixel values to between 0 and 1
original_train_images = original_train_images / 255.0
rotated_train_images = rotated_train_images / 255.0
test_images = test_images / 255.0

# Train the neural network using the augmented dataset
input_size = 784  # Size of each input image
hidden_size = 25  # Number of neurons in the hidden layer
output_size = 10  # Number of output classes

# Hyperparameters
learning_rate = 1
epochs = 10000
batch_size = 64

# Train the neural network
weights_input_hidden, biases_hidden, weights_hidden_output, biases_output = train_neural_network(
    np.concatenate((original_train_images, rotated_train_images)),
    np.concatenate((original_train_labels, rotated_train_labels)),
    test_images, test_labels,
    input_size, hidden_size, output_size,
    learning_rate, epochs, batch_size
)

# Evaluate the neural network on the test set
evaluate_neural_network(
    test_images, test_labels,
    weights_input_hidden, biases_hidden, weights_hidden_output, biases_output
)
