import os
import struct
import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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

    # Lists to store accuracy values during training
    epochs_list = []
    accuracies_list = []


    # Initialize parameters
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    biases_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.randn(hidden_size, output_size)
    biases_output = np.zeros((1, output_size))

    # Training loop
    for epoch in range(epochs):
        total_correct = 0
        total_samples = 0

        # Shuffle the training data
        indices = np.random.permutation(len(train_images))
        train_images_shuffled = train_images[indices]
        train_labels_shuffled = train_labels[indices]

        # Generate a random starting index for each epoch
        random_start = np.random.randint(0, len(train_images_shuffled) - batch_size + 1)

        # Ensure that each epoch processes exactly 64 inputs
        inputs = train_images_shuffled[random_start:random_start + batch_size]
        targets = np.eye(output_size)[train_labels_shuffled[random_start:random_start + batch_size]]

        # Forward pass
        hidden_activations, output_activations = forward_pass(
            inputs, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output
        )

        # Backward pass and update parameters
        backward_pass(
            inputs, hidden_activations, output_activations, targets,
            weights_input_hidden, biases_hidden, weights_hidden_output, biases_output, learning_rate
        )

        # Count correctly guessed numbers
        total_correct += np.sum(
            np.argmax(output_activations, axis=1) == train_labels_shuffled[random_start:random_start + batch_size])
        total_samples += len(inputs)

        # Calculate accuracy
        accuracy = total_correct / total_samples

        epochs_list.append(epoch + 1)
        accuracies_list.append(accuracy)
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

class NeuralNetworkGUI:
    def __init__(self, master, test_images, test_labels, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output):
        self.master = master
        self.test_images = test_images
        self.test_labels = test_labels
        self.weights_input_hidden = weights_input_hidden
        self.biases_hidden = biases_hidden
        self.weights_hidden_output = weights_hidden_output
        self.biases_output = biases_output
        self.index = 0

        self.create_widgets()

    def create_widgets(self):
        self.canvas = Canvas(self.master, width=280, height=280)
        self.canvas.pack()

        self.label = tk.Label(self.master, text="Prediction: ")
        self.label.pack()

        self.next_button = tk.Button(self.master, text="Next Image", command=self.show_next_image)
        self.next_button.pack()

    def show_next_image(self):
        if self.index < len(self.test_images):
            image = self.test_images[self.index].reshape((28, 28))
            self.display_image(image)
            prediction = self.predict_single_image(self.test_images[self.index])
            self.label.config(text=f"Prediction: {prediction}")
            self.index += 1

    def display_image(self, image):
        self.canvas.delete("all")
        img = Image.fromarray((image * 255).astype(np.uint8))
        img = img.resize((280, 280), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    def predict_single_image(self, image):
        hidden_activations, output_activations = forward_pass(
            image, self.weights_input_hidden, self.biases_hidden, self.weights_hidden_output, self.biases_output
        )
        return np.argmax(output_activations)



if __name__ == "__main__":
    root = tk.Tk()
    root.title("Neural Network GUI")

    gui = NeuralNetworkGUI(root, test_images, test_labels, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output)

    root.mainloop()