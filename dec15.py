import os
import struct
import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn.functional as F

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
    torch.manual_seed(42)
    weights_input_hidden = torch.randn(input_size, hidden_size, requires_grad=True)
    biases_hidden = torch.zeros(1, hidden_size, requires_grad=True)
    weights_hidden_output = torch.randn(hidden_size, output_size, requires_grad=True)
    biases_output = torch.zeros(1, output_size, requires_grad=True)
    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output

# Function to perform forward pass
def forward_pass(inputs, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output):
    hidden_activations = F.relu(torch.matmul(inputs, weights_input_hidden) + biases_hidden)
    output_activations = F.softmax(torch.matmul(hidden_activations, weights_hidden_output) + biases_output, dim=1)
    return hidden_activations, output_activations

# Function to compute loss
def compute_loss(predictions, targets):
    return -torch.sum(targets * torch.log(predictions)) / len(predictions)

# Function to perform backward pass and update parameters
def backward_pass(inputs, hidden_activations, output_activations, targets,
                  weights_input_hidden, biases_hidden, weights_hidden_output, biases_output, learning_rate):
    output_loss = output_activations - targets
    hidden_loss = torch.matmul(output_loss, weights_hidden_output.t()) * (hidden_activations > 0).float()

    weights_hidden_output.data -= learning_rate * torch.matmul(hidden_activations.t(), output_loss) / len(inputs)
    biases_output.data -= learning_rate * torch.sum(output_loss, dim=0, keepdim=True) / len(inputs)
    weights_input_hidden.data -= learning_rate * torch.matmul(inputs.t(), hidden_loss) / len(inputs)
    biases_hidden.data -= learning_rate * torch.sum(hidden_loss, dim=0, keepdim=True) / len(inputs)

# Function to train the neural network
# Function to train the neural network
def train_neural_network(train_images, train_labels, test_images, test_labels,
                         input_size, hidden_size, output_size, learning_rate, epochs, batch_size):

    # Lists to store accuracy and cost values during training
    epochs_list = []
    accuracies_list = []
    costs_list = []

    # Initialize parameters
    weights_input_hidden, biases_hidden, weights_hidden_output, biases_output = initialize_parameters(
        input_size, hidden_size, output_size
    )

    # Convert data to PyTorch tensors
    train_images_tensor = torch.tensor(train_images, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_images_tensor = torch.tensor(test_images, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    # Training loop
    for epoch in range(epochs):
        total_correct = 0
        total_samples = 0
        total_cost = 0

        # Shuffle the training data
        indices = torch.randperm(len(train_images_tensor))
        train_images_shuffled = train_images_tensor[indices]
        train_labels_shuffled = train_labels_tensor[indices]

        # Generate a random starting index for each epoch
        random_start = np.random.randint(0, len(train_images_shuffled) - batch_size + 1)

        # Ensure that each epoch processes exactly 64 inputs
        inputs = train_images_shuffled[random_start:random_start + batch_size]
        targets = F.one_hot(train_labels_shuffled[random_start:random_start + batch_size], num_classes=output_size)

        # Forward pass
        hidden_activations, output_activations = forward_pass(
            inputs, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output
        )

        # Compute loss
        loss = compute_loss(output_activations, targets)
        total_cost += loss.item()

        # Backward pass and update parameters
        backward_pass(
            inputs, hidden_activations, output_activations, targets,
            weights_input_hidden, biases_hidden, weights_hidden_output, biases_output, learning_rate
        )

        # Count correctly guessed numbers
        total_correct += torch.sum(
            torch.argmax(output_activations, dim=1) == train_labels_shuffled[random_start:random_start + batch_size]
        )
        total_samples += len(inputs)

        # Calculate accuracy
        accuracy = total_correct.item() / total_samples

        epochs_list.append(epoch + 1)
        accuracies_list.append(accuracy)
        costs_list.append(total_cost / len(inputs))
        #print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {accuracy:.2%}, Cost: {(total_cost / len(inputs)):.4}")
        print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {accuracy:.2%}")

    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output

# Function to evaluate the neural network
def evaluate_neural_network(images, labels, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output):
    inputs = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    hidden_activations, output_activations = forward_pass(
        inputs, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output
    )

    _, predictions = torch.max(output_activations, 1)
    accuracy = torch.sum(predictions == labels).item() / len(labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Load original and rotated MNIST data from IDX files
data_path = "archive"  # Replace with the actual full path to your MNIST data

original_train_images, original_train_labels = read_idx_images(
    os.path.join(data_path, "original_train_data.idx")
), read_idx_labels(os.path.join(data_path, "original_train_labels.idx"))
rotated_train_images, rotated_train_labels = read_idx_images(
    os.path.join(data_path, "rotated_train_data.idx")
), read_idx_labels(os.path.join(data_path, "rotated_train_labels.idx"))
test_images, test_labels = read_idx_images(
    os.path.join(data_path, "t10k-images.idx3-ubyte")
), read_idx_labels(os.path.join(data_path, "t10k-labels.idx1-ubyte"))

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

        self.true_label_label = tk.Label(self.master, text="True Label: ")
        self.true_label_label.pack()

        self.next_button = tk.Button(self.master, text="Next Image", command=self.show_next_image)
        self.next_button.pack()

    def show_next_image(self):
        if self.index < len(self.test_images):
            image = self.test_images[self.index].reshape((28, 28))
            self.display_image(image)
            prediction = self.predict_single_image(self.test_images[self.index])
            true_label = self.test_labels[self.index]
            self.label.config(text=f"Prediction: {prediction}")
            self.true_label_label.config(text=f"True Label: {true_label}")
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
