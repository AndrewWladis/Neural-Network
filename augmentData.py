import os
import struct
import numpy as np
from scipy.ndimage import rotate

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

# Function to save data in IDX format
def save_idx(images, labels, filename_images, filename_labels):
    with open(filename_images, 'wb') as f_images:
        # Header for images
        f_images.write(struct.pack('>IIII', 0x803, len(images), 28, 28))
        # Data for images
        f_images.write(images.astype(np.uint8).tobytes())

    with open(filename_labels, 'wb') as f_labels:
        # Header for labels
        f_labels.write(struct.pack('>II', 0x801, len(labels)))
        # Data for labels
        f_labels.write(labels.astype(np.int8).tobytes())


# Load original MNIST data
data_path = "archive"  # Replace with the actual full path to your MNIST data
original_train_images = read_idx_images(os.path.join(data_path, "train-images.idx3-ubyte"))
original_train_labels = read_idx_labels(os.path.join(data_path, "train-labels.idx1-ubyte"))

# Apply rotation to create a second dataset
rotation_range = 15
rotated_train_images = []
rotated_train_labels = []

for image, label in zip(original_train_images, original_train_labels):
    # Randomly choose a rotation angle between -rotation_range and rotation_range
    angle = np.random.uniform(-rotation_range, rotation_range)

    # Rotate the image
    rotated_image = rotate(image.reshape(28, 28), angle, reshape=False).flatten()

    rotated_train_images.append(rotated_image)
    rotated_train_labels.append(label)

rotated_train_images = np.array(rotated_train_images)
rotated_train_labels = np.array(rotated_train_labels)

# Save both datasets
save_idx(original_train_images, original_train_labels, "archive/original_train_data.idx", "archive/original_train_labels.idx")
save_idx(rotated_train_images, rotated_train_labels, "archive/rotated_train_data.idx", "archive/rotated_train_labels.idx")

