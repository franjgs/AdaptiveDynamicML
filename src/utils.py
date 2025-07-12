#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for the AdaptiveDynamicML project.

This module provides helper functions and a class for handling the EMNIST dataset in concept
drift simulations. It includes functions for defining binary classification concepts, preprocessing
images, adding noise, and visualizing samples.

Created on Wed Jul 2 10:39:09 2025
Author: Fran
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class StreamSample:
    """Represents an individual sample in a data stream.

    Stores an image, its original label, binary label for the current task, sample ID,
    and a flag indicating whether the label is available.
    """

    def __init__(self, image, original_label, binary_label, sample_id):
        """Initialize a StreamSample instance.

        Args:
            image (np.ndarray or tf.Tensor): Image data (e.g., shape (height, width, channels)).
            original_label (int): Original label from the dataset (e.g., EMNIST class).
            binary_label (int): Binary label (0 or 1) for the current concept task.
            sample_id (int): Unique identifier for the sample.
        """
        self.image = image
        self.original_label = original_label
        self.binary_label = binary_label
        self.sample_id = sample_id
        self.label_available = False  # Label is not available initially

def concept_function_numbers(label):
    """Classify digits (0-9) as even or odd based on a threshold.

    Labels digits less than 5 as 1, otherwise 0.

    Args:
        label (int): Original digit label (0-9).

    Returns:
        int: Binary label (1 for < 5, 0 for >= 5).
    """
    return 1 if label < 5 else 0

def concept_function_lowercase(label):
    """Classify lowercase letters based on parity of their numerical mapping.

    Assumes letters are mapped to numbers (a=0, b=1, ..., z=25).

    Args:
        label (int): Numerical label for a lowercase letter.

    Returns:
        int: Binary label (1 for even-indexed letters, 0 for odd).
    """
    return 1 if label % 2 == 0 else 0

def concept_function_uppercase(label):
    """Classify uppercase letters based on a threshold.

    Assumes letters are mapped to numbers (A=0, B=1, ..., Z=25).
    Labels letters after K (index 10) as 1, otherwise 0.

    Args:
        label (int): Numerical label for an uppercase letter.

    Returns:
        int: Binary label (1 for > 10, 0 for <= 10).
    """
    return 1 if label > 10 else 0

# List of concept functions for dynamic application
CONCEPT_FUNCTIONS = [
    concept_function_numbers,
    concept_function_lowercase,
    concept_function_uppercase
]

# Mapping of concept IDs to human-readable descriptions
CONCEPT_DESCRIPTIONS = {
    0: 'Digits (0-9)',
    1: 'Lowercase Letters (a-z)',
    2: 'Uppercase Letters (A-Z)'
}

def preprocess_image(image, label):
    """Normalize an image and ensure it has the correct shape (H, W, 1).

    Args:
        image (tf.Tensor): Input image, potentially with varying shapes.
        label (tf.Tensor or int): Label for the image.

    Returns:
        tuple: (normalized image, casted label) with image shape (height, width, 1).

    Raises:
        ValueError: If the image shape cannot be processed to (H, W, 1).
    """
    image = tf.image.convert_image_dtype(image, tf.float32)  # Scale to [0, 1]

    if len(image.shape) == 2:  # (H, W) -> (H, W, 1)
        image = tf.expand_dims(image, axis=-1)
    elif len(image.shape) == 4 and image.shape[-1] == 1 and image.shape[-2] == 1:  # (H, W, 1, 1) -> (H, W, 1)
        image = tf.squeeze(image, axis=-1)
    elif len(image.shape) == 3 and image.shape[-1] == 1:  # Already (H, W, 1)
        pass
    else:
        try:
            if image.shape.rank >= 2:
                image = tf.reshape(image, (image.shape[-2], image.shape[-1], 1))
            else:
                raise ValueError(f"Cannot process image with shape {image.shape} to (H, W, 1).")
        except Exception as e:
            print(f"Error processing image shape {image.shape}: {e}")
            raise ValueError(f"Cannot process image with shape {image.shape} to (H, W, 1).")

    label = tf.cast(label, tf.float32)
    return image, label

def add_gaussian_noise(image_np, noise_factor=0.1):
    """Add Gaussian noise to a NumPy image array.

    Args:
        image_np (np.ndarray): Input image array with values in [0, 1].
        noise_factor (float, optional): Scaling factor for the Gaussian noise.

    Returns:
        np.ndarray: Noisy image with values clipped to [0, 1].
    """
    noise = np.random.normal(loc=0.0, scale=1.0, size=image_np.shape)
    noisy_image = image_np + noise_factor * noise
    return np.clip(noisy_image, 0.0, 1.0)

def visualize_samples_with_labels(images_data, labels_data, num_samples=9, title="Image Samples and Labels",
                                 is_tensor=True, current_concept_id=None):
    """Display a grid of image samples with their labels and current concept.

    Visualizes up to `num_samples` images (default 3x3 grid) from the EMNIST dataset,
    rotating them for correct orientation and displaying their binary labels and the current
    concept description.

    Args:
        images_data (np.ndarray or tf.Tensor): Image data with shape (num_samples, height, width, channels).
        labels_data (np.ndarray or tf.Tensor): Labels with shape (num_samples,).
        num_samples (int, optional): Number of samples to display (default 9 for a 3x3 grid).
        title (str, optional): Title for the figure.
        is_tensor (bool, optional): If True, converts TensorFlow tensors to NumPy arrays.
        current_concept_id (int, optional): ID of the current concept for display in the title.

    Returns:
        None: Displays the plot using Matplotlib.
    """
    if images_data is None or len(images_data) == 0:
        print("No image data to display.")
        return
    if labels_data is None or len(labels_data) == 0:
        print("No label data to display.")
        return

    if is_tensor:
        images_display = images_data.numpy() if tf.is_tensor(images_data) else images_data
        labels_display = labels_data.numpy() if tf.is_tensor(labels_data) else labels_data
    else:
        images_display = images_data
        labels_display = labels_data

    num_samples_to_show = min(num_samples, images_display.shape[0])
    rows = int(np.ceil(np.sqrt(num_samples_to_show)))
    cols = int(np.ceil(num_samples_to_show / rows))

    plt.figure(figsize=(cols * 2.5, rows * 2.5 + 1))

    if current_concept_id is not None:
        concept_text = CONCEPT_DESCRIPTIONS.get(current_concept_id, 'Unknown')
        plt.suptitle(f"{title}\nCurrent Concept: {concept_text} (ID: {current_concept_id})", fontsize=16)
    else:
        plt.suptitle(title, fontsize=16)

    for i in range(num_samples_to_show):
        plt.subplot(rows, cols, i + 1)
        img_to_display = images_display[i, :, :, 0]
        img_to_display = np.rot90(img_to_display, k=-1)
        img_to_display = np.fliplr(img_to_display)
        plt.imshow(img_to_display, cmap='gray')
        plt.title(f"Label: {int(labels_display[i])}", fontsize=10)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()