#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions and classes for Adaptive Hierarchical Drift Learning (AHDL).

This module provides functionality for handling data streams in the AHDL framework, including
a class to represent individual stream samples, functions to define concept drifts for the
EMNIST dataset, and utilities for image preprocessing, noise addition, and visualization.
The concept functions map EMNIST labels to binary labels for different concepts (numbers,
lowercase, uppercase), and visualization functions display samples with their labels.

Created on Wed Jul  2 10:39:09 2025

@author: fran
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define epsilon for numerical stability in clipping operations
epsilon = tf.keras.backend.epsilon()

class StreamSample:
    """
    Represents an individual sample in the data stream.

    Stores an image, its original label, a binary label for the current task, a sample ID,
    and a flag indicating whether the label is available (e.g., after a delay).

    Args:
        image (np.ndarray): Image data with shape (height, width, channels).
        original_label (float): Original EMNIST label (0–46).
        binary_label (float): Binary label (0 or 1) for the current concept.
        sample_id (int): Unique identifier for the sample in the stream.
    """
    def __init__(self, image, original_label, binary_label, sample_id):
        self.image = image
        self.original_label = original_label
        self.binary_label = binary_label
        self.sample_id = sample_id
        self.label_available = False  # Label is not available initially

# --- Concept Functions ---
# These functions define the binary label for each concept
def concept_function_numbers(label):
    """
    Classify digits (0–9) based on whether they are less than 5.

    Maps EMNIST digit labels (0–9) to binary labels: 1 if label < 5 (e.g., 0–4),
    0 otherwise (e.g., 5–9).

    Args:
        label (float): Original EMNIST label (0–46, but 0–9 for digits).

    Returns:
        int: Binary label (1 if label < 5, else 0).
    """
    return 1 if label < 5 else 0

def concept_function_lowercase(label):
    """
    Classify lowercase letters based on even/odd label index.

    Maps EMNIST lowercase letter labels (10–35) to binary labels: 1 if the label index
    is even, 0 if odd. Assumes numerical mapping of labels.

    Args:
        label (float): Original EMNIST label (0–46, but 10–35 for lowercase).

    Returns:
        int: Binary label (1 if label is even, else 0).
    """
    return 1 if label % 2 == 0 else 0

def concept_function_uppercase(label):
    """
    Classify uppercase letters based on whether they are after the 10th letter.

    Maps EMNIST uppercase letter labels (36–46) to binary labels: 1 if label > 10
    (e.g., K–Z), 0 otherwise (e.g., A–J).

    Args:
        label (float): Original EMNIST label (0–46, but 36–46 for uppercase).

    Returns:
        int: Binary label (1 if label > 10, else 0).
    """
    return 1 if label > 10 else 0

concept_functions_image = [
    concept_function_numbers,
    concept_function_lowercase,
    concept_function_uppercase
]

# Mapping of concept IDs to human-readable descriptions
concept_descriptions = {
    0: 'Numéricos (0-9)',
    1: 'Minúsculas (a-z)',
    2: 'Mayúsculas (A-Z)'
}

def concept_0_complex(original_label):
    """
    Concept 0: Classify even digits and uppercase letters from M onward.

    Maps EMNIST labels (0–46) to binary labels:
    - Digits (0–9): 1 if even (0, 2, 4, 6, 8), else 0.
    - Uppercase letters (10–35): 1 if label >= 22 (M–Z), else 0.
    - Other labels (e.g., lowercase): 0.

    Args:
        original_label (float): Original EMNIST label (0–46).

    Returns:
        int: Binary label (1 for even digits or M–Z, else 0).
    """
    if 0 <= original_label <= 9:  # Is a digit
        return 1 if original_label % 2 == 0 else 0
    elif 10 <= original_label <= 35:  # Is an uppercase letter
        # M is label 22 (A=10, B=11, ..., M=22)
        return 1 if original_label >= 22 else 0
    else:  # Other cases (lowercase or invalid) do not apply
        return 0

def concept_1_complex(original_label):
    """
    Concept 1: Classify odd digits and lowercase letters up to 'l'.

    Maps EMNIST labels (0–46) to binary labels:
    - Digits (0–9): 1 if odd (1, 3, 5, 7, 9), else 0.
    - Lowercase letters (36–61): 1 if label <= 47 (a–l), else 0.
    - Other labels (e.g., uppercase): 0.

    Args:
        original_label (float): Original EMNIST label (0–46).

    Returns:
        int: Binary label (1 for odd digits or a–l, else 0).
    """
    if 0 <= original_label <= 9:  # Is a digit
        return 1 if original_label % 2 != 0 else 0
    elif 36 <= original_label <= 61:  # Is a lowercase letter
        # 'l' is label 47 (a=36, b=37, ..., l=47)
        return 1 if original_label <= 47 else 0
    else:  # Other cases (uppercase or invalid) do not apply
        return 0

def concept_2_complex(original_label):
    """
    Concept 2: Classify lowercase letters from 'm' onward and uppercase letters up to 'L'.

    Maps EMNIST labels (0–46) to binary labels:
    - Lowercase letters (36–61): 1 if label >= 48 (m–z), else 0.
    - Uppercase letters (10–35): 1 if label <= 21 (A–L), else 0.
    - Other labels (e.g., digits): 0.

    Args:
        original_label (float): Original EMNIST label (0–46).

    Returns:
        int: Binary label (1 for m–z or A–L, else 0).
    """
    if 36 <= original_label <= 61:  # Is a lowercase letter
        # 'm' is label 48 (a=36, ..., m=48)
        return 1 if original_label >= 48 else 0
    elif 10 <= original_label <= 35:  # Is an uppercase letter
        # L is label 21 (A=10, ..., L=21)
        return 1 if original_label <= 21 else 0
    else:  # Other cases (digits or invalid) do not apply
        return 0

"""
# Lista de funciones de concepto para ser utilizadas dinámicamente
concept_functions_image = [
    concept_0_complex,
    concept_1_complex,
    concept_2_complex
]

# Mapeo de IDs de concepto a descripciones legibles
concept_descriptions = {
    0: 'Concepto 0: Pares | Mayúsculas (M-Z)',
    1: 'Concepto 1: Impares | Minúsculas (a-l)',
    2: 'Concepto 2: Minúsculas (m-z) | Mayúsculas (A-L)'
}
"""    

# --- Preprocessing and Noise Functions ---
def preprocess_image(image, label):
    """
    Preprocess an EMNIST image and label for model input.

    Normalizes the image to [0, 1] and ensures it has shape (height, width, 1).
    Converts the label to float32.

    Args:
        image (tf.Tensor): Input image tensor, potentially with varying shapes.
        label (tf.Tensor): Original EMNIST label.

    Returns:
        tuple: (processed_image, processed_label), where processed_image is a tensor
               with shape (height, width, 1) and processed_label is a float32 scalar.
    """
    image = tf.image.convert_image_dtype(image, tf.float32)  # Scale to [0, 1]
    
    # Ensure the image has a channel dimension and no extra dimensions
    if len(image.shape) == 2:  # (H, W) -> (H, W, 1)
        image = tf.expand_dims(image, axis=-1)
    elif len(image.shape) == 4 and image.shape[-1] == 1 and image.shape[-2] == 1:  # (H, W, 1, 1) -> (H, W, 1)
        image = tf.squeeze(image, axis=-1)  # Remove redundant last dimension
    elif len(image.shape) == 3 and image.shape[-1] == 1:  # Already (H, W, 1)
        pass
    else:
        print(f"Warning: Unexpected image shape after conversion: {image.shape}. Attempting to reshape to (H, W, 1).")
        if image.shape.rank >= 2:
            image = tf.reshape(image, (image.shape[-2], image.shape[-1], 1))
        else:
            raise ValueError(f"Cannot process image with shape {image.shape} to (H, W, 1).")

    label = tf.cast(label, tf.float32)
    return image, label

def add_gaussian_noise(image_np, noise_factor=0.1):
    """
    Add Gaussian noise to a NumPy image array.

    Generates noise with mean 0 and standard deviation 1, scales it by noise_factor,
    and adds it to the image. Clips the result to [epsilon, 1-epsilon] for numerical stability.

    Args:
        image_np (np.ndarray): Input image with shape (height, width, channels).
        noise_factor (float, optional): Scaling factor for the noise. Defaults to 0.1.

    Returns:
        np.ndarray: Noisy image clipped to [epsilon, 1-epsilon].
    """
    noise = np.random.normal(loc=0.0, scale=1.0, size=image_np.shape)
    noisy_image = image_np + noise_factor * noise
    return np.clip(noisy_image, epsilon, 1. - epsilon)  # Clip to [epsilon, 1-epsilon]

# --- Visualization Function ---
def visualize_samples_with_labels(images_data, labels_data, num_samples=9,
                                 title="Muestras de Imágenes y Etiquetas",
                                 is_tensor=True,
                                 current_concept_id=None):
    """
    Display a grid of EMNIST images with their labels and current concept.

    Visualizes up to `num_samples` images in a square grid (e.g., 3x3 for 9 samples),
    rotating EMNIST images for correct orientation and displaying binary labels.
    Optionally includes the current concept description.

    Args:
        images_data (np.ndarray or tf.Tensor): Images with shape (num_samples, height, width, channels).
        labels_data (np.ndarray or tf.Tensor): Binary labels with shape (num_samples,).
        num_samples (int, optional): Number of samples to display (ideally 9 for 3x3 grid). Defaults to 9.
        title (str, optional): Title for the figure. Defaults to "Muestras de Imágenes y Etiquetas".
        is_tensor (bool, optional): If True, inputs are TensorFlow tensors and will be converted to NumPy.
                                   Defaults to True.
        current_concept_id (int, optional): ID of the current concept for display in the title.

    Returns:
        None: Displays the plot using matplotlib.
    """
    if images_data is None or len(images_data) == 0:
        print("No image data available to display.")
        return
    if labels_data is None or len(labels_data) == 0:
        print("No label data available to display.")
        return

    # Convert to NumPy if inputs are TensorFlow tensors
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
        concept_text = concept_descriptions.get(current_concept_id, 'Unknown')
        plt.suptitle(f"{title}\nCurrent Concept: {concept_text} (ID: {current_concept_id})", fontsize=16)
    else:
        plt.suptitle(title, fontsize=16)

    for i in range(num_samples_to_show):
        plt.subplot(rows, cols, i + 1)
        
        img_to_display = images_display[i, :, :, 0]
        
        img_to_display = np.rot90(img_to_display, k=-1)
        img_to_display = np.fliplr(img_to_display)

        plt.imshow(img_to_display, cmap='gray')
        
        plt.title(f"Label: {labels_display[i]}", fontsize=10)
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()