#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 10:39:09 2025

@author: fran
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class StreamSample:
    """
    Representa una muestra individual en el flujo de datos.
    Contiene la imagen, la etiqueta original, la etiqueta binaria (para la tarea actual),
    el ID de la muestra y un flag para indicar si la etiqueta ya está disponible.
    """
    def __init__(self, image, original_label, binary_label, sample_id):
        self.image = image
        self.original_label = original_label
        self.binary_label = binary_label
        self.sample_id = sample_id
        self.label_available = False # La etiqueta no está disponible al principio

# --- FUNCIONES DE CONCEPTO ---
# Estas funciones definen la etiqueta binaria para cada concepto
def concept_function_numbers(label):
    """Clasifica dígitos (0-9) como par/impar (ej. < 5 vs >= 5)."""
    return 1 if label < 5 else 0

def concept_function_lowercase(label):
    """Clasifica letras minúsculas (ej. a-m vs n-z). Asume mapeo a números."""
    return 1 if label % 2 == 0 else 0

def concept_function_uppercase(label):
    """Clasifica letras mayúsculas (ej. A-M vs N-Z). Asume mapeo a números."""
    return 1 if label > 10 else 0

concept_functions_image = [
    concept_function_numbers,
    concept_function_lowercase,
    concept_function_uppercase
]

# Mapeo de IDs de concepto a descripciones legibles
concept_descriptions = {
    0: 'Numéricos (0-9)',
    1: 'Minúsculas (a-z)',
    2: 'Mayúsculas (A-Z)'
}

def concept_0_complex(original_label):
    """
    Concepto 0:
    - Reconocer números pares (0, 2, 4, 6, 8)
    - Y letras mayúsculas de la M en adelante (M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z)
    """
    if 0 <= original_label <= 9:  # Es un dígito
        return 1 if original_label % 2 == 0 else 0
    elif 10 <= original_label <= 35: # Es una letra mayúscula
        # M es la etiqueta 22 (A=10, B=11, ..., M=22)
        return 1 if original_label >= 22 else 0
    else: # Otros casos (minúsculas u otros rangos) no cumplen la condición para este concepto
        return 0 # O podrías considerar un valor diferente si representara "no aplica"

def concept_1_complex(original_label):
    """
    Concepto 1:
    - Reconocer números impares (1, 3, 5, 7, 9)
    - Y letras minúsculas de la 'l' y anteriores (a, b, c, d, e, f, g, h, i, j, k, l)
    """
    if 0 <= original_label <= 9:  # Es un dígito
        return 1 if original_label % 2 != 0 else 0
    elif 36 <= original_label <= 61: # Es una letra minúscula
        # 'l' es la etiqueta 47 (a=36, b=37, ..., l=47)
        return 1 if original_label <= 47 else 0
    else: # Otros casos (mayúsculas u otros rangos) no cumplen la condición para este concepto
        return 0

def concept_2_complex(original_label):
    """
    Concepto 2:
    - Reconocer letras minúsculas de la 'm' en adelante (m, n, o, p, q, r, s, t, u, v, w, x, y, z)
    - Y mayúsculas de la L y anteriores (A, B, C, D, E, F, G, H, I, J, K, L)
    """
    if 36 <= original_label <= 61: # Es una letra minúscula
        # 'm' es la etiqueta 48 (a=36, ..., m=48)
        return 1 if original_label >= 48 else 0
    elif 10 <= original_label <= 35: # Es una letra mayúscula
        # L es la etiqueta 21 (A=10, ..., L=21)
        return 1 if original_label <= 21 else 0
    else: # Otros casos (dígitos u otros rangos) no cumplen la condición para este concepto
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

# --- FUNCIONES DE PREPROCESAMIENTO Y RUIDO ---
def preprocess_image(image, label):
    """Normaliza la imagen y asegura 3 dimensiones (H, W, C=1)."""
    image = tf.image.convert_image_dtype(image, tf.float32) # Escalar a [0, 1]
    
    # Asegurarse de que la imagen tiene una dimensión de canal y no dimensiones extra
    if len(image.shape) == 2:  # (H, W) -> (H, W, 1)
        image = tf.expand_dims(image, axis=-1)
    elif len(image.shape) == 4 and image.shape[-1] == 1 and image.shape[-2] == 1: # (H, W, 1, 1) -> (H, W, 1)
        image = tf.squeeze(image, axis=-1) # Eliminar la última dimensión redundante
    elif len(image.shape) == 3 and image.shape[-1] == 1: # Ya es (H, W, 1)
        pass
    else:
        print(f"Advertencia: Forma de imagen inesperada después de la conversión: {image.shape}. Intentando forzar a (H, W, 1).")
        if image.shape.rank >= 2:
            image = tf.reshape(image, (image.shape[-2], image.shape[-1], 1))
        else:
            raise ValueError(f"No se puede procesar la imagen con forma {image.shape} a (H, W, 1).")

    label = tf.cast(label, tf.float32)
    return image, label

def add_gaussian_noise(image_np, noise_factor=0.1):
    """Añade ruido gaussiano a una imagen NumPy."""
    noise = np.random.normal(loc=0.0, scale=1.0, size=image_np.shape)
    noisy_image = image_np + noise_factor * noise
    return np.clip(noisy_image, 0., 1.) # Asegurar que los valores permanezcan en el rango [0, 1]

# --- FUNCIÓN DE VISUALIZACIÓN ---
def visualize_samples_with_labels(images_data, labels_data, num_samples=9,
                                  title="Muestras de Imágenes y Etiquetas",
                                  is_tensor=True,
                                  current_concept_id=None):
    """
    Representa un número de muestras de imágenes (en una cuadrícula 3x3) junto con sus etiquetas y el concepto actual.
    Las imágenes de EMNIST se rotan para una visualización correcta.

    Args:
        images_data: Un array NumPy de imágenes o un tensor de TensorFlow.
                      Se espera que tenga la forma (num_samples, height, width, channels).
        labels_data: Un array NumPy de etiquetas o un tensor de TensorFlow.
                      Se espera que tenga la forma (num_samples,).
        num_samples (int): El número de muestras a mostrar (idealmente 9 para 3x3).
        title (str): Título principal para la figura.
        is_tensor (bool): Indica si images_data y labels_data son tensores de TensorFlow.
                          Si es True, se convertirán a NumPy antes de mostrar.
        current_concept_id (int, optional): El ID del concepto actual que se está aplicando.
                                            Se usará para mostrar una descripción.
    """
    if images_data is None or len(images_data) == 0:
        print("No hay datos de imágenes para mostrar.")
        return
    if labels_data is None or len(labels_data) == 0:
        print("No hay datos de etiquetas para mostrar.")
        return

    
    # Convertir a NumPy si son tensores de TensorFlow
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
        concept_text = concept_descriptions.get(current_concept_id, 'Desconocido')
        plt.suptitle(f"{title}\nConcepto Actual: {concept_text} (ID: {current_concept_id})", fontsize=16)
    else:
        plt.suptitle(title, fontsize=16)


    for i in range(num_samples_to_show):
        plt.subplot(rows, cols, i + 1)
        
        img_to_display = images_display[i, :, :, 0]
        
        img_to_display = np.rot90(img_to_display, k=-1)
        img_to_display = np.fliplr(img_to_display)

        plt.imshow(img_to_display, cmap='gray')
        
        plt.title(f"Etiqueta: {labels_display[i]}", fontsize=10)
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()