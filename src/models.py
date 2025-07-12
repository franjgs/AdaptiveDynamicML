#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of machine learning models for Adaptive Hierarchical Drift Learning (AHDL).

This module defines the `FastModel` and `SlowModel` for handling concept drift in data streams,
along with an `Orchestrator` to combine their predictions. The models inherit from a `BaseModel`
class that provides common functionality for training, prediction, and evaluation. The module
supports various simulation modes and metrics for performance evaluation.

Created on Wed Jul  2 10:37:01 2025

@author: fran
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, accuracy_score
import abc

# Define epsilon for numerical stability in clipping operations
epsilon = tf.keras.backend.epsilon()

# --- CONFIGURATION ---
# Metrics to report during model evaluation
metrics_to_report = [
    'accuracy',
    'balanced_accuracy',
    'f1_score',
    'matthews_corrcoef'
]

# Operation modes for the Orchestrator
MODE_FAST_ONLY = "fast_only"
MODE_SLOW_ONLY = "slow_only"
MODE_INDEPENDENT = "independent"
MODE_FIXED_BETA = "fixed_beta"
MODE_ADAPTIVE_BETA = "adaptive_beta"

class BaseModel(abc.ABC):
    """
    Abstract base class for machine learning models.

    Provides common functionality for training, prediction, and evaluation of models used in
    the AHDL framework. Subclasses must implement the `_build_model` method to define specific
    architectures.

    Args:
        img_height (int): Height of input images.
        img_width (int): Width of input images.
        n_channels (int): Number of channels in input images (e.g., 1 for grayscale).
        debug_random_output (bool, optional): If True, returns random outputs for debugging.
                                            Defaults to False.
    """
    def __init__(self, img_height, img_width, n_channels, debug_random_output=False):
        self.img_shape = (img_height, img_width, n_channels)
        self.model = self._build_model()
        self.is_trained = False
        self.debug_random_output = debug_random_output

    @abc.abstractmethod
    def _build_model(self):
        """
        Build the specific architecture of the model.

        Must be implemented by subclasses to define the neural network layers and compilation.

        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        pass

    def train(self, X_train, y_train, epochs, batch_size, validation_split):
        """
        Train the model on the provided data.

        Args:
            X_train (np.ndarray): Training images with shape (num_samples, height, width, channels).
            y_train (np.ndarray): Training labels with shape (num_samples,).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of data to use for validation (0.0 to 1.0).

        Returns:
            dict: Training results with keys 'loss', 'accuracy', 'val_loss', 'val_accuracy'.
        """
        if self.debug_random_output:
            self.is_trained = True
            return {'loss': 0.5, 'accuracy': 0.5, 'val_loss': 0.5, 'val_accuracy': 0.5}

        # Convert to tensors
        X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
        
        if X_train_tensor.shape[0] == 0:
            print(f"  [{self.__class__.__name__}] No valid training data (X_train_tensor is empty).")
            return {'loss': 0.0, 'accuracy': 0.0, 'val_loss': 0.0, 'val_accuracy': 0.0}

        if y_train_tensor.ndim == 1:
            y_train_tensor = tf.expand_dims(y_train_tensor, axis=-1)

        history = self.model.fit(X_train_tensor, y_train_tensor,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=validation_split,
                                verbose=0)

        self.is_trained = True
        return {
            'loss': history.history['loss'][-1],
            'accuracy': history.history['accuracy'][-1],
            'val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else 0.0,
            'val_accuracy': history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0.0
        }

    def predict_proba(self, X):
        """
        Predict probabilities for the input data.

        Args:
            X (np.ndarray): Input images with shape (num_samples, height, width, channels).

        Returns:
            np.ndarray: Predicted probabilities with shape (num_samples,).
        """
        if self.debug_random_output:
            return np.random.rand(len(X))

        if not self.is_trained:
            return np.array([0.5] * len(X))

        return self.model.predict(X, verbose=0).flatten()

    def evaluate(self, X, y):
        """
        Evaluate the model's performance on the provided data.

        Computes metrics specified in `metrics_to_report` plus loss. Handles single-label cases
        to suppress sklearn warnings by returning 0.0 for affected metrics.

        Args:
            X (np.ndarray): Input images with shape (num_samples, height, width, channels).
            y (np.ndarray): True labels with shape (num_samples,).

        Returns:
            dict: Evaluation results with keys from `metrics_to_report` plus 'loss'.
        """
        y = np.asarray(y)

        if self.debug_random_output:
            n_samples = len(X)
            if n_samples == 0:
                return {metric: 0.0 for metric in metrics_to_report + ['loss']}

            random_probas = np.random.rand(n_samples)
            random_preds_binary = (random_probas > 0.5).astype(int)
            
            simulated_loss = tf.keras.losses.binary_crossentropy(
                tf.convert_to_tensor(y, dtype=tf.float32), 
                tf.convert_to_tensor(random_probas, dtype=tf.float32)
            ).numpy().mean()

            results = {
                'loss': simulated_loss, 
                'accuracy': accuracy_score(y, random_preds_binary, normalize=True)
            }
            # Check for single-label cases to suppress sklearn warnings
            if len(np.unique(y)) <= 1 or len(np.unique(random_preds_binary)) <= 1:
                results.update({
                    'balanced_accuracy': 0.0,
                    'f1_score': 0.0,
                    'matthews_corrcoef': 0.0
                })
            else:
                if 'balanced_accuracy' in metrics_to_report:
                    results['balanced_accuracy'] = balanced_accuracy_score(y, random_preds_binary)
                if 'f1_score' in metrics_to_report:
                    results['f1_score'] = f1_score(y, random_preds_binary, labels=[0, 1], average='binary', zero_division=0)
                if 'matthews_corrcoef' in metrics_to_report:
                    try:
                        results['matthews_corrcoef'] = matthews_corrcoef(y, random_preds_binary)
                    except ValueError:
                        results['matthews_corrcoef'] = 0.0 
            return results

        if not self.is_trained or len(X) == 0:
            return {metric: 0.0 for metric in metrics_to_report + ['loss']}

        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

        try:
            eval_results = self.model.evaluate(X_tensor, y_tensor, verbose=0, return_dict=True)
            loss = eval_results.get('loss', 0.0)
            accuracy = eval_results.get('accuracy', 0.0)
        except Exception as e:
            print(f"Warning during model evaluation: {e}")
            return {metric: 0.0 for metric in metrics_to_report + ['loss']} 

        results = {'loss': loss, 'accuracy': accuracy}
        
        probas = self.predict_proba(X)
        preds_binary = (probas > 0.5).astype(int)

        # Check for single-label cases to suppress sklearn warnings
        if len(np.unique(y)) <= 1 or len(np.unique(preds_binary)) <= 1:
            results.update({
                'balanced_accuracy': 0.0,
                'f1_score': 0.0,
                'matthews_corrcoef': 0.0
            })
        else:
            if 'balanced_accuracy' in metrics_to_report:
                results['balanced_accuracy'] = balanced_accuracy_score(y, preds_binary)
            if 'f1_score' in metrics_to_report:
                results['f1_score'] = f1_score(y, preds_binary, labels=[0, 1], average='binary', zero_division=0)
            if 'matthews_corrcoef' in metrics_to_report:
                try:
                    results['matthews_corrcoef'] = matthews_corrcoef(y, preds_binary)
                except ValueError:
                    results['matthews_corrcoef'] = 0.0 
                
        return results

class FastModel(BaseModel):
    """
    Fast learning model with a simple architecture.

    Inherits from `BaseModel` and defines a lightweight neural network suitable for quick
    adaptation to concept drifts.

    Args:
        img_height (int): Height of input images.
        img_width (int): Width of input images.
        n_channels (int): Number of channels in input images.
        debug_random_output (bool, optional): If True, returns random outputs for debugging.
                                            Defaults to False.
    """
    def _build_model(self):
        """
        Build the architecture for the fast model.

        Uses a simple architecture with a single dense layer after flattening the input image.

        Returns:
            tf.keras.Model: Compiled Keras model with binary cross-entropy loss.
        """
        model = models.Sequential([
            tf.keras.Input(shape=self.img_shape),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

class SlowModel(BaseModel):
    """
    Slow learning model with a deeper architecture.

    Inherits from `BaseModel` and defines a more complex neural network for stable learning
    over longer periods.

    Args:
        img_height (int): Height of input images.
        img_width (int): Width of input images.
        n_channels (int): Number of channels in input images.
        debug_random_output (bool, optional): If True, returns random outputs for debugging.
                                            Defaults to False.
    """
    def _build_model(self):
        """
        Build the architecture for the slow model.

        Uses a convolutional architecture with multiple layers for robust feature extraction.

        Returns:
            tf.keras.Model: Compiled Keras model with binary cross-entropy loss.
        """
        model = models.Sequential([
            tf.keras.Input(shape=self.img_shape),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

class Orchestrator:
    """
    Combines predictions from fast and slow models using adaptive or fixed weights.

    Manages the combination of predictions from `FastModel` and `SlowModel` using a beta
    parameter, which can be fixed or adaptively updated based on model errors.

    Args:
        fast_model (BaseModel): Fast learning model instance.
        slow_model (BaseModel): Slow learning model instance.
        mu_a (float): Learning rate for the adaptation parameter 'a'.
        theta (float): Scaling factor for the sigmoid function in beta calculation.
        gamma (float): Decay factor for error tracking.
        a0 (float): Initial value for the adaptation parameter 'a'.
        p0 (float): Initial value for the prediction error variance 'p'.
        cost_function_type (str, optional): Type of cost function ('quadratic' or 'cross_entropy').
                                          Defaults to 'quadratic'.
        debug_mode (bool, optional): If True, prints debug information. Defaults to False.
    """
    def __init__(self, fast_model, slow_model, mu_a, theta, gamma, 
                 a0, p0, cost_function_type="quadratic", debug_mode=False):
        if cost_function_type not in ["quadratic", "cross_entropy"]:
            raise ValueError("cost_function_type must be 'quadratic' or 'cross_entropy'")
        self.fast_model = fast_model
        self.slow_model = slow_model
        self._initial_beta_display = 0.5
        self.mu_a = mu_a
        self.theta = theta
        self.gamma = gamma
        self.a = a0
        self.p = p0
        self.e12 = 0
        self.e22 = 0
        self.cost_function_type = cost_function_type
        self.debug_mode = debug_mode
        self.max_delta_a = 1.5

    def get_beta(self):
        """
        Compute the beta weight for combining fast and slow model predictions.

        Uses a logistic function to ensure beta is between 0 and 1.

        Returns:
            float: Beta weight (0 to 1), where higher values favor the fast model.
        """
        return 1 / (1 + np.exp(-self.theta * self.a))

    def predict_proba(self, X, mode, fixed_beta=0.5):
        """
        Predict probabilities using the specified mode.

        Args:
            X (np.ndarray): Input images with shape (num_samples, height, width, channels).
            mode (str): Simulation mode (e.g., MODE_FAST_ONLY, MODE_ADAPTIVE_BETA).
            fixed_beta (float, optional): Fixed beta value for MODE_FIXED_BETA. Defaults to 0.5.

        Returns:
            np.ndarray: Combined predicted probabilities with shape (num_samples,).
        """
        if not self.fast_model.is_trained and mode not in [MODE_SLOW_ONLY]:
            return np.array([0.5] * len(X))
        if not self.slow_model.is_trained and mode not in [MODE_FAST_ONLY]:
            return np.array([0.5] * len(X))

        if mode == MODE_FAST_ONLY:
            return self.fast_model.predict_proba(X)
        elif mode == MODE_SLOW_ONLY:
            return self.slow_model.predict_proba(X)
        else:
            fast_probas = self.fast_model.predict_proba(X)
            slow_probas = self.slow_model.predict_proba(X)
            current_beta = fixed_beta if mode == MODE_FIXED_BETA else self.get_beta()
            combined_probas = current_beta * fast_probas + (1 - current_beta) * slow_probas
            return combined_probas.flatten()

    def evaluate(self, X, y, mode, fixed_beta=0.5):
        """
        Evaluate the orchestrator's performance.

        Combines predictions based on the mode and computes metrics, handling single-label cases
        to suppress sklearn warnings.

        Args:
            X (np.ndarray): Input images with shape (num_samples, height, width, channels).
            y (np.ndarray): True labels with shape (num_samples,).
            mode (str): Simulation mode (e.g., MODE_FAST_ONLY, MODE_ADAPTIVE_BETA).
            fixed_beta (float, optional): Fixed beta value for MODE_FIXED_BETA. Defaults to 0.5.

        Returns:
            dict: Evaluation results with keys from `metrics_to_report` plus 'loss'.
        """
        if len(X) == 0:
            return {metric: 0.0 for metric in metrics_to_report + ['loss']}

        y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
        probas = self.predict_proba(X, mode=mode, fixed_beta=fixed_beta)
        probas = probas.flatten()
        combined_preds_binary = (probas > 0.5).astype(int)

        loss = tf.keras.losses.binary_crossentropy(y_tf, probas).numpy().mean()
        results = {'loss': loss}

        # Check for single-label cases to suppress sklearn warnings
        if len(np.unique(y)) <= 1 or len(np.unique(combined_preds_binary)) <= 1:
            results.update({
                'accuracy': accuracy_score(y, combined_preds_binary),
                'balanced_accuracy': 0.0,
                'f1_score': 0.0,
                'matthews_corrcoef': 0.0
            })
        else:
            if 'accuracy' in metrics_to_report:
                results['accuracy'] = accuracy_score(y, combined_preds_binary)
            if 'balanced_accuracy' in metrics_to_report:
                results['balanced_accuracy'] = balanced_accuracy_score(y, combined_preds_binary)
            if 'f1_score' in metrics_to_report:
                results['f1_score'] = f1_score(y, combined_preds_binary, zero_division=0)
            if 'matthews_corrcoef' in metrics_to_report:
                try:
                    results['matthews_corrcoef'] = matthews_corrcoef(y, combined_preds_binary)
                except ValueError:
                    results['matthews_corrcoef'] = 0.0

        return results

    def update_weights(self, y_true_batch, X_current_batch):
        """
        Update the adaptation parameter 'a' based on model predictions and true labels.

        Uses batch processing to compute the average gradient for updating the beta weight.

        Args:
            y_true_batch (np.ndarray): True labels for the batch with shape (batch_size,).
            X_current_batch (np.ndarray): Input images for the batch with shape
                                         (batch_size, height, width, channels).
        """
        if not self.fast_model or not self.slow_model or \
           not getattr(self.fast_model, 'is_trained', False) or \
           not getattr(self.slow_model, 'is_trained', False):
            if self.debug_mode:
                print("Modelos no entrenados, omitiendo actualización de pesos del orquestador.")
            return

        X_processed_batch = tf.convert_to_tensor(X_current_batch, dtype=tf.float32)
        if X_processed_batch.ndim == 3:
            if X_processed_batch.shape[0] != len(y_true_batch) and len(y_true_batch) == 1:
                X_processed_batch = tf.expand_dims(X_processed_batch, axis=0)
            if X_processed_batch.ndim == 3:
                X_processed_batch = tf.expand_dims(X_processed_batch, axis=-1)
        
        y_true_batch_tensor = tf.convert_to_tensor(y_true_batch, dtype=tf.float32)
        fast_pred_proba_batch = self.fast_model.predict_proba(X_processed_batch).flatten()
        slow_pred_proba_batch = self.slow_model.predict_proba(X_processed_batch).flatten()
        beta_t = self.get_beta()
        combined_pred_proba_batch = beta_t * fast_pred_proba_batch + (1 - beta_t) * slow_pred_proba_batch
        clipped_combined_pred_proba_batch = np.clip(combined_pred_proba_batch, epsilon, 1 - epsilon)

        z1_t_batch = np.where(y_true_batch_tensor == 1, fast_pred_proba_batch, 1 - fast_pred_proba_batch)
        z2_t_batch = np.where(y_true_batch_tensor == 1, slow_pred_proba_batch, 1 - slow_pred_proba_batch)
        e1_t_batch = 1 - z1_t_batch
        e2_t_batch = 1 - z2_t_batch
        
        self.e12 = self.gamma * self.e12 + (1 - self.gamma) * np.mean(e1_t_batch**2)
        self.e22 = self.gamma * self.e22 + (1 - self.gamma) * np.mean(e2_t_batch**2)
        delta_e2_batch = ((e2_t_batch - e1_t_batch))**2
        avg_delta_e2_batch = np.mean(delta_e2_batch)
        self.p = self.gamma * self.p + (1 - self.gamma) * avg_delta_e2_batch
        self.p = max(self.p, epsilon)

        beta_deriv_term = beta_t * (1 - beta_t) * self.theta
        delta_pred_proba_batch = fast_pred_proba_batch - slow_pred_proba_batch
        gradient_yhat_comb_batch = np.zeros_like(combined_pred_proba_batch)

        if self.cost_function_type == "quadratic":
            gradient_yhat_comb_batch = (combined_pred_proba_batch - y_true_batch_tensor)
        elif self.cost_function_type == "cross_entropy":
            gradient_yhat_comb_batch = (clipped_combined_pred_proba_batch - y_true_batch_tensor) / \
                                       (clipped_combined_pred_proba_batch * (1 - clipped_combined_pred_proba_batch))

        gradient_da_per_sample = gradient_yhat_comb_batch * delta_pred_proba_batch * beta_deriv_term
        mean_gradient_da = np.mean(gradient_da_per_sample)
        adaptation_factor = (self.mu_a / self.p)
        delta_a = -adaptation_factor * mean_gradient_da
        delta_a = np.clip(delta_a, -self.max_delta_a, self.max_delta_a)
        self.a = self.a + delta_a

        if self.debug_mode:
            print(f'\n--- Debug Info (Coste: {self.cost_function_type}) ---')
            print(f'mu_a = {self.mu_a:.4f}')
            print(f'theta = {self.theta:.4f}')
            print(f'gamma = {self.gamma:.4f}')
            print(f'e_fast(t)**2 = {self.e12:.4f}') 
            print(f'e_slow(t)**2 = {self.e22:.4f}') 
            print(f'delta_e2 = {avg_delta_e2_batch:.8f}')
            print(f'p(t) = {self.p:.4f}')
            print(f'delta_a = {delta_a:.8f}')
            if self.a < 0:
                if self.e12 < self.e22:
                    print(f'a(t) = {self.a:.4f}. a<0 y e_fast < e_slow -> favorece rápido')
                else:
                    print(f'a(t) = {self.a:.4f}. a<0 y e_slow < e_fast -> favorece lento')
            else:
                if self.e12 < self.e22:
                    print(f'a(t) = {self.a:.4f}. a>0 y e_fast < e_slow -> favorece rápido')
                else:
                    print(f'a(t) = {self.a:.4f}. a>0 y e_slow < e_fast -> favorece lento')
            if beta_t < 0.5:
                print(f'beta(t) = {self.get_beta():.4f}; modelo lento favorecido')
            else:
                print(f'beta(t) = {self.get_beta():.4f}; modelo rápido favorecido')
            print('--- End Debug Info ---')
            print()