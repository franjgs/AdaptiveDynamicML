#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of fast, slow, and orchestrator models for adaptive machine learning.

This module defines the FastModel, SlowModel, and Orchestrator classes for handling
concept drift in data streams, inspired by "Generalized CMAC Adaptive Ensembles for
Concept-Drifting Data Streams" (González-Serrano & Figueiras-Vidal, EUSIPCO 2017).

Created on Wed Jul 2 10:37:01 2025
Author: Fran
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, accuracy_score
import abc

# Global configuration for model evaluation metrics
METRICS_TO_REPORT = [
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
    """Abstract base class for machine learning models.

    Provides common functionality for training, prediction, and evaluation of models.
    Subclasses must implement the `_build_model` method to define specific architectures.
    """

    def __init__(self, img_height, img_width, n_channels, debug_random_output=False):
        """Initialize the base model.

        Args:
            img_height (int): Height of input images.
            img_width (int): Width of input images.
            n_channels (int): Number of channels in input images (e.g., 1 for grayscale).
            debug_random_output (bool, optional): If True, returns random outputs for debugging.
        """
        self.img_shape = (img_height, img_width, n_channels)
        self.model = self._build_model()
        self.is_trained = False
        self.debug_random_output = debug_random_output

    @abc.abstractmethod
    def _build_model(self):
        """Define the specific architecture of the model.

        Must be implemented by subclasses.

        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        pass

    def train(self, X_train, y_train, epochs, batch_size, validation_split):
        """Train the model on the provided data.

        Args:
            X_train (np.ndarray): Training input data of shape (samples, height, width, channels).
            y_train (np.ndarray): Training labels.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of data to use for validation (0 to 1).

        Returns:
            dict: Training metrics including loss, accuracy, validation loss, and validation accuracy.
        """
        if self.debug_random_output:
            self.is_trained = True
            return {'loss': 0.5, 'accuracy': 0.5, 'val_loss': 0.5, 'val_accuracy': 0.5}

        X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)

        if X_train_tensor.shape[0] == 0:
            print(f"[{self.__class__.__name__}] No valid training data (X_train_tensor is empty).")
            return {'loss': 0.0, 'accuracy': 0.0, 'val_loss': 0.0, 'val_accuracy': 0.0}

        if y_train_tensor.ndim == 1:
            y_train_tensor = tf.expand_dims(y_train_tensor, axis=-1)

        history = self.model.fit(
            X_train_tensor,
            y_train_tensor,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )

        self.is_trained = True
        return {
            'loss': history.history['loss'][-1],
            'accuracy': history.history['accuracy'][-1],
            'val_loss': history.history.get('val_loss', [0.0])[-1],
            'val_accuracy': history.history.get('val_accuracy', [0.0])[-1]
        }

    def predict_proba(self, X):
        """Predict probabilities for input data.

        Args:
            X (np.ndarray): Input data of shape (samples, height, width, channels).

        Returns:
            np.ndarray: Predicted probabilities for the positive class.
        """
        if self.debug_random_output:
            return np.random.rand(len(X))

        if not self.is_trained:
            return np.array([0.5] * len(X))

        return self.model.predict(X, verbose=0).flatten()

    def evaluate(self, X, y):
        """Evaluate the model's performance on the provided data.

        Args:
            X (np.ndarray): Input data for evaluation.
            y (np.ndarray): True labels for evaluation.

        Returns:
            dict: Evaluation metrics including loss and metrics specified in METRICS_TO_REPORT.
        """
        y = np.asarray(y)

        if self.debug_random_output:
            if len(X) == 0:
                return {metric: 0.0 for metric in METRICS_TO_REPORT + ['loss']}

            random_probas = np.random.rand(len(X))
            random_preds_binary = (random_probas > 0.5).astype(int)
            simulated_loss = tf.keras.losses.binary_crossentropy(
                tf.convert_to_tensor(y, dtype=tf.float32),
                tf.convert_to_tensor(random_probas, dtype=tf.float32)
            ).numpy().mean()

            results = {
                'loss': simulated_loss,
                'accuracy': accuracy_score(y, random_preds_binary, normalize=True)
            }
            if 'balanced_accuracy' in METRICS_TO_REPORT:
                results['balanced_accuracy'] = balanced_accuracy_score(y, random_preds_binary)
            if 'f1_score' in METRICS_TO_REPORT:
                results['f1_score'] = f1_score(y, random_preds_binary, labels=[0, 1], average='binary', zero_division=0)
            if 'matthews_corrcoef' in METRICS_TO_REPORT:
                try:
                    results['matthews_corrcoef'] = matthews_corrcoef(y, random_preds_binary)
                except ValueError:
                    results['matthews_corrcoef'] = 0.0
            return results

        if not self.is_trained or len(X) == 0:
            return {metric: 0.0 for metric in METRICS_TO_REPORT + ['loss']}

        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

        try:
            eval_results = self.model.evaluate(X_tensor, y_tensor, verbose=0, return_dict=True)
            loss = eval_results.get('loss', 0.0)
            accuracy = eval_results.get('accuracy', 0.0)
        except Exception as e:
            print(f"Warning during model evaluation: {e}")
            return {metric: 0.0 for metric in METRICS_TO_REPORT + ['loss']}

        results = {'loss': loss, 'accuracy': accuracy}
        probas = self.predict_proba(X)
        preds_binary = (probas > 0.5).astype(int)

        if 'balanced_accuracy' in METRICS_TO_REPORT:
            results['balanced_accuracy'] = balanced_accuracy_score(y, preds_binary)
        if 'f1_score' in METRICS_TO_REPORT:
            results['f1_score'] = f1_score(y, preds_binary, labels=[0, 1], average='binary', zero_division=0)
        if 'matthews_corrcoef' in METRICS_TO_REPORT:
            try:
                results['matthews_corrcoef'] = matthews_corrcoef(y, preds_binary)
            except ValueError:
                results['matthews_corrcoef'] = 0.0

        return results

class FastModel(BaseModel):
    """Fast learning model with a simple architecture.

    Inherits from BaseModel, implementing a lightweight neural network for rapid adaptation
    to new data patterns.
    """

    def _build_model(self):
        """Define the architecture of the fast model.

        Creates a simple neural network with a flattened input and dense layers, optimized
        for quick adaptation using a higher learning rate.

        Returns:
            tf.keras.Model: Compiled Keras model for fast learning.
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
    """Slow learning model with a complex architecture.

    Inherits from BaseModel, implementing a deeper neural network for stable, long-term learning.
    """

    def _build_model(self):
        """Define the architecture of the slow model.

        Creates a convolutional neural network with multiple layers for robust feature extraction,
        using a lower learning rate for stable learning.

        Returns:
            tf.keras.Model: Compiled Keras model for slow learning.
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
    """Orchestrator for combining predictions from fast and slow models.

    Dynamically adjusts the weighting (beta) of fast and slow model predictions based on their
    performance, using adaptive or fixed strategies to handle concept drift.
    """

    def __init__(self, fast_model, slow_model, mu_a, theta, gamma, a0, p0, cost_function_type="quadratic", debug_mode=False):
        """Initialize the Orchestrator.

        Args:
            fast_model (BaseModel): Fast learning model instance.
            slow_model (BaseModel): Slow learning model instance.
            mu_a (float): Learning rate for the 'a' parameter.
            theta (float): Scaling factor for the sigmoid function in beta calculation.
            gamma (float): Decay factor for the moving average of 'p'.
            a0 (float): Initial value for the 'a' parameter.
            p0 (float): Initial value for the 'p' parameter.
            cost_function_type (str): Cost function type ("quadratic" or "cross_entropy").
            debug_mode (bool, optional): If True, prints debugging information.
        """
        self.fast_model = fast_model
        self.slow_model = slow_model
        self.mu_a = mu_a
        self.theta = theta
        self.gamma = gamma
        self.a = a0
        self.p = p0
        self.e12 = 0.0
        self.e22 = 0.0
        self.cost_function_type = cost_function_type
        self.debug_mode = debug_mode
        self.max_delta_a = 1.5

        if cost_function_type not in ["quadratic", "cross_entropy"]:
            raise ValueError("cost_function_type must be 'quadratic' or 'cross_entropy'")

    def get_beta(self):
        """Calculate the beta parameter for weighting fast and slow model predictions.

        Uses a logistic function to ensure beta is between 0 and 1.

        Returns:
            float: Beta value (weight for the fast model).
        """
        return 1 / (1 + np.exp(-self.theta * self.a))

    def predict_proba(self, X, mode, fixed_beta=0.5):
        """Predict probabilities using the specified mode.

        Args:
            X (np.ndarray): Input data for prediction.
            mode (str): Operation mode (e.g., MODE_FAST_ONLY, MODE_ADAPTIVE_BETA).
            fixed_beta (float, optional): Fixed beta value for MODE_FIXED_BETA.

        Returns:
            np.ndarray: Combined or model-specific predicted probabilities.
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
            return current_beta * fast_probas + (1 - current_beta) * slow_probas

    def evaluate(self, X, y, mode, fixed_beta=0.5):
        """Evaluate the orchestrator's performance.

        Args:
            X (np.ndarray): Input data for evaluation.
            y (np.ndarray): True labels for evaluation.
            mode (str): Operation mode for evaluation.
            fixed_beta (float, optional): Fixed beta value for MODE_FIXED_BETA.

        Returns:
            dict: Evaluation metrics including loss and METRICS_TO_REPORT.
        """
        if len(X) == 0:
            return {metric: 0.0 for metric in METRICS_TO_REPORT + ['loss']}

        y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
        probas = self.predict_proba(X, mode=mode, fixed_beta=fixed_beta).flatten()
        combined_preds_binary = (probas > 0.5).astype(int)

        loss = tf.keras.losses.binary_crossentropy(y_tf, probas).numpy().mean()
        results = {'loss': loss}

        if 'accuracy' in METRICS_TO_REPORT:
            results['accuracy'] = accuracy_score(y, combined_preds_binary)
        if 'balanced_accuracy' in METRICS_TO_REPORT:
            results['balanced_accuracy'] = balanced_accuracy_score(y, combined_preds_binary)
        if 'f1_score' in METRICS_TO_REPORT:
            results['f1_score'] = f1_score(y, combined_preds_binary, zero_division=0)
        if 'matthews_corrcoef' in METRICS_TO_REPORT:
            try:
                results['matthews_corrcoef'] = matthews_corrcoef(y, combined_preds_binary)
            except ValueError:
                results['matthews_corrcoef'] = 0.0

        return results

    def update_weights(self, y_true_batch, X_current_batch):
        """Update the orchestrator's 'a' parameter based on batch predictions.

        Computes the average gradient over a batch to adjust the weighting of fast and
        slow model predictions.

        Args:
            y_true_batch (np.ndarray): True labels for the batch.
            X_current_batch (np.ndarray): Input data batch for prediction.
        """
        if not self.fast_model or not self.slow_model or \
           not getattr(self.fast_model, 'is_trained', False) or \
           not getattr(self.slow_model, 'is_trained', False):
            if self.debug_mode:
                print("Models not trained, skipping orchestrator weight update.")
            return

        epsilon = tf.keras.backend.epsilon()
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
        delta_e2_batch = (e2_t_batch - e1_t_batch)**2
        avg_delta_e2_batch = np.mean(delta_e2_batch)
        self.p = max(self.gamma * self.p + (1 - self.gamma) * avg_delta_e2_batch, epsilon)

        beta_deriv_term = beta_t * (1 - beta_t) * self.theta
        delta_pred_proba_batch = fast_pred_proba_batch - slow_pred_proba_batch
        gradient_yhat_comb_batch = np.zeros_like(combined_pred_proba_batch)

        if self.cost_function_type == "quadratic":
            gradient_yhat_comb_batch = combined_pred_proba_batch - y_true_batch_tensor
        elif self.cost_function_type == "cross_entropy":
            gradient_yhat_comb_batch = (clipped_combined_pred_proba_batch - y_true_batch_tensor) / \
                                       (clipped_combined_pred_proba_batch * (1 - clipped_combined_pred_proba_batch))

        gradient_da_per_sample = gradient_yhat_comb_batch * delta_pred_proba_batch * beta_deriv_term
        mean_gradient_da = np.mean(gradient_da_per_sample)
        adaptation_factor = self.mu_a / self.p
        delta_a = -adaptation_factor * mean_gradient_da
        delta_a = np.clip(delta_a, -self.max_delta_a, self.max_delta_a)
        self.a += delta_a

        if self.debug_mode:
            print(f"\n--- Debug Info (Cost Function: {self.cost_function_type}) ---")
            print(f"mu_a = {self.mu_a:.4f}")
            print(f"theta = {self.theta:.4f}")
            print(f"gamma = {self.gamma:.4f}")
            print(f"e_fast(t)^2 = {self.e12:.4f}")
            print(f"e_slow(t)^2 = {self.e22:.4f}")
            print(f"delta_e2 = {avg_delta_e2_batch:.8f}")
            print(f"p(t) = {self.p:.4f}")
            print(f"delta_a = {delta_a:.8f}")
            print(f"a(t) = {self.a:.4f}. {'Favors fast' if self.a > 0 else 'Favors slow'} "
                  f"(e_fast < e_slow: {self.e12 < self.e22})")
            print(f"beta(t) = {self.get_beta():.4f}; {'fast model favored' if beta_t >= 0.5 else 'slow model favored'}")
            print("--- End Debug Info ---")