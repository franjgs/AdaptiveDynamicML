#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 10:37:01 2025

@author: fran
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, accuracy_score
import abc # Importar el módulo abc para clases abstractas

# import pdb

# --- CONFIGURACIÓN GLOBAL (PARA ESTOS MODELOS) ---
# Métricas a reportar en la evaluación de modelos
metrics_to_report = [
    'accuracy',
    'balanced_accuracy',
    'f1_score',
    'matthews_corrcoef'
]

# Modos de operación del Orquestador
MODE_FAST_ONLY = "fast_only"
MODE_SLOW_ONLY = "slow_only"
MODE_INDEPENDENT = "independent" # No se usa directamente para la combinación, pero es un modo lógico
MODE_FIXED_BETA = "fixed_beta"
MODE_ADAPTIVE_BETA = "adaptive_beta"

class BaseModel(abc.ABC):
    """
    Clase base abstracta para modelos de aprendizaje.
    Contiene la lógica común para el entrenamiento, predicción y evaluación.
    """
    def __init__(self, img_height, img_width, n_channels, debug_random_output=False):
        self.img_shape = (img_height, img_width, n_channels)
        self.model = self._build_model()
        self.is_trained = False
        self.debug_random_output = debug_random_output

    @abc.abstractmethod
    def _build_model(self):
        """
        Método abstracto para construir la arquitectura específica del modelo.
        Debe ser implementado por las subclases.
        """
        pass

    def train(self, X_train, y_train, epochs, batch_size, validation_split):
        """Entrena el modelo con los datos proporcionados."""
        if self.debug_random_output:
            self.is_trained = True
            return {'loss': 0.5, 'accuracy': 0.5, 'val_loss': 0.5, 'val_accuracy': 0.5}

        # Convertir a tensores primero
        X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
        
        # Ahora, verificar si la dimensión del lote (primer elemento de shape) es 0
        if X_train_tensor.shape[0] == 0:
            print(f"  [{self.__class__.__name__}] No hay datos válidos para entrenar el modelo (X_train_tensor es vacío).")
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
        """Realiza predicciones de probabilidad."""
        if self.debug_random_output:
            return np.random.rand(len(X))

        if not self.is_trained:
            return np.array([0.5] * len(X))

        return self.model.predict(X, verbose=0).flatten()

    def evaluate(self, X, y):
        """Evalúa el rendimiento del modelo."""
        # Asegúrate de que y es un array de numpy para evitar problemas con sklearn
        y = np.asarray(y)

        # --- Debug Random Output Mode ---
        if self.debug_random_output:
            n_samples = len(X)
            if n_samples == 0:
                # Asegúrate de que metrics_to_report esté disponible aquí
                return {metric: 0.0 for metric in metrics_to_report + ['loss']}

            random_probas = np.random.rand(n_samples)
            random_preds_binary = (random_probas > 0.5).astype(int)
            
            # Asegurar que las etiquetas para binary_crossentropy sean float32
            simulated_loss = tf.keras.losses.binary_crossentropy(
                tf.convert_to_tensor(y, dtype=tf.float32), 
                tf.convert_to_tensor(random_probas, dtype=tf.float32)
            ).numpy().mean()

            results = {
                'loss': simulated_loss, 
                'accuracy': accuracy_score(y, random_preds_binary, normalize=True)
            }
            if 'balanced_accuracy' in metrics_to_report:
                # ELIMINAR 'labels' de aquí
                results['balanced_accuracy'] = balanced_accuracy_score(y, random_preds_binary)
            if 'f1_score' in metrics_to_report:
                # Mantener 'labels' y 'zero_division' para f1_score
                results['f1_score'] = f1_score(y, random_preds_binary, labels=[0, 1], average='binary', zero_division=0)
            if 'matthews_corrcoef' in metrics_to_report:
                try:
                    results['matthews_corrcoef'] = matthews_corrcoef(y, random_preds_binary)
                except ValueError:
                    results['matthews_corrcoef'] = 0.0 
            return results

        # --- Normal Model Evaluation Mode ---
        if not self.is_trained or len(X) == 0:
            # Asegúrate de que metrics_to_report esté disponible aquí
            return {metric: 0.0 for metric in metrics_to_report + ['loss']}

        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

        try:
            eval_results = self.model.evaluate(X_tensor, y_tensor, verbose=0, return_dict=True)
            loss = eval_results.get('loss', 0.0)
            accuracy = eval_results.get('accuracy', 0.0)
        except Exception as e:
            print(f"Advertencia durante la evaluación del modelo: {e}")
            # Asegúrate de que metrics_to_report esté disponible aquí
            return {metric: 0.0 for metric in metrics_to_report + ['loss']} 

        results = {'loss': loss, 'accuracy': accuracy}
        
        probas = self.predict_proba(X)
        preds_binary = (probas > 0.5).astype(int)

        if 'balanced_accuracy' in metrics_to_report:
            # ELIMINAR 'labels' de aquí también
            results['balanced_accuracy'] = balanced_accuracy_score(y, preds_binary)
        if 'f1_score' in metrics_to_report:
            # Mantener 'labels' y 'zero_division' para f1_score
            results['f1_score'] = f1_score(y, preds_binary, labels=[0, 1], average='binary', zero_division=0)
        if 'matthews_corrcoef' in metrics_to_report:
            try:
                results['matthews_corrcoef'] = matthews_corrcoef(y, preds_binary)
            except ValueError:
                results['matthews_corrcoef'] = 0.0 
                
        return results


class FastModel(BaseModel):
    """
    Representa un modelo de aprendizaje rápido con una arquitectura simple.
    Hereda la lógica de entrenamiento, predicción y evaluación de BaseModel.
    """
    def _build_model(self):
        """
        Implementación específica de la arquitectura del modelo rápido.
        """
        # Opción 0
        """
        model = models.Sequential([
                tf.keras.Input(shape=self.img_shape),
                layers.Conv2D(16, (3, 3), activation='relu'), # Reducir filtros
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(32, activation='relu'), # Reducir neuronas
                layers.Dense(1, activation='sigmoid')
            ])
        """
        # Opción 1
        """
        model = models.Sequential([
                tf.keras.Input(shape=self.img_shape),
                # Opción 1: Muy superficial con una sola Conv2D pequeña
                layers.Conv2D(8, (3, 3), activation='relu', padding='same'), # Menos filtros, padding='same' para mantener tamaño
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(16, activation='relu'), # Menos neuronas
                layers.Dense(1, activation='sigmoid')
            ])
        """
        # Opción 2 (Aún más superficial): Sin capas convolucionales, solo densas sobre la imagen aplanada
        # Requiere aplanar la imagen de entrada directamente:
        model = models.Sequential([
             tf.keras.Input(shape=self.img_shape),
             layers.Flatten(), # Aplanar la imagen directamente
             layers.Dense(32, activation='relu'), # Capa densa relativamente pequeña
             layers.Dense(1, activation='sigmoid')
        ])
        # Un learning rate aún más bajo para hacerlo más 'lento' en adaptarse
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Más alto
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

class SlowModel(BaseModel):
    """
    Representa un modelo de aprendizaje lento con una arquitectura más compleja.
    Hereda la lógica de entrenamiento, predicción y evaluación de BaseModel.
    """
    def _build_model(self):
        """
        Implementación específica de la arquitectura del modelo lento.
        """
        # Opción 0
        model = models.Sequential([
            tf.keras.Input(shape=self.img_shape),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        # Opción 1
        """
        model = models.Sequential([
            tf.keras.Input(shape=self.img_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(), 
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(), 
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.3), 
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid') # Correcto para binario
        ])
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # Más bajo que el modelo rápido
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

class Orchestrator:
    def __init__(self, fast_model, slow_model, mu_a, theta, gamma, 
                 a0, p0, cost_function_type="quadratic", debug_mode=False):
        """
        Inicializa el Orquestador.

        Args:
            mu_a (float): Tasa de aprendizaje para el parámetro 'a'.
            theta (float): Factor de escala para la función sigmoide.
            gamma (float): Factor de decaimiento para la media móvil de 'p'.
            cost_function_type (str): Tipo de función de coste a usar ("quadratic" o "cross_entropy").
            debug_mode (bool): Si es True, imprime información de depuración.
        """
        # Asume que fast_model y slow_model serán asignados después de la inicialización
        self.fast_model = fast_model
        self.slow_model = slow_model
        
        self._initial_beta_display = 0.5
        
        self.mu_a = mu_a
        self.theta = theta
        self.gamma = gamma
        self.a = a0  # Parámetro inicial 'a'
        self.p = p0  # Parámetro inicial 'p' para el denominador de la tasa de aprendizaje adaptativa
        
        self.e12 = 0
        self.e22 = 0

        # Nuevo parámetro para seleccionar la función de coste
        if cost_function_type not in ["quadratic", "cross_entropy"]:
            raise ValueError("cost_function_type debe ser 'quadratic' o 'cross_entropy'")
        self.cost_function_type = cost_function_type
        self.debug_mode = debug_mode
        
        self.max_delta_a = 1.5

    def get_beta(self):
        # La función logística asegura que beta esté entre 0 y 1
        return 1 / (1 + np.exp(-self.theta * self.a))

    def predict_proba(self, X):
        if not self.fast_model or not self.slow_model or \
           not getattr(self.fast_model, 'is_trained', False) or \
           not getattr(self.slow_model, 'is_trained', False):
            # Si los modelos no están entrenados, o no se han asignado,
            # podemos retornar una predicción por defecto o lanzar un error.
            # Para la simulación, podemos retornar 0.5 o la predicción de un modelo dummy si existe.
            # Aquí, retornamos 0.5 para la probabilidad.
            # Esto se puede refinar dependiendo del comportamiento deseado al inicio.
            if self.debug_mode:
                print("Advertencia: Intentando predecir con modelos no entrenados en el orquestador. Retornando 0.5.")
            # Necesitamos un array de la misma forma que X si es para múltiples muestras
            if isinstance(X, tf.Tensor) and X.ndim > 1:
                return np.full((X.shape[0],), 0.5)
            else:
                return 0.5
        
        # Las predicciones de los modelos ya deberían ser probabilidades para la clase positiva
        fast_pred_proba = self.fast_model.predict_proba(X)
        slow_pred_proba = self.slow_model.predict_proba(X)

        beta_t = self.get_beta()
        combined_pred_proba = beta_t * fast_pred_proba + (1 - beta_t) * slow_pred_proba
        return combined_pred_proba

    def evaluate(self, X_eval, y_eval, mode, fixed_beta=0.5, metrics=['accuracy']):
        """
        Evalúa el orquestador.
        Args:
            X_eval (np.array): Muestras de entrada para la evaluación.
            y_eval (np.array): Etiquetas verdaderas para la evaluación.
            mode (str): Modo de simulación (e.g., MODE_FIXED_BETA, MODE_ADAPTIVE_BETA).
            fixed_beta (float): Valor de beta si el modo es FIXED_BETA.
            metrics (list): Lista de métricas a calcular.
        Returns:
            dict: Diccionario con los resultados de la evaluación.
        """
        results = {metric: 0.0 for metric in metrics}
        if len(X_eval) == 0:
            return results

        # Obtener predicciones de los modelos subyacentes
        # Asegurarse de que las predicciones sean np.array para operaciones matriciales
        fast_preds = self.fast_model.predict_proba(X_eval).flatten()
        slow_preds = self.slow_model.predict_proba(X_eval).flatten()

        combined_preds = np.zeros_like(fast_preds)
        
        if mode == "FIXED_BETA":
            beta_val = fixed_beta
        elif mode == "ADAPTIVE_BETA":
            beta_val = self.get_beta()
        else:
            # En otros modos como FAST_ONLY o SLOW_ONLY, esta evaluación del orquestador no se debería llamar
            # O se comportaría como uno de los modelos (ya gestionado en el bucle principal).
            return results 
        
        combined_preds = beta_val * fast_preds + (1 - beta_val) * slow_preds
        
        # Calcular métricas (ejemplo para accuracy)
        # Convertir probabilidades combinadas a clases binarias (0 o 1)
        binary_predictions = (combined_preds >= 0.5).astype(int)
        
        if 'accuracy' in metrics:
            accuracy = np.mean(binary_predictions == y_eval)
            results['accuracy'] = accuracy
        
        # Aquí podrías añadir más métricas si lo deseas (precisión, recall, F1-score, etc.)

        return results

    # *** MÉTODO update_weights MODIFICADO PARA ACEPTAR BATCHES ***
    def update_weights(self, y_true_batch, X_current_batch):
        """
        Actualiza el parámetro 'a' del orquestador basándose en las predicciones
        de los modelos rápido y lento, y las etiquetas verdaderas de un batch.
        Calcula el gradiente promedio sobre el batch.
        """
        if not self.fast_model or not self.slow_model or \
           not getattr(self.fast_model, 'is_trained', False) or \
           not getattr(self.slow_model, 'is_trained', False):
            if self.debug_mode:
                print("Modelos no entrenados, omitiendo actualización de pesos del orquestador.")
            return

        epsilon = tf.keras.backend.epsilon() # Evitar división por cero o logaritmos de cero/uno
        
        # Convertir a tensores si no lo son y asegurar dimensión de batch
        X_processed_batch = tf.convert_to_tensor(X_current_batch, dtype=tf.float32)
        if X_processed_batch.ndim == 3: # Asume formato de imagen (alto, ancho, canales)
            # Si el batch tiene una sola imagen y no está expandida, expandirla
            if X_processed_batch.shape[0] != len(y_true_batch) and len(y_true_batch) == 1:
                 X_processed_batch = tf.expand_dims(X_processed_batch, axis=0) # Añadir dimensión de batch
            elif X_processed_batch.shape[0] == len(y_true_batch) and X_processed_batch.ndim == 3: # Si ya es un batch de imágenes sin canal
                # Asumimos que X_current_batch ya está formateado correctamente para el batch
                pass
            # Si es un batch de imágenes, pero con un solo canal, Keras espera (batch_size, alto, ancho, 1)
            # Si ya está en (batch_size, alto, ancho, canales), no necesitamos expand_dims
            # Este es un punto a verificar según la entrada real de tus modelos.
            # Por simplicidad, asumimos que si es 3D es (batch, H, W) y necesitamos (batch, H, W, C)
            if X_processed_batch.ndim == 3: # Si X_current_batch tenía (batch_size, height, width)
                 X_processed_batch = tf.expand_dims(X_processed_batch, axis=-1) # Añadir dimensión de canal
        
        y_true_batch_tensor = tf.convert_to_tensor(y_true_batch, dtype=tf.float32)

        # Obtener las probabilidades predichas por cada modelo para el batch
        fast_pred_proba_batch = self.fast_model.predict_proba(X_processed_batch).flatten()
        slow_pred_proba_batch = self.slow_model.predict_proba(X_processed_batch).flatten()
        
        # Calcular el peso beta actual del orquestador (este beta es para todo el batch)
        beta_t = self.get_beta()

        # Calcular la predicción combinada para todo el batch
        combined_pred_proba_batch = beta_t * fast_pred_proba_batch + (1 - beta_t) * slow_pred_proba_batch
        clipped_combined_pred_proba_batch = np.clip(combined_pred_proba_batch, epsilon, 1 - epsilon)

        # Calcular las "certezas" (likelihood) y errores para el cálculo de `self.p`
        # Este `delta_e2` se usará para actualizar `self.p`
        z1_t_batch = np.where(y_true_batch_tensor == 1, fast_pred_proba_batch, 1 - fast_pred_proba_batch)
        z2_t_batch = np.where(y_true_batch_tensor == 1, slow_pred_proba_batch, 1 - slow_pred_proba_batch)
        
        e1_t_batch = 1 - z1_t_batch
        e2_t_batch = 1 - z2_t_batch
        
        self.e12 =  self.gamma * self.e12 + (1 - self.gamma) * np.mean(e1_t_batch**2)
        self.e22 =  self.gamma * self.e22 + (1 - self.gamma) * np.mean(e2_t_batch**2)

        # Delta de error cuadrático (instantáneo) para cada muestra del batch
        # Esto es lo que se usa en el EMA de `self.p`
        delta_e2_batch = ((e2_t_batch - e1_t_batch))**2
        
        # Usar el promedio de delta_e2 del batch para actualizar self.p
        avg_delta_e2_batch = np.mean(delta_e2_batch)
        self.p = self.gamma * self.p + (1 - self.gamma) * avg_delta_e2_batch
        self.p = max(self.p, epsilon) # Asegurar que 'p' no sea cero
        

        # Término común del gradiente relacionado con beta y la diferencia de modelos
        # d(beta)/da = beta * (1 - beta) * theta
        beta_deriv_term = beta_t * (1 - beta_t) * self.theta
        
        # d(y_hat_comb)/d(beta) = (fast_pred_proba - slow_pred_proba) para cada muestra
        delta_pred_proba_batch = fast_pred_proba_batch - slow_pred_proba_batch 

        # --- Gradiente dJ/dŷ_comb según tipo de coste (calculado para cada muestra del batch) ---
        gradient_yhat_comb_batch = np.zeros_like(combined_pred_proba_batch)

        if self.cost_function_type == "quadratic":
            # dJ/d(y_hat_comb) = (y_hat_comb - y_true) para cada muestra
            gradient_yhat_comb_batch = (combined_pred_proba_batch - y_true_batch_tensor)
        elif self.cost_function_type == "cross_entropy":
            # dJ/d(y_hat_comb) = (y_hat_comb - y_true) / (y_hat_comb * (1 - y_hat_comb)) para cada muestra
            gradient_yhat_comb_batch = (clipped_combined_pred_proba_batch - y_true_batch_tensor) / \
                                       (clipped_combined_pred_proba_batch * (1 - clipped_combined_pred_proba_batch))
                                       
            """
            if np.any(np.isnan(gradient_yhat_comb_batch)) or np.any(np.isinf(gradient_yhat_comb_batch)):
                print(f"DEBUG: gradient_yhat_comb_batch contiene NaN/Inf. Deteniendo ejecución para inspeccionar.")
                pdb.set_trace() # Detiene aquí si el gradiente de la función de coste es NaN/Inf
            """
        else:
            raise ValueError(f"Tipo de función de coste no soportado: {self.cost_function_type}")

        # Calcular el gradiente total dJ/da para cada muestra del batch
        # dJ/da = (dJ/d_y_hat_comb) * (d_y_hat_comb/d_beta) * (d_beta/da)
        # Note: beta_deriv_term es el mismo para todo el batch
        gradient_da_per_sample = gradient_yhat_comb_batch * delta_pred_proba_batch * beta_deriv_term
        
        # Promediar el gradiente sobre el batch para obtener un único gradiente para la actualización
        mean_gradient_da = np.mean(gradient_da_per_sample)
        
        # Calcular el factor de adaptación final (learning rate adaptativo)
        adaptation_factor = (self.mu_a / self.p)

        # Calcular delta_a, el cambio a sumar a 'a'
        # delta_a = -learning_rate * dJ/da
        delta_a = -adaptation_factor * mean_gradient_da
        
        # Clipping de delta_a para prevenir saltos bruscos
        delta_a = np.clip(delta_a, -self.max_delta_a, self.max_delta_a)
        
        # Actualizar el parámetro 'a'
        self.a = self.a + delta_a

        # --- Modo de Depuración ---
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
            
            # Lógica de favorecimiento basada en 'a' y errores
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
            
            # Se muestra la preferencia del modelo basada en beta_t
            if beta_t < 0.5:
                print(f'beta(t) = {self.get_beta():.4f}; modelo lento favorecido')
            else:
                print(f'beta(t) = {self.get_beta():.4f}; modelo rápido favorecido')
            print('--- End Debug Info ---')
            print()

    def predict_proba(self, X, mode, fixed_beta=0.5):
        """
        Realiza predicciones de probabilidad utilizando el modelo o combinación de modelos
        especificado por el modo.
        """
        if not self.fast_model.is_trained and mode not in [MODE_SLOW_ONLY]:
            return np.array([0.5] * len(X))
        if not self.slow_model.is_trained and mode not in [MODE_FAST_ONLY]:
            return np.array([0.5] * len(X))

        if mode == MODE_FAST_ONLY:
            return self.fast_model.predict_proba(X)
        elif mode == MODE_SLOW_ONLY:
            return self.slow_model.predict_proba(X)
        else: # MODE_INDEPENDENT, MODE_FIXED_BETA, MODE_ADAPTIVE_BETA
            fast_probas = self.fast_model.predict_proba(X)
            slow_probas = self.slow_model.predict_proba(X)

            if mode == MODE_FIXED_BETA:
                current_beta = fixed_beta
            elif mode == MODE_ADAPTIVE_BETA:
                current_beta = self.get_beta()
            else:
                current_beta = 0.5
            
            combined_probas = current_beta * fast_probas + (1 - current_beta) * slow_probas
            return combined_probas.flatten()


    def evaluate(self, X, y, mode, fixed_beta=0.5):
        """
        Evalúa el rendimiento de la predicción del Orquestador (o de un modelo individual)
        para un conjunto de datos dado, devolviendo un diccionario de métricas.
        """
        if len(X) == 0:
            return {metric: 0.0 for metric in metrics_to_report + ['loss']}

        y_tf = tf.convert_to_tensor(y, dtype=tf.float32)

        probas = self.predict_proba(X, mode=mode, fixed_beta=fixed_beta)
        
        probas = probas.flatten()
        combined_preds_binary = (probas > 0.5).astype(int)

        loss = tf.keras.losses.binary_crossentropy(y_tf, probas).numpy().mean()

        results = {'loss': loss}

        if 'accuracy' in metrics_to_report:
            results['accuracy'] = accuracy_score(y, combined_preds_binary)
        if 'balanced_accuracy' in metrics_to_report:
            results['balanced_accuracy'] = balanced_accuracy_score(y, combined_preds_binary)
        if 'f1_score' in metrics_to_report:
            results['f1_score'] = f1_score(y, combined_preds_binary, zero_division=0)
        if 'matthews_corrcoef' in metrics_to_report:
            results['matthews_corrcoef'] = matthews_corrcoef(y, combined_preds_binary)

        return results