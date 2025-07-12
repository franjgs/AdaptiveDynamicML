#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Hierarchical Drift Learning (AHDL) simulation using the EMNIST dataset.

This script simulates concept drift handling with fast and slow models combined via an adaptive
orchestrator. It processes EMNIST balanced dataset images, applies concept drifts (sudden and
gradual), trains models, evaluates performance, and generates visualizations. The simulation runs
without a main() function for debugging purposes, logging metrics and saving plots.

@author: fran
"""
import numpy as np
from collections import deque
import tensorflow as tf
import tensorflow_datasets as tfds
import random
import gc

# Import model, utility, and plotting functions
from src.models import FastModel, SlowModel, Orchestrator, metrics_to_report, \
                      MODE_FAST_ONLY, MODE_SLOW_ONLY, MODE_INDEPENDENT, \
                      MODE_FIXED_BETA, MODE_ADAPTIVE_BETA
from src.utils import StreamSample, concept_functions_image, \
                     preprocess_image, add_gaussian_noise, visualize_samples_with_labels
from src.plottings import setup_visualization_points, create_chart, plot_results

# Define epsilon for numerical stability in clipping operations
epsilon = tf.keras.backend.epsilon()

# Configuration dictionary for simulation, image, drift, model, and visualization settings
CONFIG = {
    'simulation': {
        'mode': MODE_ADAPTIVE_BETA,  # Simulation mode (adaptive beta for orchestrator)
        'num_repetitions': 5,        # Number of simulation repetitions
        'num_samples': 12000,        # Total samples per repetition
        'random_seed': 42,           # Random seed for reproducibility
    },
    'image': {
        'height': 28,                # Image height (pixels)
        'width': 28,                 # Image width (pixels)
        'channels': 1,               # Number of channels (1 for grayscale)
        'noise_level': 0.1,          # Gaussian noise factor for images
    },
    'drift': {
        3000: (2, 'sudden', 0),     # Switch to uppercase at sample 3000
        7000: (0, 'gradual', 1000), # Gradual transition to numbers over 1000 samples
        10000: (1, 'sudden', 0),    # Switch to lowercase at sample 10000
    },
    'model': {
        'label_delay': 32,           # Delay before labels become available
        'fast_buffer_size': 512,     # Buffer size for fast model training
        'slow_buffer_size': 1024,    # Buffer size for slow model training
        'fast_retrain_interval': 32, # Interval for fast model retraining
        'slow_retrain_interval': 32, # Interval for slow model retraining
        'evaluation_interval': 32,   # Interval for model evaluation
        'metrics_averaging_window': 1, # Window for averaging metrics
        'fast_epochs': 1,            # Epochs for fast model training
        'slow_epochs': 10,           # Epochs for slow model training
        'fast_batch_size': 32,       # Batch size for fast model
        'slow_batch_size': 64,       # Batch size for slow model
        'fast_validation_split': 0.0, # Validation split for fast model
        'slow_validation_split': 0.0, # Validation split for slow model
        'fixed_beta': 0.5,           # Fixed beta for MODE_FIXED_BETA
        'als_params': {
            'mu_a': 0.5,             # Learning rate for adaptive beta
            'gamma': 0.9,            # Discount factor for error tracking
            'theta': 1.0,            # Scaling factor for error updates
            'a0': 0.0,               # Initial adaptation parameter
            'p0': 1e-2,              # Initial prediction error variance
            'cost_function_type': 'quadratic', # Cost function ('quadratic' or 'cross_entropy')
        },
    },
    'visualization': {
        'enabled': False,            # Enable/disable sample visualizations
        'save_plots': True,          # Save performance plots
        'plot_dir': './plots',       # Directory for saving plots
    },
}

def load_emnist_data():
    """
    Load and preprocess the EMNIST balanced dataset.

    Fetches the EMNIST balanced dataset using TensorFlow Datasets, preprocesses images by
    normalizing them to [0, 1] and adding Gaussian noise, and converts labels to float32.
    If loading fails, generates synthetic data as a fallback.

    Returns:
        tuple: (images, labels) as NumPy arrays, where images have shape
               (num_samples, height, width, channels) and labels have shape (num_samples,).
    """
    try:
        ds_train, ds_info = tfds.load(
            'emnist/balanced', split='train', shuffle_files=True, with_info=True, as_supervised=True
        )
        samples_needed = (CONFIG['simulation']['num_samples'] + 
                         CONFIG['model']['label_delay'] + 
                         max(CONFIG['model']['fast_buffer_size'], CONFIG['model']['slow_buffer_size']) + 
                         100) * CONFIG['simulation']['num_repetitions']
        
        print(f"Loading {samples_needed} samples from EMNIST...")
        
        images, labels = [], []
        for img, lbl in ds_train.shuffle(buffer_size=10000, seed=CONFIG['simulation']['random_seed']).take(samples_needed):
            img_processed, lbl_processed = preprocess_image(img, lbl)
            img_np = img_processed.numpy()
            img_noisy = add_gaussian_noise(img_np, noise_factor=CONFIG['image']['noise_level']) if CONFIG['image']['noise_level'] > 0 else img_np
            images.append(img_noisy)
            labels.append(lbl_processed)
        
        return np.array(images), np.array(labels)
    
    except Exception as e:
        print(f"Error loading EMNIST dataset: {e}")
        print("Using synthetic data for simulation.")
        samples_needed = (CONFIG['simulation']['num_samples'] + 
                         CONFIG['model']['label_delay'] + 
                         max(CONFIG['model']['fast_buffer_size'], CONFIG['model']['slow_buffer_size']) + 100)
        return (np.random.rand(samples_needed, CONFIG['image']['height'], 
                              CONFIG['image']['width'], CONFIG['image']['channels']).astype(np.float32),
                np.random.randint(0, 26, samples_needed).astype(np.int32))

def get_emnist_sample(images, labels, global_idx):
    """
    Retrieve a single sample from preloaded EMNIST data.

    Args:
        images (np.ndarray): Array of preloaded images.
        labels (np.ndarray): Array of preloaded labels.
        global_idx (int): Index of the desired sample.

    Returns:
        tuple: (image, label) if available, else (None, None).
    """
    if global_idx < len(images):
        return images[global_idx], labels[global_idx]
    return None, None

# Set random seeds for reproducibility
random.seed(CONFIG['simulation']['random_seed'])
np.random.seed(CONFIG['simulation']['random_seed'])
tf.random.set_seed(CONFIG['simulation']['random_seed'])

# Setup visualization points for sample visualization
visualization_points = setup_visualization_points(CONFIG['drift'], CONFIG['simulation']['num_samples']) if CONFIG['visualization']['enabled'] else []

# Load data
all_images, all_labels = load_emnist_data()

# Initialize logs for storing metrics across repetitions
all_logs = {
    'sample_indices': [],
    'fast_metrics': [],
    'slow_metrics': [],
    'orchestrator_metrics': [],
    'orchestrator_beta': [],
    'slow_retrain_points': [],
    'slow_retrain_losses': [],
    'slow_retrain_val_losses': [],
    'slow_retrain_val_accuracies': [],
}

# Simulation loop
for rep in range(CONFIG['simulation']['num_repetitions']):
    """
    Run a single repetition of the simulation.

    Initializes models, buffers, and logs; processes samples with concept drifts; trains
    fast and slow models; evaluates performance; and logs metrics.
    """
    print(f"\n--- Starting Repetition {rep + 1}/{CONFIG['simulation']['num_repetitions']} ---")
    
    # Initialize models and orchestrator
    fast_model = FastModel(CONFIG['image']['height'], CONFIG['image']['width'], CONFIG['image']['channels'])
    slow_model = SlowModel(CONFIG['image']['height'], CONFIG['image']['width'], CONFIG['image']['channels'])
    orchestrator = Orchestrator(
        fast_model, slow_model, **CONFIG['model']['als_params'], debug_mode=False
    )
    
    # Initialize buffers for training and evaluation
    master_buffer = deque(maxlen=max(CONFIG['model']['fast_buffer_size'], CONFIG['model']['slow_buffer_size']) + 
                                CONFIG['model']['label_delay'])
    eval_buffer_X = deque(maxlen=CONFIG['model']['evaluation_interval'])
    eval_buffer_y = deque(maxlen=CONFIG['model']['evaluation_interval'])
    
    # Initialize logs for this repetition
    rep_logs = {
        'sample_indices': [],
        'fast_metrics': {metric: [] for metric in metrics_to_report + ['loss']},
        'slow_metrics': {metric: [] for metric in metrics_to_report + ['loss']},
        'orchestrator_metrics': {metric: [] for metric in metrics_to_report + ['loss']},
        'orchestrator_beta': [],
        'slow_retrain_points': [],
        'slow_retrain_losses': [],
        'slow_retrain_val_losses': [],
        'slow_retrain_val_accuracies': [],
    }
    
    # Drift state for tracking concept changes
    current_concept = 0
    in_gradual_drift = False
    drift_start, drift_end, old_concept, new_concept = -1, -1, -1, -1
    
    # Training and evaluation state
    next_fast_retrain = 0
    next_slow_retrain = 0
    next_eval = CONFIG['model']['label_delay'] - 1
    
    # Process samples
    for sample_idx in range(CONFIG['simulation']['num_samples']):
        """
        Process a single sample in the simulation.

        Handles concept drift, assigns binary labels, updates buffers, trains models, and
        evaluates performance at specified intervals.
        """
        global_idx = rep * CONFIG['simulation']['num_samples'] + sample_idx
        if global_idx >= len(all_images):
            print(f"No more EMNIST samples available after {sample_idx} samples in Repetition {rep + 1}.")
            break
        
        # Get sample
        image, label = get_emnist_sample(all_images, all_labels, global_idx)
        if image is None:
            print(f"No more samples available at index {sample_idx} in Repetition {rep + 1}.")
            break
        
        # Handle concept drift
        if sample_idx in CONFIG['drift']:
            old_concept = current_concept
            new_concept, drift_type, duration = CONFIG['drift'][sample_idx]
            
            if drift_type == 'sudden':
                current_concept = new_concept
                in_gradual_drift = False
                print(f"\n--- Sudden Drift at Sample {sample_idx}, Rep {rep + 1}. New Concept: {['Numbers', 'Lowercase', 'Uppercase'][current_concept]} ---")
            else:
                in_gradual_drift = True
                drift_start = sample_idx
                drift_end = sample_idx + duration
                print(f"\n--- Gradual Drift at Sample {sample_idx}, Rep {rep + 1}. Transition to {['Numbers', 'Lowercase', 'Uppercase'][new_concept]} over {duration} samples ---")
        
        # Assign binary label
        if in_gradual_drift and sample_idx < drift_end:
            progress = (sample_idx - drift_start) / (drift_end - drift_start)
            binary_label = concept_functions_image[new_concept](label) if random.random() < progress else \
                          concept_functions_image[old_concept](label)
        else:
            if in_gradual_drift and sample_idx >= drift_end:
                in_gradual_drift = False
                current_concept = new_concept
                print(f"--- Gradual Drift Ended at Sample {sample_idx}. Concept: {['Numbers', 'Lowercase', 'Uppercase'][current_concept]} ---")
            binary_label = concept_functions_image[current_concept](label)
        
        # Add to master buffer
        master_buffer.append(StreamSample(image, label, binary_label, sample_idx))
        
        # Mark delayed labels
        for s in master_buffer:
            if s.sample_id == sample_idx - CONFIG['model']['label_delay'] and not s.label_available:
                s.label_available = True
                eval_buffer_X.append(s.image)
                eval_buffer_y.append(s.binary_label)
                break
        
        # Train models
        train_samples = [s for s in master_buffer if s.label_available]
        
        # Fast model training
        if sample_idx >= next_fast_retrain and len(train_samples) >= CONFIG['model']['fast_buffer_size']:
            X_fast = np.array([s.image for s in train_samples[-CONFIG['model']['fast_buffer_size']:]])
            y_fast = np.array([s.binary_label for s in train_samples[-CONFIG['model']['fast_buffer_size']:]])
            
            if len(X_fast) >= CONFIG['model']['fast_batch_size'] and CONFIG['simulation']['mode'] in \
               [MODE_FAST_ONLY, MODE_INDEPENDENT, MODE_FIXED_BETA, MODE_ADAPTIVE_BETA]:
                fast_model.train(X_fast, y_fast, epochs=CONFIG['model']['fast_epochs'],
                                batch_size=CONFIG['model']['fast_batch_size'],
                                validation_split=CONFIG['model']['fast_validation_split'])
                if CONFIG['simulation']['mode'] == MODE_ADAPTIVE_BETA:
                    orchestrator.update_weights(y_fast, X_fast)
                next_fast_retrain = sample_idx + CONFIG['model']['fast_retrain_interval']
        
        # Slow model training
        if sample_idx >= next_slow_retrain and len(train_samples) >= CONFIG['model']['slow_buffer_size']:
            X_slow = np.array([s.image for s in train_samples[-CONFIG['model']['slow_buffer_size']:]])
            y_slow = np.array([s.binary_label for s in train_samples[-CONFIG['model']['slow_buffer_size']:]])
            
            if len(X_slow) >= CONFIG['model']['slow_batch_size'] and CONFIG['simulation']['mode'] in \
               [MODE_SLOW_ONLY, MODE_INDEPENDENT, MODE_FIXED_BETA, MODE_ADAPTIVE_BETA]:
                results = slow_model.train(X_slow, y_slow, epochs=CONFIG['model']['slow_epochs'],
                                         batch_size=CONFIG['model']['slow_batch_size'],
                                         validation_split=CONFIG['model']['slow_validation_split'])
                if results['accuracy'] > 0:
                    rep_logs['slow_retrain_points'].append(sample_idx)
                    rep_logs['slow_retrain_losses'].append(results['loss'])
                    rep_logs['slow_retrain_val_losses'].append(results.get('val_loss', 0.0))
                    rep_logs['slow_retrain_val_accuracies'].append(results.get('val_accuracy', 0.0))
                next_slow_retrain = sample_idx + CONFIG['model']['slow_retrain_interval']
        
        # Evaluate models
        if sample_idx >= next_eval and len(eval_buffer_X) == CONFIG['model']['evaluation_interval']:
            X_eval = np.array(list(eval_buffer_X))
            y_eval = np.array(list(eval_buffer_y))
            
            # Visualize evaluation samples
            if CONFIG['visualization']['enabled'] and sample_idx in visualization_points:
                print(f"Visualizing evaluation samples at sample {sample_idx}, Rep {rep + 1}...")
                visualize_samples_with_labels(X_eval, y_eval, num_samples=9,
                                            title=f"Evaluation Samples (Sample {sample_idx})",
                                            is_tensor=False, current_concept_id=current_concept)
            
            # Calculate imbalance ratio
            class_counts = dict(zip(*np.unique(y_eval, return_counts=True)))
            count_0, count_1 = class_counts.get(0, 0), class_counts.get(1, 0)
            imbalance_ratio = max(count_0, count_1) / min(count_0, count_1) if min(count_0, count_1) > 0 else float('inf')
            balance_info = f"IR: {imbalance_ratio:.2f}" if min(count_0, count_1) > 0 else "N/A"
            
            # Evaluate fast model
            fast_metrics = {metric: 0.0 for metric in metrics_to_report + ['loss']}
            if fast_model.is_trained and CONFIG['simulation']['mode'] in \
               [MODE_FAST_ONLY, MODE_INDEPENDENT, MODE_FIXED_BETA, MODE_ADAPTIVE_BETA]:
                fast_metrics = fast_model.evaluate(X_eval, y_eval)
            
            # Evaluate slow model
            slow_metrics = {metric: 0.0 for metric in metrics_to_report + ['loss']}
            if slow_model.is_trained and CONFIG['simulation']['mode'] in \
               [MODE_SLOW_ONLY, MODE_INDEPENDENT, MODE_FIXED_BETA, MODE_ADAPTIVE_BETA]:
                slow_metrics = slow_model.evaluate(X_eval, y_eval)
            
            # Evaluate orchestrator
            orch_metrics = {metric: 0.0 for metric in metrics_to_report + ['loss']}
            orch_beta = 0.0
            if CONFIG['simulation']['mode'] == MODE_FAST_ONLY:
                orch_metrics, orch_beta = fast_metrics, 1.0
            elif CONFIG['simulation']['mode'] == MODE_SLOW_ONLY:
                orch_metrics, orch_beta = slow_metrics, 0.0
            elif CONFIG['simulation']['mode'] in [MODE_FIXED_BETA, MODE_ADAPTIVE_BETA] and \
                 fast_model.is_trained and slow_model.is_trained:
                orch_metrics = orchestrator.evaluate(X_eval, y_eval, mode=CONFIG['simulation']['mode'],
                                                   fixed_beta=CONFIG['model']['fixed_beta'])
                orch_beta = orchestrator.get_beta() if CONFIG['simulation']['mode'] == MODE_ADAPTIVE_BETA else \
                            CONFIG['model']['fixed_beta']
            
            # Log metrics
            for metric in metrics_to_report + ['loss']:
                rep_logs['fast_metrics'][metric].append(fast_metrics[metric])
                rep_logs['slow_metrics'][metric].append(slow_metrics[metric])
                rep_logs['orchestrator_metrics'][metric].append(orch_metrics[metric])
            rep_logs['orchestrator_beta'].append(orch_beta)
            rep_logs['sample_indices'].append(sample_idx)
            
            # Print averaged metrics
            if len(rep_logs['fast_metrics']['accuracy']) >= CONFIG['model']['metrics_averaging_window']:
                avg_metrics = {
                    metric: np.mean(rep_logs['fast_metrics'][metric][-CONFIG['model']['metrics_averaging_window']:])
                    for metric in metrics_to_report + ['loss']
                }
                avg_metrics.update({
                    f'slow_{metric}': np.mean(rep_logs['slow_metrics'][metric][-CONFIG['model']['metrics_averaging_window']:])
                    for metric in metrics_to_report + ['loss']
                })
                avg_metrics.update({
                    f'orch_{metric}': np.mean(rep_logs['orchestrator_metrics'][metric][-CONFIG['model']['metrics_averaging_window']:])
                    for metric in metrics_to_report + ['loss']
                })
                avg_metrics['orch_beta'] = np.mean(rep_logs['orchestrator_beta'][-CONFIG['model']['metrics_averaging_window']:])
                print(f"Rep {rep + 1}/{CONFIG['simulation']['num_repetitions']}. Sample {sample_idx} "
                      f"(Avg over {CONFIG['model']['metrics_averaging_window']} blocks): "
                      f"Fast Acc={avg_metrics['accuracy']:.4f}, Fast Loss={avg_metrics['loss']:.4f}, "
                      f"Slow Acc={avg_metrics['slow_accuracy']:.4f}, Slow Loss={avg_metrics['slow_loss']:.4f}, "
                      f"Orch Acc={avg_metrics['orch_accuracy']:.4f}, Orch Loss={avg_metrics['orch_loss']:.4f}, "
                      f"Beta={avg_metrics['orch_beta']:.4f}, {balance_info}")
            
            eval_buffer_X.clear()
            eval_buffer_y.clear()
            next_eval = sample_idx + CONFIG['model']['evaluation_interval']
    
    # Save repetition logs
    for key, value in rep_logs.items():
        all_logs[key].append(value)
    
    # Debug: Print log lengths for this repetition
    print(f"Rep {rep + 1} log lengths: sample_indices={len(rep_logs['sample_indices'])}, "
          f"fast_metrics={len(rep_logs['fast_metrics']['accuracy'])}, "
          f"slow_metrics={len(rep_logs['slow_metrics']['accuracy'])}, "
          f"orchestrator_metrics={len(rep_logs['orchestrator_metrics']['accuracy'])}, "
          f"orchestrator_beta={len(rep_logs['orchestrator_beta'])}")
    
    gc.collect()

# Average results across repetitions
print("\n--- Calculating Averages Across Repetitions ---")
max_log_len = min(len(log) for log in all_logs['sample_indices']) if all_logs['sample_indices'] else 0
print(f"Max log length for averaging: {max_log_len}")
averaged_logs = {
    'sample_indices': all_logs['sample_indices'][0][:max_log_len] if all_logs['sample_indices'] else [],
    'fast_metrics': {metric: [] for metric in metrics_to_report + ['loss']},
    'slow_metrics': {metric: [] for metric in metrics_to_report + ['loss']},
    'orchestrator_metrics': {metric: [] for metric in metrics_to_report + ['loss']},
    'orchestrator_beta': [],
}

for i in range(max_log_len):
    for metric in metrics_to_report + ['loss']:
        fast_values = [log[metric][i] for log in all_logs['fast_metrics'] if i < len(log[metric])]
        slow_values = [log[metric][i] for log in all_logs['slow_metrics'] if i < len(log[metric])]
        orch_values = [log[metric][i] for log in all_logs['orchestrator_metrics'] if i < len(log[metric])]
        averaged_logs['fast_metrics'][metric].append(np.mean(fast_values) if fast_values else 0.0)
        averaged_logs['slow_metrics'][metric].append(np.mean(slow_values) if slow_values else 0.0)
        averaged_logs['orchestrator_metrics'][metric].append(np.mean(orch_values) if orch_values else 0.0)
    beta_values = [log[i] for log in all_logs['orchestrator_beta'] if i < len(log)]
    averaged_logs['orchestrator_beta'].append(np.mean(beta_values) if beta_values else 0.0)

# Debug: Print averaged logs
print(f"Averaged logs lengths: sample_indices={len(averaged_logs['sample_indices'])}, "
      f"fast_metrics={len(averaged_logs['fast_metrics']['accuracy'])}, "
      f"slow_metrics={len(averaged_logs['slow_metrics']['accuracy'])}, "
      f"orchestrator_metrics={len(averaged_logs['orchestrator_metrics']['accuracy'])}, "
      f"orchestrator_beta={len(averaged_logs['orchestrator_beta'])}")
print(f"Sample indices: {averaged_logs['sample_indices'][:10]}...")
print(f"Fast accuracy (first 10): {averaged_logs['fast_metrics']['accuracy'][:10]}...")
print(f"Slow accuracy (first 10): {averaged_logs['slow_metrics']['accuracy'][:10]}...")
print(f"Orchestrator accuracy (first 10): {averaged_logs['orchestrator_metrics']['accuracy'][:10]}...")
print(f"Orchestrator beta (first 10): {averaged_logs['orchestrator_beta'][:10]}...")

# Create visualizations
create_chart(
    averaged_logs['sample_indices'],
    averaged_logs['fast_metrics'],
    averaged_logs['slow_metrics'],
    averaged_logs['orchestrator_metrics'],
    CONFIG['drift'],
    CONFIG['simulation']['num_samples'],
    CONFIG['visualization']['plot_dir']
)

plot_results(
    averaged_logs['sample_indices'],
    averaged_logs['fast_metrics'],
    averaged_logs['slow_metrics'],
    averaged_logs['orchestrator_metrics'],
    averaged_logs['orchestrator_beta'],
    CONFIG['simulation']['mode'],
    CONFIG['drift'],
    CONFIG['simulation']['num_repetitions'],
    CONFIG['visualization']['plot_dir']
)

# Print final results
print(f"\nSimulation completed over {CONFIG['simulation']['num_repetitions']} repetitions.")
print(f"{CONFIG['model']['als_params']['cost_function_type']}. mu={CONFIG['model']['als_params']['mu_a']}, "
      f"gamma={CONFIG['model']['als_params']['gamma']}, theta={CONFIG['model']['als_params']['theta']}")
print("Final averages:")
print(f"  Fast Model Accuracy: {averaged_logs['fast_metrics']['accuracy'][-1] if averaged_logs['fast_metrics']['accuracy'] else 0.0:.4f}")
print(f"  Slow Model Accuracy: {averaged_logs['slow_metrics']['accuracy'][-1] if averaged_logs['slow_metrics']['accuracy'] else 0.0:.4f}")
print(f"  Orchestrator Accuracy: {averaged_logs['orchestrator_metrics']['accuracy'][-1] if averaged_logs['orchestrator_metrics']['accuracy'] else 0.0:.4f}")
print(f"  Orchestrator Loss: {averaged_logs['orchestrator_metrics']['loss'][-1] if averaged_logs['orchestrator_metrics']['loss'] else 0.0:.4f}")
print(f"  Final Beta: {averaged_logs['orchestrator_beta'][-1] if averaged_logs['orchestrator_beta'] else 0.0:.4f} (Applicable in ADAPTIVE_BETA mode)")