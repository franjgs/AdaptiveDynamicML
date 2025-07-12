#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities for the AdaptiveDynamicML project.

This module provides functions to generate visualization points and create plots for model performance
metrics (accuracy, loss, and beta) in the context of concept drift simulations. It supports both
Matplotlib for static PNG plots and Chart.js for interactive web-based visualizations.

Created on Wed Jul 2 10:37:01 2025
Author: Fran
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import uuid
import json

from src.models import MODE_FAST_ONLY, MODE_SLOW_ONLY, MODE_INDEPENDENT, MODE_FIXED_BETA, MODE_ADAPTIVE_BETA

def setup_visualization_points(drift_points, num_samples):
    """Determine sample indices for visualizing data during simulation.

    Identifies key points for visualization, including an initial point, midpoints between concept
    drifts, and a point after the final drift, to highlight significant moments in the data stream.

    Args:
        drift_points (dict): Dictionary with sample indices as keys and tuples of
                            (concept_idx, drift_type, duration) as values.
        num_samples (int): Total number of samples in the simulation.

    Returns:
        list: Sorted list of sample indices for visualization.
    """
    vis_points = {100}  # Initial visualization point
    sorted_drift_points = sorted(drift_points.keys())

    for i in range(len(sorted_drift_points) - 1):
        p1, p2 = sorted_drift_points[i], sorted_drift_points[i + 1]
        vis_points.add(p1 + (p2 - p1) // 2)

    if sorted_drift_points:
        vis_points.add(sorted_drift_points[-1] + 1000)

    return sorted(list(vis_points))

def create_chart(sample_indices, fast_metrics, slow_metrics, orchestrator_metrics, drift_points, num_samples, plot_dir):
    """Generate a Chart.js configuration for visualizing model accuracy metrics.

    Creates a JSON configuration for a Chart.js line chart to plot the accuracy of the fast model,
    slow model, and orchestrator over sample indices. Includes annotations for sudden and gradual
    concept drifts. The configuration is saved to a JSON file and printed for debugging.

    Args:
        sample_indices (list): List of sample indices where metrics were recorded.
        fast_metrics (dict): Dictionary with 'accuracy' key and list of metric values for the fast model.
        slow_metrics (dict): Dictionary with 'accuracy' key and list of metric values for the slow model.
        orchestrator_metrics (dict): Dictionary with 'accuracy' key and list of metric values for the orchestrator.
        drift_points (dict): Dictionary with sample indices as keys and tuples of
                            (concept_idx, drift_type, duration) as values.
        num_samples (int): Total number of samples in the simulation.
        plot_dir (str): Directory path to save the Chart.js configuration.

    Returns:
        dict: Chart.js configuration dictionary.
    """
    labels = [str(i) for i in sample_indices]
    datasets = [
        {
            'label': 'Fast Model Accuracy',
            'data': fast_metrics['accuracy'],
            'borderColor': '#1f77b4',
            'backgroundColor': '#1f77b4',
            'fill': False,
            'tension': 0.1
        },
        {
            'label': 'Slow Model Accuracy',
            'data': slow_metrics['accuracy'],
            'borderColor': '#2ca02c',
            'backgroundColor': '#2ca02c',
            'fill': False,
            'tension': 0.1
        },
        {
            'label': 'Orchestrator Accuracy',
            'data': orchestrator_metrics['accuracy'],
            'borderColor': '#d62728',
            'backgroundColor': '#d62728',
            'fill': False,
            'tension': 0.1
        }
    ]

    annotations = []
    for point, (concept_idx, drift_type, duration) in drift_points.items():
        if point > num_samples:
            continue
        idx = next((i for i, x in enumerate(sample_indices) if x >= point), len(sample_indices) - 1)
        if drift_type == 'sudden':
            annotations.append({
                'type': 'line',
                'xMin': idx,
                'xMax': idx,
                'borderColor': 'red',
                'borderWidth': 2,
                'borderDash': [5, 5],
                'label': {
                    'content': f'Sudden Drift {point}',
                    'enabled': True,
                    'position': 'top'
                }
            })
        else:
            end_point = point + duration
            end_idx = next((i for i, x in enumerate(sample_indices) if x >= end_point), len(sample_indices) - 1)
            annotations.append({
                'type': 'box',
                'xMin': idx,
                'xMax': end_idx,
                'backgroundColor': 'rgba(255, 165, 0, 0.2)',
                'borderColor': 'orange',
                'borderWidth': 1,
                'label': {
                    'content': f'Gradual Drift {point}-{end_point}',
                    'enabled': True,
                    'position': 'top'
                }
            })

    chart_config = {
        'type': 'line',
        'data': {
            'labels': labels,
            'datasets': datasets
        },
        'options': {
            'responsive': True,
            'plugins': {
                'title': {
                    'display': True,
                    'text': 'Model Accuracy Evolution (Avg over Runs)'
                },
                'legend': {
                    'position': 'top'
                },
                'annotation': {
                    'annotations': annotations
                }
            },
            'scales': {
                'x': {
                    'title': {
                        'display': True,
                        'text': 'Samples Processed'
                    }
                },
                'y': {
                    'title': {
                        'display': True,
                        'text': 'Accuracy'
                    },
                    'min': 0,
                    'max': 1.05
                }
            }
        }
    }

    os.makedirs(plot_dir, exist_ok=True)
    config_path = os.path.join(plot_dir, f'chart_config_{uuid.uuid4()}.json')
    with open(config_path, 'w') as f:
        json.dump(chart_config, f, indent=4)
    print(f"Chart.js configuration saved to: {config_path}")
    print(f"Chart.js configuration:\n{json.dumps(chart_config, indent=4)}")

    return chart_config

def _add_drift_annotations(plt, sample_indices, drift_points, y_min, y_max, is_beta_plot=False):
    """Helper function to add drift annotations to Matplotlib subplots.

    Adds vertical lines for sudden drifts and shaded regions for gradual drifts, with text labels,
    to visualize concept drift points in accuracy, loss, or beta subplots.

    Args:
        plt: Matplotlib.pyplot instance for plotting.
        sample_indices (list): List of sample indices.
        drift_points (dict): Dictionary with sample indices as keys and tuples of
                            (concept_idx, drift_type, duration) as values.
        y_min (float): Minimum y-axis value for text positioning.
        y_max (float): Maximum y-axis value for text positioning.
        is_beta_plot (bool, optional): If True, adjusts text positioning for beta subplot.
    """
    drift_label_added = False
    max_index = max(sample_indices, default=1)

    for point, (concept_idx, drift_type, duration) in drift_points.items():
        if point > max_index:
            continue
        if drift_type == 'sudden':
            label = f'Drift {point} (Sudden)' if not drift_label_added else None
            plt.axvline(x=point, color='red', linestyle='--', label=label, alpha=0.7)
            text_y = y_min + 0.01 if is_beta_plot else y_min + 0.1 * (y_max - y_min)
            plt.text(point + 0.005 * max_index, text_y, f'Sudden Drift {point}',
                     rotation=90, va='bottom', ha='left', color='red', fontsize=9)
        else:
            end_point = point + duration
            label = f'Drift {point}-{end_point} (Gradual)' if not drift_label_added else None
            plt.axvspan(point, end_point, color='orange', alpha=0.2, label=label)
            plt.axvline(x=point, color='orange', linestyle=':', alpha=0.7)
            plt.axvline(x=end_point, color='orange', linestyle=':', alpha=0.7)
            text_y = y_min + 0.01 if is_beta_plot else y_min + 0.1 * (y_max - y_min)
            plt.text(point + 0.005 * max_index, text_y, f'Gradual Start {point}',
                     rotation=90, va='bottom', ha='left', color='orange', fontsize=9)
            plt.text(end_point + 0.005 * max_index, text_y, f'Gradual End {end_point}',
                     rotation=90, va='bottom', ha='left', color='orange', fontsize=9)
        drift_label_added = True

def plot_results(sample_indices, fast_metrics, slow_metrics, orchestrator_metrics, beta_values, simulation_mode, drift_points, num_repetitions, cost_function, plot_dir):
    """Create and save a Matplotlib plot with subplots for accuracy, loss, and beta values.

    Generates a figure with three subplots to visualize model performance (accuracy and loss) and
    the orchestrator’s beta parameter, averaged across multiple runs. Includes annotations for
    concept drifts to highlight their impact. Saves the plot as a PNG file.

    Args:
        sample_indices (list): List of sample indices where metrics were recorded.
        fast_metrics (dict): Dictionary with 'accuracy' and 'loss' keys and lists of metric values.
        slow_metrics (dict): Dictionary with 'accuracy' and 'loss' keys and lists of metric values.
        orchestrator_metrics (dict): Dictionary with 'accuracy' and 'loss' keys and lists of metric values.
        beta_values (list): List of beta values for the orchestrator.
        simulation_mode (str): Simulation mode (e.g., MODE_ADAPTIVE_BETA, MODE_FAST_ONLY).
        drift_points (dict): Dictionary with sample indices as keys and tuples of
                            (concept_idx, drift_type, duration) as values.
        num_repetitions (int): Number of simulation runs for averaging metrics.
        cost_function (str):  Type of cost function ('quadratic' or 'cross_entropy').
        plot_dir (str): Directory path to save the plot.

    Returns:
        None: Saves the plot as a PNG file in the specified directory.
    """
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))

    # Accuracy subplot
    plt.subplot(3, 1, 1)
    plt.plot(sample_indices, fast_metrics['accuracy'], label='Fast Model (Avg)', color='blue')
    plt.plot(sample_indices, slow_metrics['accuracy'], label='Slow Model (Avg)', color='green')

    if simulation_mode in [MODE_FIXED_BETA, MODE_ADAPTIVE_BETA]:
        plt.plot(sample_indices, orchestrator_metrics['accuracy'], label='Orchestrator (Avg)', color='red')
    elif simulation_mode == MODE_FAST_ONLY:
        plt.plot(sample_indices, orchestrator_metrics['accuracy'], label='Orchestrator (Fast - Avg)', color='red', linestyle=':')
    elif simulation_mode == MODE_SLOW_ONLY:
        plt.plot(sample_indices, orchestrator_metrics['accuracy'], label='Orchestrator (Slow - Avg)', color='red', linestyle=':')

    y_min, y_max = plt.ylim()
    _add_drift_annotations(plt, sample_indices, drift_points, y_min, y_max)
    plt.title(f'Model Accuracy Evolution (Avg over {num_repetitions} Runs)')
    plt.xlabel('Samples Processed')
    plt.ylabel('Accuracy')
    plt.ylim(0.0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(True)

    # Loss subplot
    plt.subplot(3, 1, 2)
    plt.plot(sample_indices, fast_metrics['loss'], label='Fast Model Loss (Avg)', color='blue', linestyle='--')
    plt.plot(sample_indices, slow_metrics['loss'], label='Slow Model Loss (Avg)', color='green', linestyle='--')
    if simulation_mode in [MODE_FIXED_BETA, MODE_ADAPTIVE_BETA]:
        plt.plot(sample_indices, orchestrator_metrics['loss'], label='Orchestrator Loss (Avg)', color='red', linestyle='--')
    elif simulation_mode == MODE_FAST_ONLY:
        plt.plot(sample_indices, orchestrator_metrics['loss'], label='Orchestrator Loss (Fast - Avg)', color='red', linestyle=':')
    elif simulation_mode == MODE_SLOW_ONLY:
        plt.plot(sample_indices, orchestrator_metrics['loss'], label='Orchestrator Loss (Slow - Avg)', color='red', linestyle=':')

    y_min, y_max = plt.ylim()
    _add_drift_annotations(plt, sample_indices, drift_points, y_min, y_max)
    plt.title(f'Model Loss Evolution (Avg over {num_repetitions} Runs)')
    plt.xlabel('Samples Processed')
    plt.ylabel('Loss (Binary Cross-Entropy)')
    plt.ylim(-0.05, 2.0)
    plt.legend(loc='upper right')
    plt.grid(True)

    # Beta subplot
    plt.subplot(3, 1, 3)
    if simulation_mode == MODE_ADAPTIVE_BETA:
        plt.plot(sample_indices, beta_values, label='Orchestrator Beta (Avg)', color='purple')
    elif simulation_mode == MODE_FIXED_BETA:
        plt.axhline(y=beta_values[0], color='purple', linestyle='-', label=f'Fixed Beta = {beta_values[0]}')
    elif simulation_mode == MODE_FAST_ONLY:
        plt.axhline(y=1.0, color='purple', linestyle=':', label='Beta (Fast Only) = 1.0')
    elif simulation_mode == MODE_SLOW_ONLY:
        plt.axhline(y=0.0, color='purple', linestyle=':', label='Beta (Slow Only) = 0.0')
    else:
        plt.plot([], [], label='Beta not applicable in independent mode', color='purple')

    y_min, y_max = plt.ylim()
    _add_drift_annotations(plt, sample_indices, drift_points, y_min, y_max, is_beta_plot=True)
    plt.title(f'Orchestrator Beta Evolution ({cost_function}) (Avg over {num_repetitions} Runs)')
    plt.xlabel('Samples Processed')
    plt.ylabel('Beta (Fast Model Weight)')
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f'results_{uuid.uuid4()}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")