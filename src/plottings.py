import matplotlib.pyplot as plt
import numpy as np
import os
import uuid

from src.models import MODE_FAST_ONLY, MODE_SLOW_ONLY, MODE_INDEPENDENT, MODE_FIXED_BETA, MODE_ADAPTIVE_BETA

def setup_visualization_points(drift_points, num_samples):
    """
    Determine specific sample indices for visualizing data during the simulation.

    This function identifies key points in the simulation where data samples should be visualized,
    such as an initial point, midpoints between concept drifts, and a point after the final drift.
    It helps focus visualization on significant moments in the data stream, particularly around
    concept drifts defined in the configuration.

    Args:
        drift_points (dict): Dictionary with sample indices as keys and tuples of
                            (concept_idx, drift_type, duration) as values, indicating where
                            concept drifts occur.
        num_samples (int): Total number of samples in the simulation.

    Returns:
        list: Sorted list of sample indices where visualizations should occur.
    """
    vis_points = set([100])  # Initial point
    drift_points = sorted(drift_points.keys())
    
    for i in range(len(drift_points) - 1):
        p1, p2 = drift_points[i], drift_points[i + 1]
        vis_points.add(p1 + (p2 - p1) // 2)
    
    if drift_points:
        vis_points.add(drift_points[-1] + 1000)
    
    return sorted(list(vis_points))

def create_chart(sample_indices, fast_metrics, slow_metrics, orchestrator_metrics, drift_points, num_samples):
    """
    Generate a Chart.js configuration for a line chart visualizing model accuracy metrics.

    This function creates a JSON-like configuration for a Chart.js line chart to plot the accuracy
    of the fast model, slow model, and orchestrator over time (sample indices). It includes
    annotations to mark sudden and gradual concept drifts, making it easier to observe how model
    performance changes around drift points. The chart is printed as a JSON string for external use.

    Args:
        sample_indices (list): List of sample indices where metrics were recorded.
        fast_metrics (dict): Dictionary with metric names (e.g., 'accuracy') as keys and lists of
                            metric values for the fast model.
        slow_metrics (dict): Dictionary with metric names and lists of metric values for the slow model.
        orchestrator_metrics (dict): Dictionary with metric names and lists of metric values for the orchestrator.
        drift_points (dict): Dictionary with sample indices as keys and tuples of
                            (concept_idx, drift_type, duration) as values.
        num_samples (int): Total number of samples in the simulation.

    Returns:
        None: Prints the Chart.js configuration as a JSON-like string.
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

    # Add vertical lines for drift points
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
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": datasets
        },
        "options": {
            "responsive": True,
            "plugins": {
                "title": {
                    "display": True,
                    "text": "Model Accuracy Evolution (Avg over Runs)"
                },
                "legend": {
                    "position": "top"
                },
                "annotation": {
                    "annotations": annotations
                }
            },
            "scales": {
                "x": {
                    "title": {
                        "display": True,
                        "text": "Samples Processed"
                    }
                },
                "y": {
                    "title": {
                        "display": True,
                        "text": "Accuracy"
                    },
                    "min": 0,
                    "max": 1.05
                }
            }
        }
    }

    # Output Chart.js configuration as JSON-like string
    print(f"Chart.js configuration:\n{chart_config}")

def plot_results(sample_indices, fast_metrics, slow_metrics, orchestrator_metrics, beta_values, simulation_mode, drift_points, num_repetitions, plot_dir):
    """
    Create and save a Matplotlib plot with subplots for accuracy, loss, and beta values.

    This function generates a three-subplot figure to visualize the evolution of model performance
    (accuracy and loss) and the orchestrator’s beta parameter (if applicable) over sample indices,
    averaged across multiple runs. It includes annotations for sudden and gradual concept drifts
    to highlight their impact on performance. The plot is saved as a PNG file with a unique filename.

    Args:
        sample_indices (list): List of sample indices where metrics were recorded.
        fast_metrics (dict): Dictionary with metric names (e.g., 'accuracy', 'loss') and lists of
                            metric values for the fast model.
        slow_metrics (dict): Dictionary with metric names and lists of metric values for the slow model.
        orchestrator_metrics (dict): Dictionary with metric names and lists of metric values for the orchestrator.
        beta_values (list): List of beta values (weights for fast model in orchestrator) over sample indices.
        simulation_mode (str): Simulation mode (e.g., MODE_ADAPTIVE_BETA, MODE_FAST_ONLY) determining
                              how the orchestrator’s metrics and beta are plotted.
        drift_points (dict): Dictionary with sample indices as keys and tuples of
                            (concept_idx, drift_type, duration) as values.
        num_repetitions (int): Number of simulation runs for averaging metrics.
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
    
    drift_label_added = False
    for point, (concept_idx, drift_type, duration) in drift_points.items():
        if point > max(sample_indices, default=0):
            continue
        if drift_type == 'sudden':
            label = f'Drift {point} (Sudden)' if not drift_label_added else None
            plt.axvline(x=point, color='red', linestyle='--', label=label, alpha=0.7)
            y_min, y_max = plt.ylim()
            plt.text(point + 0.005 * max(sample_indices, default=1), y_min + 0.1*(y_max-y_min), 
                     f'Sudden Drift {point}', rotation=90, va='bottom', ha='left', color='red', fontsize=9)
        else:
            end_point = point + duration
            label = f'Drift {point}-{end_point} (Gradual)' if not drift_label_added else None
            plt.axvspan(point, end_point, color='orange', alpha=0.2, label=label)
            plt.axvline(x=point, color='orange', linestyle=':', alpha=0.7)
            plt.axvline(x=end_point, color='orange', linestyle=':', alpha=0.7)
            y_min, y_max = plt.ylim()
            plt.text(point + 0.005 * max(sample_indices, default=1), y_min + 0.1*(y_max-y_min), 
                     f'Gradual Start {point}', rotation=90, va='bottom', ha='left', color='orange', fontsize=9)
            plt.text(end_point + 0.005 * max(sample_indices, default=1), y_min + 0.1*(y_max-y_min), 
                     f'Gradual End {end_point}', rotation=90, va='bottom', ha='left', color='orange', fontsize=9)
        drift_label_added = True
    
    plt.title(f'Model Accuracy Evolution (Avg over {num_repetitions} Runs)')
    plt.xlabel('Samples Processed')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1.05])
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
    
    drift_label_added = False
    for point, (concept_idx, drift_type, duration) in drift_points.items():
        if point > max(sample_indices, default=0):
            continue
        if drift_type == 'sudden':
            label = f'Drift {point} (Sudden)' if not drift_label_added else None
            plt.axvline(x=point, color='red', linestyle='--', label=label, alpha=0.7)
            y_min, y_max = plt.ylim()
            plt.text(point + 0.005 * max(sample_indices, default=1), y_min + 0.1*(y_max-y_min), 
                     f'Sudden Drift {point}', rotation=90, va='bottom', ha='left', color='red', fontsize=9)
        else:
            end_point = point + duration
            label = f'Drift {point}-{end_point} (Gradual)' if not drift_label_added else None
            plt.axvspan(point, end_point, color='orange', alpha=0.2, label=label)
            plt.axvline(x=point, color='orange', linestyle=':', alpha=0.7)
            plt.axvline(x=end_point, color='orange', linestyle=':', alpha=0.7)
            y_min, y_max = plt.ylim()
            plt.text(point + 0.005 * max(sample_indices, default=1), y_min + 0.1*(y_max-y_min), 
                     f'Gradual Start {point}', rotation=90, va='bottom', ha='left', color='orange', fontsize=9)
            plt.text(end_point + 0.005 * max(sample_indices, default=1), y_min + 0.1*(y_max-y_min), 
                     f'Gradual End {end_point}', rotation=90, va='bottom', ha='left', color='orange', fontsize=9)
        drift_label_added = True
    
    plt.title(f'Model Loss Evolution (Avg over {num_repetitions} Runs)')
    plt.xlabel('Samples Processed')
    plt.ylabel('Loss (Binary Cross-Entropy)')
    plt.ylim([-0.05, 2.0])
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
    
    drift_label_added = False
    for point, (concept_idx, drift_type, duration) in drift_points.items():
        if point > max(sample_indices, default=0):
            continue
        if drift_type == 'sudden':
            label = f'Drift {point} (Sudden)' if not drift_label_added else None
            plt.axvline(x=point, color='red', linestyle='--', label=label, alpha=0.7)
            y_min, y_max = plt.ylim()
            plt.text(point + 0.005 * max(sample_indices, default=1), y_min + 0.01, 
                     f'Sudden Drift {point}', rotation=90, va='bottom', ha='left', color='red', fontsize=9)
        else:
            end_point = point + duration
            label = f'Drift {point}-{end_point} (Gradual)' if not drift_label_added else None
            plt.axvspan(point, end_point, color='orange', alpha=0.2, label=label)
            plt.axvline(x=point, color='orange', linestyle=':', alpha=0.7)
            plt.axvline(x=end_point, color='orange', linestyle=':', alpha=0.7)
            y_min, y_max = plt.ylim()
            plt.text(point + 0.005 * max(sample_indices, default=1), y_min + 0.01, 
                     f'Gradual Start {point}', rotation=90, va='bottom', ha='left', color='orange', fontsize=9)
            plt.text(end_point + 0.005 * max(sample_indices, default=1), y_min + 0.01, 
                     f'Gradual End {end_point}', rotation=90, va='bottom', ha='left', color='orange', fontsize=9)
        drift_label_added = True
    
    plt.title(f'Orchestrator Beta Evolution (Avg over {num_repetitions} Runs)')
    plt.xlabel('Samples Processed')
    plt.ylabel('Beta (Fast Model Weight)')
    plt.ylim([-0.05, 1.05])
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/results_{uuid.uuid4()}.png")
    plt.close()