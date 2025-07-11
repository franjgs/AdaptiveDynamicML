# AdaptiveDynamicML

Research and development of machine learning models that dynamically adapt to evolving data streams and changing underlying concepts.

## Project Summary

This repository contains the source code for a research line focused on the study and development of dynamic and adaptive Machine Learning models. The primary objective is to explore and evaluate how combining learning models with different adaptation speeds, coordinated by an intelligent orchestrator, can improve robustness and performance in environments where data distributions and/or underlying concepts change over time (concept drift).

The current project, implemented within this repository, focuses on simulating concept drift using the EMNIST dataset. It explores different types of drift (sudden, gradual, cyclic) through changes in target classes (e.g., numbers to uppercase/lowercase letters) and the introduction of noise or alterations in image features.

This work draws inspiration from previous research on adaptive ensemble learning frameworks with varying levels of diversity, as explored in the following paper:

## Key Features

-   **Fast and Slow Models (`src/models.py`):** Implementation of two image classification models (based on TensorFlow/Keras) with different learning and adaptation capabilities.
    -   *Fast Model:* Designed for rapid adaptation to recent changes in data distribution.
    -   *Slow Model:* A more complex and deep model, with greater long-term learning capacity and stability.
-   **Adaptive Orchestrator (`src/models.py`):** An algorithm that dynamically manages and combines the predictions from the Fast and Slow models, adjusting their weights (`beta`) based on observed performance and drift signals.
-   **Concept Drift Simulation (`AHDL_EMNIST.py` and `src/utils.py`):** Mechanisms to introduce and manage concept drift in the EMNIST dataset data stream. This includes concept functions to classify EMNIST images in different ways over time.
-   **Data Handling and Preprocessing (`AHDL_EMNIST.py` and `src/utils.py`):** Functions for loading, preprocessing, and serving the EMNIST dataset as a continuous data stream.
-   **Results Visualization (`src/plottings.py`):** Tools to generate plots showing the evolution of performance metrics (accuracy, loss) and the orchestrator's behavior (`beta` parameter) throughout the simulation.
-   **Flexible Configuration:** A `CONFIG` dictionary (in `AHDL_EMNIST.py`) allows for easy adjustment of simulation, model, and visualization parameters.

## Repository Structure
Aquí tienes la sección de "Repository Structure" pulida para una mejor visualización y con la corrección del nombre del archivo `__init__.py`:

```markdown
## Repository Structure

```

AdaptiveDynamicML/
├── AHDL\_EMNIST.py           \# Main script for running simulations.
├── src/
│   ├── **init**.py          \# Makes 'src' a Python package, facilitating imports.
│   ├── models.py            \# Definitions of FastModel, SlowModel, Orchestrator, and operation modes.
│   ├── plottings.py         \# Functions for visualizing simulation results.
│   └── utils.py             \# Utility functions for data handling, concepts, and noise.
├── data/                    \# Directory to store datasets (raw/processed).
├── results/                 \# Directory for storing simulation logs.
│   └── plots/               \# Subdirectory for generated plots.
├── config/                  \# (Optional) For external configuration files.
├── notebooks/               \# (Optional) For Jupyter notebooks for exploration or analysis.
├── .gitignore               \# Defines which files and directories Git should ignore.
├── README.md                \# This file.
└── requirements.txt         \# List of Python dependencies.

```
```

## Setup and Execution

### System Requirements

-   Python 3.x
-   A virtual environment (recommended: `conda` or `venv`).
-   GPU acceleration hardware (e.g., Apple Silicon M1/M2/M3 with `tensorflow-metal` for optimal TensorFlow performance) is recommended.

### Installing Dependencies

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/franjgs/AdaptiveDynamicML.git](https://github.com/franjgs/AdaptiveDynamicML.git)
    cd AdaptiveDynamicML
    ```
2.  **Create and Activate Virtual Environment (e.g., with Conda):**
    ```bash
    conda create -n adml_env python=3.9 # Adjust Python version if needed
    conda activate adml_env
    ```
3.  **Install packages:**
    Ensure your environment is active and run:
    ```bash
    pip install -r requirements.txt
    ```

### Running Simulations

To run a simulation, execute the main script `AHDL_EMNIST.py`:

```bash
python AHDL_EMNIST.py
