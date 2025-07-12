# AdaptiveDynamicML

Research and development of machine learning models that dynamically adapt to evolving data streams and concept drift.

## Project Summary

Inspired by the human brain’s dual learning mechanisms, this project implements a machine learning framework that combines fast and slow learning strategies to handle concept drift—changes in data distribution over time. Similar to how the sympathetic nervous system enables rapid reactions and the central nervous system builds long-term knowledge, our system uses:
- A **fast model** that quickly adapts to new patterns in the data.
- A **slow model** that learns stable, deep knowledge over time.
- An **orchestrator** that intelligently combines predictions from both models, adjusting their contributions based on performance and detected drifts.

This approach is based on the paper: **"Generalized CMAC Adaptive Ensembles for Concept-Drifting Data Streams" by Francisco J. González-Serrano and Aníbal R. Figueiras-Vidal, presented at EUSIPCO 2017 in Kos, Greece.**

The current implementation uses the **EMNIST dataset** to demonstrate the framework’s effectiveness in a controlled environment. We simulate concept drifts (sudden and gradual) using binary classification tasks (e.g., digits as < 5 vs. ≥ 5, lowercase letters by parity, uppercase letters by threshold) to evaluate the adaptive system’s performance. The framework is designed to be flexible, applicable to various real-world data streams and tasks beyond EMNIST.

## Key Features

- **Fast and Slow Models (`src/models.py`)**: Two TensorFlow/Keras-based image classification models:
  - **Fast Model**: Rapidly adapts to new data patterns with a lightweight architecture.
  - **Slow Model**: Builds stable, long-term knowledge with a deeper convolutional network.
- **Adaptive Orchestrator (`src/models.py`)**: Dynamically weights predictions from fast and slow models based on their performance and detected concept drifts.
- **Concept Drift Simulation (`AHDL_EMNIST.py`, `src/utils.py`)**: Tools to simulate sudden and gradual drifts in the EMNIST dataset, using binary classification tasks defined by concept functions.
- **Data Handling and Preprocessing (`AHDL_EMNIST.py`, `src/utils.py`)**: Functions to load, preprocess, and stream EMNIST data for continuous learning.
- **Results Visualization (`src/plottings.py`)**: Generates Matplotlib plots and Chart.js configurations (saved as JSON files) to visualize accuracy, loss, and orchestrator beta values over time.
- **Flexible Configuration**: The `CONFIG` dictionary in `AHDL_EMNIST.py` allows easy customization of simulation parameters, model settings, and visualization options.

## Repository Structure

```
AdaptiveDynamicML/
├── AHDL_EMNIST.py           # Main script to run EMNIST concept drift simulations
├── src/                     # Source code for models, utilities, and visualizations
│   ├── __init__.py          # Makes src/ a Python package for proper imports
│   ├── models.py            # Definitions for FastModel, SlowModel, and Orchestrator
│   ├── plottings.py         # Visualization functions for plotting metrics and drifts
│   └── utils.py             # Helper functions for data handling and drift simulation
├── data/                    # Directory for storing datasets (original and processed)
├── results/                 # Directory for simulation outputs
│   └── plots/               # Subdirectory for generated plots and Chart.js configurations (chart_config_<uuid>.json)
├── config/                  # (Optional) Directory for configuration files
├── notebooks/               # (Optional) Directory for Jupyter notebooks for analysis
├── .gitignore               # Specifies files/folders to ignore (e.g., __pycache__, plot_dir/)
├── README.md                # Project overview and setup instructions
└── requirements.txt         # List of required Python dependencies
```

## Setup and Execution

### System Requirements
- Python 3.9 or later
- Conda or virtualenv (Conda recommended for dependency management)
- Optional: GPU support (e.g., Apple Silicon M1/M2/M3 with `tensorflow-metal`) for faster TensorFlow performance

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/franjgs/AdaptiveDynamicML.git
   cd AdaptiveDynamicML
   ```

2. **Set Up Conda Environment**:
   ```bash
   conda create -n base python=3.9
   conda activate base
   ```

3. **Install Dependencies**:
   Ensure the Conda environment is active, then run:
   ```bash
   pip install -r requirements.txt
   ```
   For Apple Silicon (M1/M2/M3), use:
   ```bash
   pip install tensorflow-macos==2.19.0 tensorflow-metal numpy==1.26.4 tensorflow-datasets matplotlib==3.6.3
   ```
   **Note**: Verify TensorFlow compatibility for your system. For Apple Silicon, ensure `tensorflow-macos` and `tensorflow-metal` are installed correctly for GPU acceleration.

### Running Simulations
Run the main script to execute the EMNIST simulation:
```bash
export TF_ENABLE_ONEDNN_OPTS=0  # Disable oneDNN optimizations
export CUDA_VISIBLE_DEVICES=""   # Disable GPU if not using tensorflow-metal
python AHDL_EMNIST.py
```
- Customize simulation parameters in the `CONFIG` dictionary in `AHDL_EMNIST.py`.
- Results (plots and Chart.js configurations) are saved in `results/plots/`.

### Visualizing Results
- **Matplotlib Plots**: Generated automatically and saved in `results/plots/` as PNG files, showing accuracy, loss, and beta evolution.
- **Chart.js Configuration**: Saved as `results/plots/chart_config_<uuid>.json` and printed to the console for web-based visualization (see [Using Chart.js Output](#using-chartjs-output)).

## Using Chart.js Output
The `create_chart` function in `src/plottings.py` generates a JSON configuration for Chart.js to visualize model accuracy with drift annotations. The configuration is saved as `results/plots/chart_config_<uuid>.json`. To use it:
1. Locate the JSON file in `results/plots/`.
2. Create an HTML file to render the chart in a web browser (example below).
3. Run Local Server: `python3 -m http.server 8000`
4. Open a browser and go to [http://localhost:8000/chart.html](http://localhost:8000/chart.html) to view the interactive chart.
5. Stop the server with Ctrl+C.

### Chart.js Example
```html
<!DOCTYPE html>
<html>
<head>
    <title>Model Accuracy Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation"></script>
</head>
<body>
    <canvas id="accuracyChart" style="max-height: 400px;"></canvas>
    <script>
        fetch('results/plots/chart_config_<uuid>.json')  // Replace <uuid> with the actual filename
            .then(response => {
                if (!response.ok) throw new Error('Failed to load chart configuration');
                return response.json();
            })
            .then(config => {
                new Chart(document.getElementById('accuracyChart'), config);
            })
            .catch(error => {
                console.error('Error loading chart:', error);
                alert('Could not load chart configuration. Check the file path and JSON format.');
            });
    </script>
</body>
</html>
```
Save as `chart.html` in the project root, replace `<uuid>` with the actual filename (e.g., `chart_config_123e4567-e89b-12d3-a456-426614174000.json`), and open in a browser.

## Contributions
Contributions are welcome! To contribute:
- Report bugs or suggest features by creating an **Issue**.
- Submit improvements via **Pull Requests** with new datasets, models, or drift strategies.
- Ensure code follows PEP 8 and includes tests where applicable.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
Francisco J. González-Serrano - [fran@ing.uc3m.es](mailto:fran@ing.uc3m.es)

## References
González-Serrano, F. J., & Figueiras-Vidal, A. R. (2017). *Generalized CMAC Adaptive Ensembles for Concept-Drifting Data Streams*. In 2017 25th European Signal Processing Conference (EUSIPCO) (pp. 1-5). IEEE. Kos, Greece. [Link](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570340671.pdf)
