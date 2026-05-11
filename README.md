# Ensemble Learning Methods

A comprehensive implementation of ensemble learning methods for research and education purposes. This project showcases various ensemble techniques including Voting, Stacking, Bagging, Dynamic Selection, and Meta-Learning approaches.

## Author

**kryptologyst** - [GitHub](https://github.com/kryptologyst)

## Overview

Ensemble learning combines multiple models to improve predictive performance, stability, and generalization. This project provides:

- **Strong Baselines**: Logistic regression, decision trees, k-NN, SVM, Random Forest, Extra Trees, Gradient Boosting, AdaBoost
- **Advanced Ensemble Methods**: Voting, Stacking, Bagging, Dynamic Selection, Meta-Learning
- **Comprehensive Evaluation**: Multiple metrics, leaderboards, and performance comparisons
- **Interactive Demos**: Streamlit and Gradio applications for hands-on exploration
- **Modern ML Practices**: Type hints, configuration management, reproducible experiments

## Features

### Ensemble Methods Implemented

1. **Voting Ensembles**
   - Hard voting (majority vote)
   - Soft voting (weighted average of probabilities)

2. **Stacking (Stacked Generalization)**
   - Meta-learner approach
   - Cross-validation for robust predictions

3. **Bagging (Bootstrap Aggregating)**
   - Bootstrap sampling
   - Feature and sample randomization

4. **Dynamic Selection**
   - Best model selection
   - k-best ensemble
   - Local accuracy-based selection

5. **Meta-Learning**
   - Feature extraction from base models
   - Meta-learner optimization

### Evaluation Framework

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression Metrics**: R², MSE, RMSE, MAE, MAPE, SMAPE
- **Model Comparison**: Side-by-side performance analysis
- **Leaderboards**: Ranked model performance
- **Improvement Analysis**: Ensemble vs baseline comparisons

### Visualization Tools

- Model leaderboards
- Metric comparisons
- Confusion matrices
- Learning curves
- Feature importance plots
- Summary dashboards

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Ensemble-Learning-Methods.git
cd Ensemble-Learning-Methods

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"

# Install deep learning dependencies (optional)
pip install -e ".[deep]"

# Install tracking dependencies (optional)
pip install -e ".[tracking]"
```

### Manual Installation

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn plotly tqdm pyyaml omegaconf hydra-core tensorboard optuna xgboost lightgbm catboost streamlit gradio fastapi uvicorn pydantic rich loguru
```

## Quick Start

### Command Line Interface

```bash
# Train models with default configuration
python -m src.cli train

# Train with custom configuration
python -m src.cli train --config configs/regression.yaml

# Evaluate results
python -m src.cli eval --results results/experiment_results.pkl

# Run interactive demo
python -m src.cli demo

# Create default configuration
python -m src.cli create-config --output my_config.yaml
```

### Python API

```python
from src.train.pipeline import EnsemblePipeline
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("configs/default.yaml")

# Initialize pipeline
pipeline = EnsemblePipeline(config)

# Run experiment
results = pipeline.run_experiment()

# Print summary
pipeline.print_summary()
```

### Interactive Demos

#### Streamlit Demo
```bash
streamlit run demo/streamlit_app.py
```

#### Gradio Demo
```bash
python demo/gradio_app.py
```

## Configuration

The project uses YAML configuration files for easy experimentation:

```yaml
# configs/default.yaml
random_state: 42
task_type: classification

dataset:
  name: breast_cancer
  n_samples: 1000
  n_features: 20

data_split:
  test_size: 0.2
  val_size: 0.2

preprocessing:
  scaling: standard

ensembles:
  voting:
    enabled: true
    voting_type: hard
  stacking:
    enabled: true
    cv: 5
  bagging:
    enabled: true
    n_estimators: 10
```

## Dataset Support

### Built-in Datasets
- **Classification**: Breast Cancer, Digits, Iris, Wine
- **Regression**: Synthetic regression datasets

### Custom Datasets
- CSV, Excel, Parquet files
- Custom data loaders
- Synthetic data generation

## Project Structure

```
ensemble-learning-methods/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model implementations
│   ├── metrics/        # Evaluation metrics
│   ├── train/          # Training pipelines
│   ├── viz/            # Visualization tools
│   ├── utils/          # Core utilities
│   └── cli.py          # Command-line interface
├── configs/            # Configuration files
├── demo/               # Interactive demos
├── tests/              # Unit tests
├── data/               # Data storage
├── results/            # Experiment results
├── assets/             # Generated assets
└── notebooks/          # Jupyter notebooks
```

## Usage Examples

### Basic Classification Experiment

```python
from src.train.pipeline import EnsemblePipeline
from omegaconf import OmegaConf

# Create configuration
config = {
    "random_state": 42,
    "task_type": "classification",
    "dataset": {"name": "breast_cancer"},
    "data_split": {"test_size": 0.2, "val_size": 0.2},
    "preprocessing": {"scaling": "standard"},
    "ensembles": {
        "voting": {"enabled": True},
        "stacking": {"enabled": True},
        "bagging": {"enabled": True}
    },
    "evaluation": {"primary_metric": "accuracy"}
}

config = OmegaConf.create(config)

# Run experiment
pipeline = EnsemblePipeline(config)
results = pipeline.run_experiment()

# Display results
print(f"Best model: {results['best_model']}")
print(f"Best accuracy: {results['best_score']:.4f}")
```

### Custom Ensemble Method

```python
from src.models.ensembles import VotingEnsemble
from src.models.baselines import BaselineModels

# Get baseline models
baseline_models = BaselineModels().get_classification_models()

# Create voting ensemble
voting_ensemble = VotingEnsemble(
    estimators=list(baseline_models.items())[:3],
    voting="soft"
)

# Train and evaluate
voting_ensemble.fit(X_train, y_train)
predictions = voting_ensemble.predict(X_test)
```

### Visualization

```python
from src.viz.visualizer import EnsembleVisualizer

# Create visualizer
visualizer = EnsembleVisualizer()

# Plot leaderboard
visualizer.plot_leaderboard(leaderboard, metric="accuracy")

# Plot metric comparison
visualizer.plot_metric_comparison(results, ["accuracy", "f1", "precision"])

# Create summary dashboard
visualizer.create_summary_dashboard(leaderboard, results, config)
```

## Expected Performance

### Classification (Breast Cancer Dataset)
- **Baseline Models**: 0.90-0.96 accuracy
- **Voting Ensemble**: 0.95-0.97 accuracy
- **Stacking Ensemble**: 0.96-0.98 accuracy
- **Bagging Ensemble**: 0.94-0.97 accuracy

### Regression (Synthetic Dataset)
- **Baseline Models**: 0.80-0.95 R²
- **Voting Ensemble**: 0.85-0.96 R²
- **Stacking Ensemble**: 0.90-0.98 R²
- **Bagging Ensemble**: 0.88-0.97 R²

*Note: Performance may vary based on dataset characteristics and hyperparameters.*

## Safety and Limitations

### Important Disclaimers

⚠️ **This is a research and educational tool. Results should NOT be used for production decisions without proper validation and expert review.**

### Safety Considerations

- **No Production Use**: This tool is designed for research and education only
- **Data Privacy**: Ensure compliance with data protection regulations
- **Model Validation**: Always validate models on independent test sets
- **Expert Review**: Consult domain experts for critical applications

### Limitations

- **Dataset Dependent**: Performance varies significantly across datasets
- **Hyperparameter Sensitivity**: Results depend on proper hyperparameter tuning
- **Computational Cost**: Some ensemble methods are computationally expensive
- **Overfitting Risk**: Complex ensembles may overfit on small datasets

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff check src/

# Type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{ensemble_learning_methods,
  title={Ensemble Learning Methods: A Comprehensive Implementation},
  author={kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Ensemble-Learning-Methods}
}
```

## Acknowledgments

- Scikit-learn team for the excellent ML library
- The open-source community for various dependencies
- Contributors and users who provide feedback
# Ensemble-Learning-Methods
