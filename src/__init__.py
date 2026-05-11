"""Ensemble Learning Methods - Research and Education Tool."""

__version__ = "1.0.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"
__description__ = "Comprehensive ensemble learning methods implementation with modern ML practices"

from src.data.loader import DataLoader, DataPreprocessor, DataSplitter
from src.metrics.evaluator import EnsembleEvaluator
from src.models.baselines import BaselineModels, ModelTrainer
from src.models.ensembles import (
    BaggingEnsemble,
    DynamicSelectionEnsemble,
    MetaLearningEnsemble,
    StackingEnsemble,
    VotingEnsemble,
)
from src.train.pipeline import EnsemblePipeline
from src.utils.core import set_seed, get_device
from src.viz.visualizer import EnsembleVisualizer

__all__ = [
    # Core utilities
    "set_seed",
    "get_device",
    
    # Data handling
    "DataLoader",
    "DataPreprocessor", 
    "DataSplitter",
    
    # Models
    "BaselineModels",
    "ModelTrainer",
    "VotingEnsemble",
    "StackingEnsemble",
    "BaggingEnsemble",
    "DynamicSelectionEnsemble",
    "MetaLearningEnsemble",
    
    # Evaluation
    "EnsembleEvaluator",
    
    # Pipeline
    "EnsemblePipeline",
    
    # Visualization
    "EnsembleVisualizer",
]
