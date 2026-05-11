"""Core utilities for ensemble learning methods."""

import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Try to set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"PyTorch seed set to {seed}")
    except ImportError:
        pass
    
    logger.info(f"Random seed set to {seed}")


def get_device() -> str:
    """Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu').
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Silicon MPS device")
        else:
            device = "cpu"
            logger.info("Using CPU device")
    except ImportError:
        device = "cpu"
        logger.info("PyTorch not available, using CPU")
    
    return device


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration object.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object.
        save_path: Path to save configuration.
    """
    OmegaConf.save(config, save_path)
    logger.info(f"Saved configuration to {save_path}")


def create_directories(paths: List[str]) -> None:
    """Create directories if they don't exist.
    
    Args:
        paths: List of directory paths to create.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Created directory: {path}")


def validate_data_split(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_size: float = 0.2,
    val_size: Optional[float] = None,
) -> bool:
    """Validate data split parameters.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Proportion of data for testing.
        val_size: Proportion of data for validation.
        
    Returns:
        True if validation passes.
    """
    if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    if val_size is not None:
        if not isinstance(val_size, (int, float)) or not 0 < val_size < 1:
            raise ValueError("val_size must be between 0 and 1")
        if test_size + val_size >= 1:
            raise ValueError("test_size + val_size must be less than 1")
    
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    
    if len(X) == 0:
        raise ValueError("Data cannot be empty")
    
    return True


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def log_model_info(model: Any, model_name: str) -> None:
    """Log model information.
    
    Args:
        model: Model object.
        model_name: Name of the model.
    """
    logger.info(f"Model: {model_name}")
    logger.info(f"Type: {type(model).__name__}")
    
    # Try to get model parameters
    try:
        if hasattr(model, "get_params"):
            params = model.get_params()
            logger.info(f"Parameters: {params}")
    except Exception:
        pass


class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        mode: str = "min",
    ) -> None:
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            restore_best_weights: Whether to restore best weights.
            mode: 'min' for minimizing, 'max' for maximizing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: Any) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score.
            model: Model object.
            
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights and hasattr(model, "state_dict"):
                self.best_weights = model.state_dict().copy()
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights and hasattr(model, "state_dict"):
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
            
        return False
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score is an improvement."""
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
