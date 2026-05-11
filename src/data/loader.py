"""Data handling utilities for ensemble learning."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
    make_regression,
)
from sklearn.model_selection import train_test_split


class DataLoader:
    """Data loading and preprocessing utilities."""
    
    def __init__(self, random_state: int = 42) -> None:
        """Initialize data loader.
        
        Args:
            random_state: Random state for reproducibility.
        """
        self.random_state = random_state
        self.datasets = {
            "breast_cancer": load_breast_cancer,
            "digits": load_digits,
            "iris": load_iris,
            "wine": load_wine,
        }
    
    def load_dataset(
        self,
        dataset_name: str,
        return_X_y: bool = True,
        as_frame: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.Series], Any]:
        """Load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load.
            return_X_y: Whether to return X, y separately.
            as_frame: Whether to return as pandas DataFrame/Series.
            
        Returns:
            Dataset as specified format.
        """
        if dataset_name in self.datasets:
            loader = self.datasets[dataset_name]
            return loader(return_X_y=return_X_y, as_frame=as_frame)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def create_synthetic_dataset(
        self,
        task_type: str = "classification",
        n_samples: int = 1000,
        n_features: int = 20,
        n_classes: int = 2,
        noise: float = 0.1,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a synthetic dataset.
        
        Args:
            task_type: Type of task ('classification' or 'regression').
            n_samples: Number of samples.
            n_features: Number of features.
            n_classes: Number of classes (for classification).
            noise: Amount of noise to add.
            random_state: Random state.
            
        Returns:
            Tuple of (X, y).
        """
        rs = random_state or self.random_state
        
        if task_type == "classification":
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                n_redundant=0,
                n_informative=n_features,
                random_state=rs,
                noise=noise,
            )
        elif task_type == "regression":
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise,
                random_state=rs,
            )
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")
        
        logger.info(f"Created synthetic {task_type} dataset: {X.shape}, {y.shape}")
        return X, y
    
    def load_from_file(
        self,
        file_path: str,
        target_column: Optional[str] = None,
        sep: str = ",",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from file.
        
        Args:
            file_path: Path to the data file.
            target_column: Name of the target column.
            sep: Separator for CSV files.
            
        Returns:
            Tuple of (X, y).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, sep=sep)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        if target_column is None:
            # Assume last column is target
            target_column = df.columns[-1]
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        
        logger.info(f"Loaded data from {file_path}: {X.shape}, {y.shape}")
        return X, y


class DataSplitter:
    """Data splitting utilities for ensemble learning."""
    
    def __init__(self, random_state: int = 42) -> None:
        """Initialize data splitter.
        
        Args:
            random_state: Random state for reproducibility.
        """
        self.random_state = random_state
    
    def train_test_split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        test_size: float = 0.2,
        stratify: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            test_size: Proportion of data for testing.
            stratify: Whether to stratify the split.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify,
        )
    
    def train_val_test_split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        test_size: float = 0.2,
        val_size: float = 0.2,
        stratify: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation, and test sets.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            test_size: Proportion of data for testing.
            val_size: Proportion of data for validation.
            stratify: Whether to stratify the split.
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify,
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify,
        )
        
        logger.info(f"Data split: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def cross_validation_split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_folds: int = 5,
        stratify: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Create cross-validation splits.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            n_folds: Number of folds.
            stratify: Whether to stratify the splits.
            
        Returns:
            List of (X_train, X_val, y_train, y_val) tuples.
        """
        from sklearn.model_selection import StratifiedKFold, KFold
        
        if stratify is not None:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            splits = cv.split(X, stratify)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            splits = cv.split(X)
        
        cv_splits = []
        for train_idx, val_idx in splits:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            cv_splits.append((X_train, X_val, y_train, y_val))
        
        logger.info(f"Created {n_folds}-fold cross-validation splits")
        return cv_splits


class DataPreprocessor:
    """Data preprocessing utilities."""
    
    def __init__(self) -> None:
        """Initialize preprocessor."""
        self.scalers = {}
        self.encoders = {}
    
    def fit_transform(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        scaling: str = "standard",
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Fit scaler and transform data.
        
        Args:
            X_train: Training features.
            X_val: Validation features.
            X_test: Test features.
            scaling: Type of scaling ('standard', 'minmax', 'robust', 'none').
            
        Returns:
            Tuple of scaled features.
        """
        if scaling == "none":
            return X_train, X_val, X_test
        
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        scaler_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        
        if scaling not in scaler_map:
            raise ValueError(f"Unknown scaling method: {scaling}")
        
        scaler = scaler_map[scaling]
        X_train_scaled = scaler.fit_transform(X_train)
        
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
        
        self.scalers[scaling] = scaler
        logger.info(f"Applied {scaling} scaling to features")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def transform(
        self,
        X: np.ndarray,
        scaling: str = "standard",
    ) -> np.ndarray:
        """Transform data using fitted scaler.
        
        Args:
            X: Features to transform.
            scaling: Type of scaling used.
            
        Returns:
            Transformed features.
        """
        if scaling not in self.scalers:
            raise ValueError(f"Scaler for {scaling} not fitted")
        
        return self.scalers[scaling].transform(X)
