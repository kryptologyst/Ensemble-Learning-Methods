"""Evaluation metrics and framework for ensemble learning."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)


class ClassificationMetrics:
    """Classification metrics calculator."""
    
    def __init__(self) -> None:
        """Initialize classification metrics."""
        self.metrics = {}
    
    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = "weighted",
    ) -> Dict[str, float]:
        """Calculate all classification metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities.
            average: Averaging method for multi-class.
            
        Returns:
            Dictionary of metrics.
        """
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # ROC AUC (if probabilities available)
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics["roc_auc"] = 0.0
        
        self.metrics = metrics
        logger.info(f"Calculated {len(metrics)} classification metrics")
        return metrics
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """Get detailed classification report.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            target_names: Names of target classes.
            
        Returns:
            Classification report string.
        """
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False,
    ) -> np.ndarray:
        """Get confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            normalize: Whether to normalize the matrix.
            
        Returns:
            Confusion matrix.
        """
        return confusion_matrix(y_true, y_pred, normalize=normalize)


class RegressionMetrics:
    """Regression metrics calculator."""
    
    def __init__(self) -> None:
        """Initialize regression metrics."""
        self.metrics = {}
    
    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate all regression metrics.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            Dictionary of metrics.
        """
        metrics = {}
        
        # Basic metrics
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2"] = r2_score(y_true, y_pred)
        
        # Additional metrics
        metrics["mape"] = self._calculate_mape(y_true, y_pred)
        metrics["smape"] = self._calculate_smape(y_true, y_pred)
        
        self.metrics = metrics
        logger.info(f"Calculated {len(metrics)} regression metrics")
        return metrics
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not np.any(mask):
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if not np.any(mask):
            return 0.0
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


class EnsembleEvaluator:
    """Comprehensive ensemble evaluation framework."""
    
    def __init__(self, task_type: str = "classification") -> None:
        """Initialize ensemble evaluator.
        
        Args:
            task_type: Type of task ('classification' or 'regression').
        """
        self.task_type = task_type
        self.results = {}
        
        if task_type == "classification":
            self.metrics_calculator = ClassificationMetrics()
        else:
            self.metrics_calculator = RegressionMetrics()
    
    def evaluate_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple models.
        
        Args:
            models: Dictionary of trained models.
            X_test: Test features.
            y_test: Test targets.
            X_val: Validation features.
            y_val: Validation targets.
            
        Returns:
            Dictionary of model results.
        """
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            try:
                # Test set evaluation
                y_pred_test = model.predict(X_test)
                
                if self.task_type == "classification":
                    y_proba_test = None
                    if hasattr(model, "predict_proba"):
                        y_proba_test = model.predict_proba(X_test)
                    
                    test_metrics = self.metrics_calculator.calculate_all(
                        y_test, y_pred_test, y_proba_test
                    )
                else:
                    test_metrics = self.metrics_calculator.calculate_all(
                        y_test, y_pred_test
                    )
                
                results[name] = {
                    "test": test_metrics,
                }
                
                # Validation set evaluation (if available)
                if X_val is not None and y_val is not None:
                    y_pred_val = model.predict(X_val)
                    
                    if self.task_type == "classification":
                        y_proba_val = None
                        if hasattr(model, "predict_proba"):
                            y_proba_val = model.predict_proba(X_val)
                        
                        val_metrics = self.metrics_calculator.calculate_all(
                            y_val, y_pred_val, y_proba_val
                        )
                    else:
                        val_metrics = self.metrics_calculator.calculate_all(
                            y_val, y_pred_val
                        )
                    
                    results[name]["validation"] = val_metrics
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                continue
        
        self.results = results
        logger.info(f"Evaluated {len(results)} models")
        return results
    
    def create_leaderboard(
        self,
        results: Optional[Dict[str, Dict[str, float]]] = None,
        metric: str = "accuracy",
        dataset: str = "test",
    ) -> pd.DataFrame:
        """Create a leaderboard from results.
        
        Args:
            results: Model results dictionary.
            metric: Primary metric for ranking.
            dataset: Dataset to rank on ('test' or 'validation').
            
        Returns:
            Leaderboard DataFrame.
        """
        if results is None:
            results = self.results
        
        if not results:
            logger.warning("No results available for leaderboard")
            return pd.DataFrame()
        
        leaderboard_data = []
        
        for model_name, model_results in results.items():
            if dataset in model_results:
                metrics = model_results[dataset]
                
                row = {"model": model_name}
                row.update(metrics)
                leaderboard_data.append(row)
        
        if not leaderboard_data:
            logger.warning(f"No data available for dataset: {dataset}")
            return pd.DataFrame()
        
        leaderboard = pd.DataFrame(leaderboard_data)
        
        # Sort by primary metric
        if metric in leaderboard.columns:
            leaderboard = leaderboard.sort_values(metric, ascending=False)
            leaderboard["rank"] = range(1, len(leaderboard) + 1)
        
        logger.info(f"Created leaderboard with {len(leaderboard)} models")
        return leaderboard
    
    def compare_models(
        self,
        results: Optional[Dict[str, Dict[str, float]]] = None,
        metric: str = "accuracy",
        dataset: str = "test",
    ) -> pd.DataFrame:
        """Compare models side by side.
        
        Args:
            results: Model results dictionary.
            metric: Primary metric for comparison.
            dataset: Dataset to compare on.
            
        Returns:
            Comparison DataFrame.
        """
        if results is None:
            results = self.results
        
        if not results:
            logger.warning("No results available for comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, model_results in results.items():
            if dataset in model_results:
                metrics = model_results[dataset]
                row = {"model": model_name}
                row.update(metrics)
                comparison_data.append(row)
        
        if not comparison_data:
            logger.warning(f"No data available for dataset: {dataset}")
            return pd.DataFrame()
        
        comparison = pd.DataFrame(comparison_data)
        comparison = comparison.set_index("model")
        
        logger.info(f"Created comparison table with {len(comparison)} models")
        return comparison
    
    def get_best_model(
        self,
        results: Optional[Dict[str, Dict[str, float]]] = None,
        metric: str = "accuracy",
        dataset: str = "test",
    ) -> Tuple[str, float]:
        """Get the best performing model.
        
        Args:
            results: Model results dictionary.
            metric: Metric to optimize.
            dataset: Dataset to evaluate on.
            
        Returns:
            Tuple of (model_name, score).
        """
        if results is None:
            results = self.results
        
        if not results:
            logger.warning("No results available")
            return "", 0.0
        
        best_model = ""
        best_score = -np.inf if metric in ["accuracy", "f1", "precision", "recall", "r2", "roc_auc"] else np.inf
        
        for model_name, model_results in results.items():
            if dataset in model_results and metric in model_results[dataset]:
                score = model_results[dataset][metric]
                
                if metric in ["accuracy", "f1", "precision", "recall", "r2", "roc_auc"]:
                    if score > best_score:
                        best_score = score
                        best_model = model_name
                else:  # For metrics like mse, mae, etc.
                    if score < best_score:
                        best_score = score
                        best_model = model_name
        
        logger.info(f"Best model: {best_model} with {metric}={best_score:.4f}")
        return best_model, best_score
    
    def calculate_improvement(
        self,
        baseline_model: str,
        ensemble_model: str,
        metric: str = "accuracy",
        dataset: str = "test",
    ) -> float:
        """Calculate improvement of ensemble over baseline.
        
        Args:
            baseline_model: Name of baseline model.
            ensemble_model: Name of ensemble model.
            metric: Metric to compare.
            dataset: Dataset to compare on.
            
        Returns:
            Improvement percentage.
        """
        if baseline_model not in self.results or ensemble_model not in self.results:
            logger.warning("Models not found in results")
            return 0.0
        
        baseline_score = self.results[baseline_model][dataset][metric]
        ensemble_score = self.results[ensemble_model][dataset][metric]
        
        if metric in ["accuracy", "f1", "precision", "recall", "r2", "roc_auc"]:
            improvement = ((ensemble_score - baseline_score) / baseline_score) * 100
        else:  # For metrics like mse, mae, etc.
            improvement = ((baseline_score - ensemble_score) / baseline_score) * 100
        
        logger.info(f"Improvement: {improvement:.2f}%")
        return improvement
