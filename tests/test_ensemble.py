"""Test suite for ensemble learning methods."""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression

from src.data.loader import DataLoader, DataSplitter, DataPreprocessor
from src.models.baselines import BaselineModels, ModelTrainer
from src.models.ensembles import VotingEnsemble, StackingEnsemble, BaggingEnsemble
from src.metrics.evaluator import EnsembleEvaluator, ClassificationMetrics, RegressionMetrics
from src.utils.core import set_seed


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_synthetic_classification(self):
        """Test synthetic classification dataset creation."""
        loader = DataLoader()
        X, y = loader.create_synthetic_dataset(
            task_type="classification",
            n_samples=100,
            n_features=10,
            n_classes=2
        )
        
        assert X.shape == (100, 10)
        assert y.shape == (100,)
        assert len(np.unique(y)) == 2
    
    def test_synthetic_regression(self):
        """Test synthetic regression dataset creation."""
        loader = DataLoader()
        X, y = loader.create_synthetic_dataset(
            task_type="regression",
            n_samples=100,
            n_features=10
        )
        
        assert X.shape == (100, 10)
        assert y.shape == (100,)
    
    def test_builtin_dataset(self):
        """Test built-in dataset loading."""
        loader = DataLoader()
        X, y = loader.load_dataset("iris")
        
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] > 0


class TestDataSplitter:
    """Test data splitting functionality."""
    
    def test_train_test_split(self):
        """Test train-test split."""
        splitter = DataSplitter(random_state=42)
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        X_train, X_test, y_train, y_test = splitter.train_test_split(X, y, test_size=0.2)
        
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20
    
    def test_train_val_test_split(self):
        """Test train-validation-test split."""
        splitter = DataSplitter(random_state=42)
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            X, y, test_size=0.2, val_size=0.2
        )
        
        assert X_train.shape[0] == 64
        assert X_val.shape[0] == 16
        assert X_test.shape[0] == 20


class TestDataPreprocessor:
    """Test data preprocessing functionality."""
    
    def test_standard_scaling(self):
        """Test standard scaling."""
        preprocessor = DataPreprocessor()
        X_train = np.random.randn(100, 10)
        X_test = np.random.randn(50, 10)
        
        X_train_scaled, X_test_scaled, _ = preprocessor.fit_transform(
            X_train, X_test, None, scaling="standard"
        )
        
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        assert np.allclose(np.mean(X_train_scaled, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_train_scaled, axis=0), 1, atol=1e-10)


class TestBaselineModels:
    """Test baseline model functionality."""
    
    def test_classification_models(self):
        """Test classification model creation."""
        models = BaselineModels().get_classification_models()
        
        assert len(models) > 0
        assert "logistic_regression" in models
        assert "random_forest" in models
    
    def test_regression_models(self):
        """Test regression model creation."""
        models = BaselineModels().get_regression_models()
        
        assert len(models) > 0
        assert "ridge" in models
        assert "random_forest" in models


class TestEnsembleModels:
    """Test ensemble model functionality."""
    
    def test_voting_ensemble(self):
        """Test voting ensemble."""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        
        estimators = [
            ("dt", DecisionTreeClassifier(random_state=42)),
            ("lr", LogisticRegression(random_state=42))
        ]
        
        ensemble = VotingEnsemble(estimators, voting="hard")
        
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        ensemble.fit(X, y)
        
        predictions = ensemble.predict(X)
        assert len(predictions) == len(y)
    
    def test_stacking_ensemble(self):
        """Test stacking ensemble."""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        
        estimators = [
            ("dt", DecisionTreeClassifier(random_state=42)),
            ("lr", LogisticRegression(random_state=42))
        ]
        
        ensemble = StackingEnsemble(estimators)
        
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        ensemble.fit(X, y)
        
        predictions = ensemble.predict(X)
        assert len(predictions) == len(y)
    
    def test_bagging_ensemble(self):
        """Test bagging ensemble."""
        from sklearn.tree import DecisionTreeClassifier
        
        ensemble = BaggingEnsemble(
            base_estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=5
        )
        
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        ensemble.fit(X, y)
        
        predictions = ensemble.predict(X)
        assert len(predictions) == len(y)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        metrics_calc = ClassificationMetrics()
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1])
        y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])
        
        metrics = metrics_calc.calculate_all(y_true, y_pred, y_proba)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
    
    def test_regression_metrics(self):
        """Test regression metrics calculation."""
        metrics_calc = RegressionMetrics()
        
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = metrics_calc.calculate_all(y_true, y_pred)
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert "smape" in metrics


class TestEvaluator:
    """Test ensemble evaluator."""
    
    def test_classification_evaluation(self):
        """Test classification evaluation."""
        evaluator = EnsembleEvaluator(task_type="classification")
        
        # Create mock models
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        
        models = {
            "dt": DecisionTreeClassifier(random_state=42),
            "lr": LogisticRegression(random_state=42)
        }
        
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        # Train models
        for model in models.values():
            model.fit(X, y)
        
        # Evaluate
        results = evaluator.evaluate_models(models, X, y)
        
        assert len(results) == 2
        assert "dt" in results
        assert "lr" in results
    
    def test_regression_evaluation(self):
        """Test regression evaluation."""
        evaluator = EnsembleEvaluator(task_type="regression")
        
        # Create mock models
        from sklearn.linear_model import Ridge
        from sklearn.tree import DecisionTreeRegressor
        
        models = {
            "ridge": Ridge(random_state=42),
            "dt": DecisionTreeRegressor(random_state=42)
        }
        
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        
        # Train models
        for model in models.values():
            model.fit(X, y)
        
        # Evaluate
        results = evaluator.evaluate_models(models, X, y)
        
        assert len(results) == 2
        assert "ridge" in results
        assert "dt" in results


class TestUtilities:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate random numbers
        np.random.seed(42)
        val1 = np.random.rand()
        
        set_seed(42)
        np.random.seed(42)
        val2 = np.random.rand()
        
        assert val1 == val2


if __name__ == "__main__":
    pytest.main([__file__])
