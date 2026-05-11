"""Training and evaluation pipeline for ensemble learning."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

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
from src.utils.core import set_seed


class EnsemblePipeline:
    """Complete pipeline for ensemble learning experiments."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize ensemble pipeline.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.random_state = config.get("random_state", 42)
        self.task_type = config.get("task_type", "classification")
        
        # Set random seed
        set_seed(self.random_state)
        
        # Initialize components
        self.data_loader = DataLoader(random_state=self.random_state)
        self.data_splitter = DataSplitter(random_state=self.random_state)
        self.data_preprocessor = DataPreprocessor()
        self.baseline_models = BaselineModels(random_state=self.random_state)
        self.model_trainer = ModelTrainer(random_state=self.random_state)
        self.evaluator = EnsembleEvaluator(task_type=self.task_type)
        
        # Data containers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Model containers
        self.baseline_models_dict = {}
        self.ensemble_models_dict = {}
        self.all_models_dict = {}
        
        # Results
        self.results = {}
        self.leaderboard = None
        
        logger.info("Initialized ensemble learning pipeline")
    
    def load_data(self) -> None:
        """Load and prepare data."""
        logger.info("Loading data...")
        
        dataset_config = self.config.get("dataset", {})
        dataset_name = dataset_config.get("name", "breast_cancer")
        
        if dataset_name in ["synthetic_classification", "synthetic_regression"]:
            # Create synthetic dataset
            task_type = dataset_name.split("_")[1]
            n_samples = dataset_config.get("n_samples", 1000)
            n_features = dataset_config.get("n_features", 20)
            n_classes = dataset_config.get("n_classes", 2)
            noise = dataset_config.get("noise", 0.1)
            
            X, y = self.data_loader.create_synthetic_dataset(
                task_type=task_type,
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                noise=noise,
            )
        else:
            # Load real dataset
            X, y = self.data_loader.load_dataset(dataset_name)
        
        logger.info(f"Loaded dataset: {X.shape}, {y.shape}")
        
        # Split data
        split_config = self.config.get("data_split", {})
        test_size = split_config.get("test_size", 0.2)
        val_size = split_config.get("val_size", 0.2)
        
        if val_size > 0:
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = (
                self.data_splitter.train_val_test_split(
                    X, y, test_size=test_size, val_size=val_size
                )
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                self.data_splitter.train_test_split(X, y, test_size=test_size)
            )
            self.X_val = None
            self.y_val = None
        
        logger.info(f"Data split: Train {self.X_train.shape}, Test {self.X_test.shape}")
        if self.X_val is not None:
            logger.info(f"Validation: {self.X_val.shape}")
    
    def preprocess_data(self) -> None:
        """Preprocess the data."""
        logger.info("Preprocessing data...")
        
        preprocessing_config = self.config.get("preprocessing", {})
        scaling = preprocessing_config.get("scaling", "standard")
        
        self.X_train, self.X_val, self.X_test = self.data_preprocessor.fit_transform(
            self.X_train, self.X_val, self.X_test, scaling=scaling
        )
        
        logger.info("Data preprocessing completed")
    
    def train_baseline_models(self) -> None:
        """Train baseline models."""
        logger.info("Training baseline models...")
        
        # Get baseline models
        self.baseline_models_dict = self.baseline_models.get_models(self.task_type)
        
        # Train models
        self.baseline_models_dict = self.model_trainer.train_models(
            self.baseline_models_dict,
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
        )
        
        logger.info(f"Trained {len(self.baseline_models_dict)} baseline models")
    
    def train_ensemble_models(self) -> None:
        """Train ensemble models."""
        logger.info("Training ensemble models...")
        
        ensemble_config = self.config.get("ensembles", {})
        
        # Voting Ensemble
        if ensemble_config.get("voting", {}).get("enabled", True):
            logger.info("Training voting ensemble...")
            voting_config = ensemble_config["voting"]
            
            # Select subset of baseline models for voting
            voting_models = list(self.baseline_models_dict.items())[:3]  # Use first 3 models
            
            voting_ensemble = VotingEnsemble(
                estimators=voting_models,
                voting=voting_config.get("voting_type", "hard"),
                weights=voting_config.get("weights", None),
            )
            
            voting_ensemble.fit(self.X_train, self.y_train)
            self.ensemble_models_dict["voting"] = voting_ensemble
        
        # Stacking Ensemble
        if ensemble_config.get("stacking", {}).get("enabled", True):
            logger.info("Training stacking ensemble...")
            stacking_config = ensemble_config["stacking"]
            
            # Select subset of baseline models for stacking
            stacking_models = list(self.baseline_models_dict.items())[:3]  # Use first 3 models
            
            stacking_ensemble = StackingEnsemble(
                base_estimators=stacking_models,
                meta_estimator=None,  # Use default
                cv=stacking_config.get("cv", 5),
            )
            
            stacking_ensemble.fit(self.X_train, self.y_train)
            self.ensemble_models_dict["stacking"] = stacking_ensemble
        
        # Bagging Ensemble
        if ensemble_config.get("bagging", {}).get("enabled", True):
            logger.info("Training bagging ensemble...")
            bagging_config = ensemble_config["bagging"]
            
            # Use decision tree as base estimator
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            
            if self.task_type == "classification":
                base_estimator = DecisionTreeClassifier(random_state=self.random_state)
            else:
                base_estimator = DecisionTreeRegressor(random_state=self.random_state)
            
            bagging_ensemble = BaggingEnsemble(
                base_estimator=base_estimator,
                n_estimators=bagging_config.get("n_estimators", 10),
                max_samples=bagging_config.get("max_samples", 1.0),
                max_features=bagging_config.get("max_features", 1.0),
            )
            
            bagging_ensemble.fit(self.X_train, self.y_train)
            self.ensemble_models_dict["bagging"] = bagging_ensemble
        
        # Dynamic Selection Ensemble
        if ensemble_config.get("dynamic_selection", {}).get("enabled", True):
            logger.info("Training dynamic selection ensemble...")
            ds_config = ensemble_config["dynamic_selection"]
            
            # Use all baseline models
            ds_ensemble = DynamicSelectionEnsemble(
                base_estimators=list(self.baseline_models_dict.items()),
                selection_method=ds_config.get("selection_method", "best"),
                k=ds_config.get("k", 5),
            )
            
            ds_ensemble.fit(self.X_train, self.y_train)
            self.ensemble_models_dict["dynamic_selection"] = ds_ensemble
        
        # Meta-Learning Ensemble
        if ensemble_config.get("meta_learning", {}).get("enabled", True):
            logger.info("Training meta-learning ensemble...")
            ml_config = ensemble_config["meta_learning"]
            
            # Use all baseline models
            ml_ensemble = MetaLearningEnsemble(
                base_estimators=list(self.baseline_models_dict.items()),
                meta_learner=None,  # Use default
                feature_extraction=ml_config.get("feature_extraction", "predictions"),
            )
            
            ml_ensemble.fit(self.X_train, self.y_train)
            self.ensemble_models_dict["meta_learning"] = ml_ensemble
        
        logger.info(f"Trained {len(self.ensemble_models_dict)} ensemble models")
    
    def evaluate_models(self) -> None:
        """Evaluate all models."""
        logger.info("Evaluating models...")
        
        # Combine all models
        self.all_models_dict = {**self.baseline_models_dict, **self.ensemble_models_dict}
        
        # Evaluate models
        self.results = self.evaluator.evaluate_models(
            self.all_models_dict,
            self.X_test,
            self.y_test,
            self.X_val,
            self.y_val,
        )
        
        # Create leaderboard
        primary_metric = self.config.get("evaluation", {}).get("primary_metric", "accuracy")
        self.leaderboard = self.evaluator.create_leaderboard(
            self.results, metric=primary_metric
        )
        
        logger.info("Model evaluation completed")
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment pipeline.
        
        Returns:
            Experiment results.
        """
        start_time = time.time()
        
        logger.info("Starting ensemble learning experiment...")
        
        try:
            # Load and preprocess data
            self.load_data()
            self.preprocess_data()
            
            # Train models
            self.train_baseline_models()
            self.train_ensemble_models()
            
            # Evaluate models
            self.evaluate_models()
            
            # Get best model
            primary_metric = self.config.get("evaluation", {}).get("primary_metric", "accuracy")
            best_model, best_score = self.evaluator.get_best_model(
                self.results, metric=primary_metric
            )
            
            # Calculate improvements
            improvements = {}
            if len(self.baseline_models_dict) > 0:
                baseline_name = list(self.baseline_models_dict.keys())[0]
                for ensemble_name in self.ensemble_models_dict.keys():
                    improvement = self.evaluator.calculate_improvement(
                        baseline_name, ensemble_name, metric=primary_metric
                    )
                    improvements[f"{ensemble_name}_vs_{baseline_name}"] = improvement
            
            experiment_results = {
                "config": self.config,
                "data_info": {
                    "train_shape": self.X_train.shape,
                    "test_shape": self.X_test.shape,
                    "val_shape": self.X_val.shape if self.X_val is not None else None,
                },
                "model_counts": {
                    "baseline": len(self.baseline_models_dict),
                    "ensemble": len(self.ensemble_models_dict),
                    "total": len(self.all_models_dict),
                },
                "results": self.results,
                "leaderboard": self.leaderboard,
                "best_model": best_model,
                "best_score": best_score,
                "improvements": improvements,
                "execution_time": time.time() - start_time,
            }
            
            logger.info(f"Experiment completed in {experiment_results['execution_time']:.2f} seconds")
            logger.info(f"Best model: {best_model} with {primary_metric}={best_score:.4f}")
            
            return experiment_results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
    
    def save_results(self, save_path: str) -> None:
        """Save experiment results.
        
        Args:
            save_path: Path to save results.
        """
        import pickle
        
        results_to_save = {
            "config": self.config,
            "results": self.results,
            "leaderboard": self.leaderboard,
            "data_info": {
                "train_shape": self.X_train.shape,
                "test_shape": self.X_test.shape,
                "val_shape": self.X_val.shape if self.X_val is not None else None,
            },
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(results_to_save, f)
        
        logger.info(f"Results saved to {save_path}")
    
    def print_summary(self) -> None:
        """Print experiment summary."""
        if self.leaderboard is not None:
            print("\n" + "="*50)
            print("ENSEMBLE LEARNING EXPERIMENT SUMMARY")
            print("="*50)
            
            print(f"\nDataset: {self.config.get('dataset', {}).get('name', 'unknown')}")
            print(f"Task Type: {self.task_type}")
            print(f"Train Size: {self.X_train.shape}")
            print(f"Test Size: {self.X_test.shape}")
            
            if self.X_val is not None:
                print(f"Validation Size: {self.X_val.shape}")
            
            print(f"\nModels Trained:")
            print(f"  Baseline: {len(self.baseline_models_dict)}")
            print(f"  Ensemble: {len(self.ensemble_models_dict)}")
            print(f"  Total: {len(self.all_models_dict)}")
            
            print(f"\nTop 5 Models:")
            print(self.leaderboard.head().to_string(index=False))
            
            if self.results:
                primary_metric = self.config.get("evaluation", {}).get("primary_metric", "accuracy")
                best_model, best_score = self.evaluator.get_best_model(
                    self.results, metric=primary_metric
                )
                print(f"\nBest Model: {best_model} ({primary_metric}={best_score:.4f})")
            
            print("="*50)
