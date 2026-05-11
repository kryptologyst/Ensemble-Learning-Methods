"""Visualization utilities for ensemble learning results."""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import confusion_matrix


class EnsembleVisualizer:
    """Visualization utilities for ensemble learning."""
    
    def __init__(self, style: str = "whitegrid", figsize: Tuple[int, int] = (10, 8)) -> None:
        """Initialize visualizer.
        
        Args:
            style: Seaborn style.
            figsize: Default figure size.
        """
        self.style = style
        self.figsize = figsize
        sns.set_style(style)
        plt.rcParams["figure.figsize"] = figsize
        
    def plot_leaderboard(
        self,
        leaderboard: pd.DataFrame,
        metric: str = "accuracy",
        top_k: int = 10,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot model leaderboard.
        
        Args:
            leaderboard: Leaderboard DataFrame.
            metric: Primary metric to plot.
            top_k: Number of top models to show.
            save_path: Path to save the plot.
        """
        if leaderboard.empty:
            logger.warning("Empty leaderboard provided")
            return
        
        # Get top k models
        top_models = leaderboard.head(top_k)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_models)), top_models[metric])
        
        # Color bars based on model type
        colors = []
        for model_name in top_models["model"]:
            if any(ensemble_type in model_name.lower() for ensemble_type in 
                   ["voting", "stacking", "bagging", "dynamic", "meta"]):
                colors.append("orange")  # Ensemble models
            else:
                colors.append("skyblue")  # Baseline models
        
        for i, (bar, color) in enumerate(zip(bars, colors)):
            bar.set_color(color)
        
        plt.yticks(range(len(top_models)), top_models["model"])
        plt.xlabel(metric.title())
        plt.title(f"Top {top_k} Models - {metric.title()}")
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, v in enumerate(top_models[metric]):
            plt.text(v + 0.001, i, f"{v:.4f}", va="center")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Leaderboard plot saved to {save_path}")
        
        plt.show()
    
    def plot_metric_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        dataset: str = "test",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot comparison of multiple metrics across models.
        
        Args:
            results: Model results dictionary.
            metrics: List of metrics to compare.
            dataset: Dataset to compare on.
            save_path: Path to save the plot.
        """
        if not results:
            logger.warning("No results provided")
            return
        
        # Prepare data
        model_names = []
        metric_values = {metric: [] for metric in metrics}
        
        for model_name, model_results in results.items():
            if dataset in model_results:
                model_names.append(model_name)
                for metric in metrics:
                    if metric in model_results[dataset]:
                        metric_values[metric].append(model_results[dataset][metric])
                    else:
                        metric_values[metric].append(0.0)
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Color bars based on model type
            colors = []
            for model_name in model_names:
                if any(ensemble_type in model_name.lower() for ensemble_type in 
                       ["voting", "stacking", "bagging", "dynamic", "meta"]):
                    colors.append("orange")
                else:
                    colors.append("skyblue")
            
            bars = ax.bar(range(len(model_names)), metric_values[metric], color=colors)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha="right")
            ax.set_ylabel(metric.title())
            ax.set_title(f"{metric.title()} Comparison")
            
            # Add value labels
            for bar, value in zip(bars, metric_values[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f"{value:.3f}", ha="center", va="bottom")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Metric comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        normalize: bool = False,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            model_name: Name of the model.
            normalize: Whether to normalize the matrix.
            save_path: Path to save the plot.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            fmt = "d"
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(
        self,
        train_scores: List[float],
        val_scores: List[float],
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot learning curves.
        
        Args:
            train_scores: Training scores.
            val_scores: Validation scores.
            model_name: Name of the model.
            save_path: Path to save the plot.
        """
        epochs = range(1, len(train_scores) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_scores, "b-", label="Training", linewidth=2)
        plt.plot(epochs, val_scores, "r-", label="Validation", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"Learning Curves - {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Learning curves saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        model_name: str = "Model",
        top_k: int = 20,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot feature importance.
        
        Args:
            feature_names: Names of features.
            importances: Feature importances.
            model_name: Name of the model.
            top_k: Number of top features to show.
            save_path: Path to save the plot.
        """
        # Get top k features
        top_indices = np.argsort(importances)[-top_k:]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_importances)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel("Importance")
        plt.title(f"Top {top_k} Feature Importances - {model_name}")
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_importances):
            plt.text(v + 0.001, i, f"{v:.4f}", va="center")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_ensemble_performance(
        self,
        baseline_scores: Dict[str, float],
        ensemble_scores: Dict[str, float],
        metric: str = "accuracy",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot ensemble vs baseline performance comparison.
        
        Args:
            baseline_scores: Baseline model scores.
            ensemble_scores: Ensemble model scores.
            metric: Metric being compared.
            save_path: Path to save the plot.
        """
        # Calculate improvements
        improvements = {}
        for ensemble_name, ensemble_score in ensemble_scores.items():
            best_baseline_score = max(baseline_scores.values())
            improvement = ((ensemble_score - best_baseline_score) / best_baseline_score) * 100
            improvements[ensemble_name] = improvement
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Score comparison
        all_scores = {**baseline_scores, **ensemble_scores}
        model_names = list(all_scores.keys())
        scores = list(all_scores.values())
        
        colors = []
        for name in model_names:
            if name in ensemble_scores:
                colors.append("orange")
            else:
                colors.append("skyblue")
        
        bars1 = ax1.bar(range(len(model_names)), scores, color=colors)
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha="right")
        ax1.set_ylabel(metric.title())
        ax1.set_title(f"{metric.title()} Comparison")
        
        # Add value labels
        for bar, score in zip(bars1, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{score:.3f}", ha="center", va="bottom")
        
        # Plot 2: Improvement percentages
        ensemble_names = list(improvements.keys())
        improvement_values = list(improvements.values())
        
        bars2 = ax2.bar(range(len(ensemble_names)), improvement_values, color="green")
        ax2.set_xticks(range(len(ensemble_names)))
        ax2.set_xticklabels(ensemble_names, rotation=45, ha="right")
        ax2.set_ylabel("Improvement (%)")
        ax2.set_title("Ensemble Improvements over Best Baseline")
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        
        # Add value labels
        for bar, improvement in zip(bars2, improvement_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{improvement:.1f}%", ha="center", va="bottom")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Ensemble performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        dataset: str = "test",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot correlation matrix of metrics across models.
        
        Args:
            results: Model results dictionary.
            metrics: List of metrics to correlate.
            dataset: Dataset to analyze.
            save_path: Path to save the plot.
        """
        # Prepare data matrix
        data_matrix = []
        model_names = []
        
        for model_name, model_results in results.items():
            if dataset in model_results:
                model_names.append(model_name)
                row = []
                for metric in metrics:
                    if metric in model_results[dataset]:
                        row.append(model_results[dataset][metric])
                    else:
                        row.append(0.0)
                data_matrix.append(row)
        
        if not data_matrix:
            logger.warning("No data available for correlation matrix")
            return
        
        # Create DataFrame
        df = pd.DataFrame(data_matrix, index=model_names, columns=metrics)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0,
                   square=True, fmt=".2f")
        plt.title("Metric Correlation Matrix")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Correlation matrix saved to {save_path}")
        
        plt.show()
    
    def create_summary_dashboard(
        self,
        leaderboard: pd.DataFrame,
        results: Dict[str, Dict[str, float]],
        config: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> None:
        """Create a comprehensive summary dashboard.
        
        Args:
            leaderboard: Model leaderboard.
            results: Model results.
            config: Experiment configuration.
            save_path: Path to save the dashboard.
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Plot 1: Top 5 models leaderboard
        ax1 = fig.add_subplot(gs[0, :2])
        top_5 = leaderboard.head(5)
        colors = ["orange" if any(ensemble_type in name.lower() for ensemble_type in 
                                 ["voting", "stacking", "bagging", "dynamic", "meta"]) 
                 else "skyblue" for name in top_5["model"]]
        
        bars = ax1.barh(range(len(top_5)), top_5.iloc[:, 1], color=colors)
        ax1.set_yticks(range(len(top_5)))
        ax1.set_yticklabels(top_5["model"])
        ax1.set_xlabel("Score")
        ax1.set_title("Top 5 Models")
        ax1.invert_yaxis()
        
        # Plot 2: Model type distribution
        ax2 = fig.add_subplot(gs[0, 2:])
        ensemble_count = sum(1 for name in leaderboard["model"] 
                           if any(ensemble_type in name.lower() for ensemble_type in 
                                 ["voting", "stacking", "bagging", "dynamic", "meta"]))
        baseline_count = len(leaderboard) - ensemble_count
        
        ax2.pie([baseline_count, ensemble_count], 
               labels=["Baseline", "Ensemble"], 
               colors=["skyblue", "orange"],
               autopct="%1.1f%%")
        ax2.set_title("Model Type Distribution")
        
        # Plot 3: Metric comparison (if multiple metrics available)
        ax3 = fig.add_subplot(gs[1, :])
        if len(leaderboard.columns) > 2:
            metrics = leaderboard.columns[1:]  # Exclude model name
            x = np.arange(len(metrics))
            width = 0.35
            
            # Get top 3 models
            top_3_models = leaderboard.head(3)
            
            for i, (_, model_row) in enumerate(top_3_models.iterrows()):
                model_name = model_row["model"]
                scores = model_row.iloc[1:].values
                
                color = "orange" if any(ensemble_type in model_name.lower() for ensemble_type in 
                                      ["voting", "stacking", "bagging", "dynamic", "meta"]) else "skyblue"
                
                ax3.plot(x, scores, marker="o", label=model_name, color=color, linewidth=2)
            
            ax3.set_xlabel("Metrics")
            ax3.set_ylabel("Score")
            ax3.set_title("Metric Comparison - Top 3 Models")
            ax3.set_xticks(x)
            ax3.set_xticklabels(metrics, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Experiment info
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis("off")
        
        # Create info text
        info_text = f"""
        Experiment Configuration:
        • Dataset: {config.get('dataset', {}).get('name', 'Unknown')}
        • Task Type: {config.get('task_type', 'Unknown')}
        • Total Models: {len(leaderboard)}
        • Baseline Models: {baseline_count}
        • Ensemble Models: {ensemble_count}
        • Best Model: {leaderboard.iloc[0]['model']}
        • Best Score: {leaderboard.iloc[0].iloc[1]:.4f}
        """
        
        ax4.text(0.1, 0.5, info_text, fontsize=12, verticalalignment="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
        
        plt.suptitle("Ensemble Learning Experiment Summary", fontsize=16, fontweight="bold")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Summary dashboard saved to {save_path}")
        
        plt.show()
