"""Command-line interface for ensemble learning experiments."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.train.pipeline import EnsemblePipeline
from src.viz.visualizer import EnsembleVisualizer


def setup_logging(config: DictConfig) -> None:
    """Setup logging configuration.
    
    Args:
        config: Configuration object.
    """
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    log_format = log_config.get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    log_file = log_config.get("file", "logs/experiment.log")
    
    # Create logs directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level=log_level, format=log_format)
    logger.add(log_file, level=log_level, format=log_format, rotation="10 MB")
    
    logger.info("Logging configured")


def train_command(args) -> None:
    """Train ensemble models.
    
    Args:
        args: Command line arguments.
    """
    logger.info("Starting training...")
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup logging
    setup_logging(config)
    
    # Create output directories
    output_config = config.get("output", {})
    if output_config.get("save_results", True):
        results_dir = output_config.get("results_dir", "results/")
        os.makedirs(results_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = EnsemblePipeline(config)
    
    # Run experiment
    results = pipeline.run_experiment()
    
    # Save results
    if output_config.get("save_results", True):
        results_path = os.path.join(results_dir, "experiment_results.pkl")
        pipeline.save_results(results_path)
    
    # Print summary
    pipeline.print_summary()
    
    logger.info("Training completed successfully")


def eval_command(args) -> None:
    """Evaluate trained models.
    
    Args:
        args: Command line arguments.
    """
    logger.info("Starting evaluation...")
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup logging
    setup_logging(config)
    
    # Load results
    if not os.path.exists(args.results):
        logger.error(f"Results file not found: {args.results}")
        return
    
    import pickle
    with open(args.results, "rb") as f:
        saved_results = pickle.load(f)
    
    # Create visualizations
    visualizer = EnsembleVisualizer()
    
    # Plot leaderboard
    if "leaderboard" in saved_results and not saved_results["leaderboard"].empty:
        visualizer.plot_leaderboard(
            saved_results["leaderboard"],
            metric=config.get("evaluation", {}).get("primary_metric", "accuracy"),
            save_path=args.output_dir + "/leaderboard.png" if args.output_dir else None
        )
    
    # Plot metric comparison
    if "results" in saved_results:
        metrics = config.get("evaluation", {}).get("metrics", {}).get(config.get("task_type", "classification"), [])
        if metrics:
            visualizer.plot_metric_comparison(
                saved_results["results"],
                metrics,
                save_path=args.output_dir + "/metric_comparison.png" if args.output_dir else None
            )
    
    # Create summary dashboard
    visualizer.create_summary_dashboard(
        saved_results["leaderboard"],
        saved_results["results"],
        saved_results["config"],
        save_path=args.output_dir + "/summary_dashboard.png" if args.output_dir else None
    )
    
    logger.info("Evaluation completed successfully")


def demo_command(args) -> None:
    """Run interactive demo.
    
    Args:
        args: Command line arguments.
    """
    logger.info("Starting demo...")
    
    try:
        import streamlit as st
        from demo.streamlit_app import run_streamlit_demo
        
        # Set Streamlit config
        st.set_page_config(
            page_title="Ensemble Learning Demo",
            page_icon="🤖",
            layout="wide"
        )
        
        # Run demo
        run_streamlit_demo()
        
    except ImportError:
        logger.error("Streamlit not installed. Please install with: pip install streamlit")
        logger.info("Falling back to Gradio demo...")
        
        try:
            import gradio as gr
            from demo.gradio_app import run_gradio_demo
            
            # Run Gradio demo
            run_gradio_demo()
            
        except ImportError:
            logger.error("Neither Streamlit nor Gradio installed.")
            logger.info("Please install one of them:")
            logger.info("  pip install streamlit")
            logger.info("  pip install gradio")


def create_default_config(output_path: str) -> None:
    """Create a default configuration file.
    
    Args:
        output_path: Path to save the configuration.
    """
    default_config = {
        "random_state": 42,
        "task_type": "classification",
        "dataset": {
            "name": "breast_cancer",
            "n_samples": 1000,
            "n_features": 20,
            "n_classes": 2,
            "noise": 0.1
        },
        "data_split": {
            "test_size": 0.2,
            "val_size": 0.2
        },
        "preprocessing": {
            "scaling": "standard"
        },
        "baselines": {
            "enabled": True,
            "models": [
                "logistic_regression",
                "decision_tree",
                "knn",
                "svm",
                "random_forest",
                "extra_trees",
                "gradient_boosting",
                "adaboost"
            ]
        },
        "ensembles": {
            "voting": {
                "enabled": True,
                "voting_type": "hard",
                "weights": None
            },
            "stacking": {
                "enabled": True,
                "cv": 5,
                "meta_estimator": None
            },
            "bagging": {
                "enabled": True,
                "n_estimators": 10,
                "max_samples": 1.0,
                "max_features": 1.0
            },
            "dynamic_selection": {
                "enabled": True,
                "selection_method": "best",
                "k": 5
            },
            "meta_learning": {
                "enabled": True,
                "feature_extraction": "predictions"
            }
        },
        "evaluation": {
            "primary_metric": "accuracy",
            "metrics": {
                "classification": ["accuracy", "precision", "recall", "f1", "roc_auc"],
                "regression": ["r2", "mse", "rmse", "mae", "mape", "smape"]
            }
        },
        "logging": {
            "level": "INFO",
            "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            "file": "logs/experiment.log"
        },
        "output": {
            "save_results": True,
            "results_dir": "results/",
            "save_models": False,
            "models_dir": "models/"
        }
    }
    
    with open(output_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Default configuration created at {output_path}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ensemble Learning Methods - Research and Education Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models with default configuration
  python -m src.cli train

  # Train with custom configuration
  python -m src.cli train --config configs/custom.yaml

  # Evaluate results
  python -m src.cli eval --results results/experiment_results.pkl

  # Run interactive demo
  python -m src.cli demo

  # Create default configuration
  python -m src.cli create-config --output configs/my_config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train ensemble models")
    train_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained models")
    eval_parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results file"
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    eval_parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save evaluation plots"
    )
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    
    # Create config command
    config_parser = subparsers.add_parser("create-config", help="Create default configuration")
    config_parser.add_argument(
        "--output",
        type=str,
        default="config.yaml",
        help="Output path for configuration file"
    )
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_command(args)
    elif args.command == "eval":
        eval_command(args)
    elif args.command == "demo":
        demo_command(args)
    elif args.command == "create-config":
        create_default_config(args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
