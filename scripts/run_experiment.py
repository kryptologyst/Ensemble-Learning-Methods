"""Example scripts for ensemble learning experiments."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omegaconf import OmegaConf
from src.train.pipeline import EnsemblePipeline
from src.viz.visualizer import EnsembleVisualizer


def run_classification_experiment():
    """Run a classification experiment."""
    print("Running Classification Experiment...")
    
    # Create configuration
    config = {
        "random_state": 42,
        "task_type": "classification",
        "dataset": {
            "name": "breast_cancer",
        },
        "data_split": {
            "test_size": 0.2,
            "val_size": 0.2,
        },
        "preprocessing": {
            "scaling": "standard",
        },
        "ensembles": {
            "voting": {"enabled": True, "voting_type": "hard"},
            "stacking": {"enabled": True, "cv": 5},
            "bagging": {"enabled": True, "n_estimators": 10},
            "dynamic_selection": {"enabled": True},
            "meta_learning": {"enabled": True},
        },
        "evaluation": {
            "primary_metric": "accuracy",
        },
    }
    
    config = OmegaConf.create(config)
    
    # Run experiment
    pipeline = EnsemblePipeline(config)
    results = pipeline.run_experiment()
    
    # Print results
    pipeline.print_summary()
    
    return results


def run_regression_experiment():
    """Run a regression experiment."""
    print("Running Regression Experiment...")
    
    # Create configuration
    config = {
        "random_state": 42,
        "task_type": "regression",
        "dataset": {
            "name": "synthetic_regression",
            "n_samples": 1000,
            "n_features": 20,
            "noise": 0.1,
        },
        "data_split": {
            "test_size": 0.2,
            "val_size": 0.2,
        },
        "preprocessing": {
            "scaling": "standard",
        },
        "ensembles": {
            "voting": {"enabled": True},
            "stacking": {"enabled": True, "cv": 5},
            "bagging": {"enabled": True, "n_estimators": 10},
            "dynamic_selection": {"enabled": True},
            "meta_learning": {"enabled": True},
        },
        "evaluation": {
            "primary_metric": "r2",
        },
    }
    
    config = OmegaConf.create(config)
    
    # Run experiment
    pipeline = EnsemblePipeline(config)
    results = pipeline.run_experiment()
    
    # Print results
    pipeline.print_summary()
    
    return results


def run_comparison_experiment():
    """Run a comparison experiment between different ensemble methods."""
    print("Running Comparison Experiment...")
    
    # Test different ensemble configurations
    ensemble_configs = [
        {"name": "voting_only", "ensembles": {"voting": {"enabled": True}}},
        {"name": "stacking_only", "ensembles": {"stacking": {"enabled": True}}},
        {"name": "bagging_only", "ensembles": {"bagging": {"enabled": True}}},
        {"name": "all_ensembles", "ensembles": {
            "voting": {"enabled": True},
            "stacking": {"enabled": True},
            "bagging": {"enabled": True},
            "dynamic_selection": {"enabled": True},
            "meta_learning": {"enabled": True},
        }},
    ]
    
    results_comparison = {}
    
    for config_info in ensemble_configs:
        print(f"\nTesting {config_info['name']}...")
        
        config = {
            "random_state": 42,
            "task_type": "classification",
            "dataset": {"name": "breast_cancer"},
            "data_split": {"test_size": 0.2, "val_size": 0.2},
            "preprocessing": {"scaling": "standard"},
            "evaluation": {"primary_metric": "accuracy"},
            **config_info["ensembles"]
        }
        
        config = OmegaConf.create(config)
        
        # Run experiment
        pipeline = EnsemblePipeline(config)
        results = pipeline.run_experiment()
        
        results_comparison[config_info["name"]] = results
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    for name, results in results_comparison.items():
        print(f"\n{name.upper()}:")
        print(f"  Best Model: {results['best_model']}")
        print(f"  Best Score: {results['best_score']:.4f}")
        print(f"  Total Models: {results['model_counts']['total']}")
        print(f"  Ensemble Models: {results['model_counts']['ensemble']}")
    
    return results_comparison


def create_visualizations(results):
    """Create visualizations for experiment results."""
    print("Creating visualizations...")
    
    # Create output directory
    os.makedirs("assets", exist_ok=True)
    
    # Create visualizer
    visualizer = EnsembleVisualizer()
    
    # Plot leaderboard
    if results["leaderboard"] is not None and not results["leaderboard"].empty:
        visualizer.plot_leaderboard(
            results["leaderboard"],
            metric=results["config"].get("evaluation", {}).get("primary_metric", "accuracy"),
            save_path="assets/leaderboard.png"
        )
    
    # Plot metric comparison
    task_type = results["config"].get("task_type", "classification")
    if task_type == "classification":
        metrics = ["accuracy", "precision", "recall", "f1"]
    else:
        metrics = ["r2", "mse", "rmse", "mae"]
    
    visualizer.plot_metric_comparison(
        results["results"],
        metrics,
        save_path="assets/metric_comparison.png"
    )
    
    # Create summary dashboard
    visualizer.create_summary_dashboard(
        results["leaderboard"],
        results["results"],
        results["config"],
        save_path="assets/summary_dashboard.png"
    )
    
    print("Visualizations saved to assets/ directory")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ensemble learning experiments")
    parser.add_argument("--experiment", choices=["classification", "regression", "comparison"], 
                       default="classification", help="Type of experiment to run")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    
    args = parser.parse_args()
    
    if args.experiment == "classification":
        results = run_classification_experiment()
    elif args.experiment == "regression":
        results = run_regression_experiment()
    elif args.experiment == "comparison":
        results = run_comparison_experiment()
    
    if args.visualize and args.experiment != "comparison":
        create_visualizations(results)
    
    print("\nExperiment completed successfully!")
