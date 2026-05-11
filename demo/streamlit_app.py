"""Streamlit demo application for ensemble learning."""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger
from omegaconf import OmegaConf

from src.data.loader import DataLoader
from src.models.baselines import BaselineModels
from src.models.ensembles import (
    BaggingEnsemble,
    DynamicSelectionEnsemble,
    MetaLearningEnsemble,
    StackingEnsemble,
    VotingEnsemble,
)
from src.train.pipeline import EnsemblePipeline
from src.viz.visualizer import EnsembleVisualizer


def load_config(config_path: str) -> Dict:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    if os.path.exists(config_path):
        return OmegaConf.load(config_path)
    else:
        return {}


def create_synthetic_data(task_type: str, n_samples: int, n_features: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic dataset.
    
    Args:
        task_type: Type of task ('classification' or 'regression').
        n_samples: Number of samples.
        n_features: Number of features.
        noise: Amount of noise.
        
    Returns:
        Tuple of (X, y).
    """
    from sklearn.datasets import make_classification, make_regression
    
    if task_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=2,
            n_redundant=0,
            n_informative=n_features,
            random_state=42,
            noise=noise,
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=42,
        )
    
    return X, y


def run_streamlit_demo() -> None:
    """Run the Streamlit demo application."""
    
    st.title("🤖 Ensemble Learning Methods Demo")
    st.markdown("""
    This demo showcases various ensemble learning methods including:
    - **Voting Ensembles**: Hard and soft voting
    - **Stacking**: Meta-learning approach
    - **Bagging**: Bootstrap aggregating
    - **Dynamic Selection**: Adaptive model selection
    - **Meta-Learning**: Advanced ensemble techniques
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Task type selection
    task_type = st.sidebar.selectbox(
        "Task Type",
        ["classification", "regression"],
        index=0
    )
    
    # Dataset selection
    dataset_option = st.sidebar.selectbox(
        "Dataset",
        ["breast_cancer", "digits", "iris", "wine", "synthetic"],
        index=0
    )
    
    # Synthetic dataset parameters
    if dataset_option == "synthetic":
        n_samples = st.sidebar.slider("Number of Samples", 100, 2000, 1000)
        n_features = st.sidebar.slider("Number of Features", 5, 50, 20)
        noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
    else:
        n_samples = 1000
        n_features = 20
        noise = 0.1
    
    # Ensemble methods selection
    st.sidebar.header("Ensemble Methods")
    enable_voting = st.sidebar.checkbox("Voting Ensemble", value=True)
    enable_stacking = st.sidebar.checkbox("Stacking Ensemble", value=True)
    enable_bagging = st.sidebar.checkbox("Bagging Ensemble", value=True)
    enable_dynamic = st.sidebar.checkbox("Dynamic Selection", value=True)
    enable_meta = st.sidebar.checkbox("Meta-Learning", value=True)
    
    # Main content
    if st.button("🚀 Run Experiment", type="primary"):
        
        with st.spinner("Running ensemble learning experiment..."):
            
            # Create configuration
            config = {
                "random_state": 42,
                "task_type": task_type,
                "dataset": {
                    "name": dataset_option if dataset_option != "synthetic" else f"synthetic_{task_type}",
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "noise": noise,
                },
                "data_split": {
                    "test_size": 0.2,
                    "val_size": 0.2,
                },
                "preprocessing": {
                    "scaling": "standard",
                },
                "ensembles": {
                    "voting": {"enabled": enable_voting},
                    "stacking": {"enabled": enable_stacking},
                    "bagging": {"enabled": enable_bagging},
                    "dynamic_selection": {"enabled": enable_dynamic},
                    "meta_learning": {"enabled": enable_meta},
                },
                "evaluation": {
                    "primary_metric": "accuracy" if task_type == "classification" else "r2",
                },
            }
            
            # Convert to OmegaConf
            config = OmegaConf.create(config)
            
            # Initialize pipeline
            pipeline = EnsemblePipeline(config)
            
            # Run experiment
            results = pipeline.run_experiment()
            
            # Display results
            st.success("✅ Experiment completed successfully!")
            
            # Results summary
            st.header("📊 Results Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Models", len(results["model_counts"]["total"]))
            
            with col2:
                st.metric("Baseline Models", len(results["model_counts"]["baseline"]))
            
            with col3:
                st.metric("Ensemble Models", len(results["model_counts"]["ensemble"]))
            
            with col4:
                primary_metric = config.get("evaluation", {}).get("primary_metric", "accuracy")
                st.metric(f"Best {primary_metric.title()}", f"{results['best_score']:.4f}")
            
            # Leaderboard
            st.header("🏆 Model Leaderboard")
            
            if results["leaderboard"] is not None and not results["leaderboard"].empty:
                st.dataframe(results["leaderboard"], use_container_width=True)
                
                # Visualization
                st.header("📈 Visualizations")
                
                # Create visualizer
                visualizer = EnsembleVisualizer()
                
                # Plot leaderboard
                fig = visualizer.plot_leaderboard(results["leaderboard"], primary_metric)
                st.pyplot(fig)
                
                # Metric comparison
                if task_type == "classification":
                    metrics = ["accuracy", "precision", "recall", "f1"]
                else:
                    metrics = ["r2", "mse", "rmse", "mae"]
                
                fig2 = visualizer.plot_metric_comparison(results["results"], metrics)
                st.pyplot(fig2)
                
                # Summary dashboard
                fig3 = visualizer.create_summary_dashboard(
                    results["leaderboard"],
                    results["results"],
                    results["config"]
                )
                st.pyplot(fig3)
            
            # Improvements
            if results["improvements"]:
                st.header("📈 Ensemble Improvements")
                
                improvements_df = pd.DataFrame([
                    {"Ensemble Method": method, "Improvement (%)": improvement}
                    for method, improvement in results["improvements"].items()
                ])
                
                st.dataframe(improvements_df, use_container_width=True)
                
                # Plot improvements
                import matplotlib.pyplot as plt
                fig4, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(improvements_df["Ensemble Method"], improvements_df["Improvement (%)"])
                ax.set_ylabel("Improvement (%)")
                ax.set_title("Ensemble Improvements over Best Baseline")
                ax.tick_params(axis="x", rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, improvements_df["Improvement (%)"]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f"{value:.1f}%", ha="center", va="bottom")
                
                st.pyplot(fig4)
            
            # Model details
            st.header("🔍 Model Details")
            
            for model_name, model_results in results["results"].items():
                with st.expander(f"📋 {model_name}"):
                    if "test" in model_results:
                        st.write("**Test Performance:**")
                        test_metrics = model_results["test"]
                        for metric, value in test_metrics.items():
                            st.write(f"- {metric.title()}: {value:.4f}")
                    
                    if "validation" in model_results:
                        st.write("**Validation Performance:**")
                        val_metrics = model_results["validation"]
                        for metric, value in val_metrics.items():
                            st.write(f"- {metric.title()}: {value:.4f}")
    
    # Information section
    st.header("ℹ️ About Ensemble Learning")
    
    st.markdown("""
    ### What is Ensemble Learning?
    
    Ensemble learning is a machine learning technique that combines multiple models to improve 
    predictive performance. The key idea is that by combining diverse models, we can reduce 
    overfitting, improve generalization, and achieve better results than any single model.
    
    ### Methods Implemented:
    
    **1. Voting Ensembles**
    - Hard Voting: Majority vote of predictions
    - Soft Voting: Weighted average of prediction probabilities
    
    **2. Stacking (Stacked Generalization)**
    - Uses a meta-learner to combine base model predictions
    - Often achieves superior performance through learned combinations
    
    **3. Bagging (Bootstrap Aggregating)**
    - Trains models on different bootstrap samples
    - Reduces variance and overfitting
    
    **4. Dynamic Selection**
    - Selects different models for different instances
    - Adapts to local performance patterns
    
    **5. Meta-Learning**
    - Learns how to combine models optimally
    - Uses meta-features to improve ensemble decisions
    
    ### Safety Notice
    This is a research and educational tool. Results should not be used for production 
    decisions without proper validation and expert review.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Author**: kryptologyst | **GitHub**: [https://github.com/kryptologyst](https://github.com/kryptologyst)
    
    This project is for research and educational purposes only.
    """)


if __name__ == "__main__":
    run_streamlit_demo()
