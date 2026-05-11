"""Gradio demo application for ensemble learning."""

import os
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from src.train.pipeline import EnsemblePipeline
from src.viz.visualizer import EnsembleVisualizer


def run_gradio_demo() -> None:
    """Run the Gradio demo application."""
    
    def run_experiment(
        task_type: str,
        dataset: str,
        n_samples: int,
        n_features: int,
        noise: float,
        enable_voting: bool,
        enable_stacking: bool,
        enable_bagging: bool,
        enable_dynamic: bool,
        enable_meta: bool,
    ) -> Tuple[str, str, str]:
        """Run ensemble learning experiment.
        
        Args:
            task_type: Type of task.
            dataset: Dataset name.
            n_samples: Number of samples.
            n_features: Number of features.
            noise: Noise level.
            enable_voting: Enable voting ensemble.
            enable_stacking: Enable stacking ensemble.
            enable_bagging: Enable bagging ensemble.
            enable_dynamic: Enable dynamic selection.
            enable_meta: Enable meta-learning.
            
        Returns:
            Tuple of (summary, leaderboard, improvements).
        """
        try:
            # Create configuration
            config = {
                "random_state": 42,
                "task_type": task_type,
                "dataset": {
                    "name": dataset if dataset != "synthetic" else f"synthetic_{task_type}",
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
            
            # Create summary
            summary = f"""
# Experiment Results Summary

**Dataset**: {dataset}
**Task Type**: {task_type}
**Total Models**: {results['model_counts']['total']}
**Baseline Models**: {results['model_counts']['baseline']}
**Ensemble Models**: {results['model_counts']['ensemble']}
**Best Model**: {results['best_model']}
**Best Score**: {results['best_score']:.4f}
**Execution Time**: {results['execution_time']:.2f} seconds
            """
            
            # Create leaderboard
            if results["leaderboard"] is not None and not results["leaderboard"].empty:
                leaderboard_text = results["leaderboard"].to_string(index=False)
            else:
                leaderboard_text = "No leaderboard available"
            
            # Create improvements
            if results["improvements"]:
                improvements_text = "## Ensemble Improvements\n\n"
                for method, improvement in results["improvements"].items():
                    improvements_text += f"- **{method}**: {improvement:.2f}%\n"
            else:
                improvements_text = "No improvements calculated"
            
            return summary, leaderboard_text, improvements_text
            
        except Exception as e:
            error_msg = f"Error running experiment: {str(e)}"
            logger.error(error_msg)
            return error_msg, "", ""
    
    # Create Gradio interface
    with gr.Blocks(title="Ensemble Learning Demo") as demo:
        gr.Markdown("""
        # 🤖 Ensemble Learning Methods Demo
        
        This demo showcases various ensemble learning methods including Voting, Stacking, 
        Bagging, Dynamic Selection, and Meta-Learning approaches.
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Configuration")
                
                task_type = gr.Dropdown(
                    choices=["classification", "regression"],
                    value="classification",
                    label="Task Type"
                )
                
                dataset = gr.Dropdown(
                    choices=["breast_cancer", "digits", "iris", "wine", "synthetic"],
                    value="breast_cancer",
                    label="Dataset"
                )
                
                n_samples = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=1000,
                    step=100,
                    label="Number of Samples (for synthetic)"
                )
                
                n_features = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=1,
                    label="Number of Features (for synthetic)"
                )
                
                noise = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.05,
                    label="Noise Level (for synthetic)"
                )
                
                gr.Markdown("### Ensemble Methods")
                
                enable_voting = gr.Checkbox(value=True, label="Voting Ensemble")
                enable_stacking = gr.Checkbox(value=True, label="Stacking Ensemble")
                enable_bagging = gr.Checkbox(value=True, label="Bagging Ensemble")
                enable_dynamic = gr.Checkbox(value=True, label="Dynamic Selection")
                enable_meta = gr.Checkbox(value=True, label="Meta-Learning")
                
                run_button = gr.Button("🚀 Run Experiment", variant="primary")
            
            with gr.Column():
                gr.Markdown("### Results")
                
                summary_output = gr.Markdown(label="Summary")
                leaderboard_output = gr.Textbox(label="Leaderboard", lines=10)
                improvements_output = gr.Markdown(label="Improvements")
        
        # Event handlers
        run_button.click(
            fn=run_experiment,
            inputs=[
                task_type,
                dataset,
                n_samples,
                n_features,
                noise,
                enable_voting,
                enable_stacking,
                enable_bagging,
                enable_dynamic,
                enable_meta,
            ],
            outputs=[summary_output, leaderboard_output, improvements_output]
        )
        
        # Information section
        gr.Markdown("""
        ---
        
        ## ℹ️ About Ensemble Learning
        
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
        
        ---
        
        **Author**: kryptologyst | **GitHub**: [https://github.com/kryptologyst](https://github.com/kryptologyst)
        
        This project is for research and educational purposes only.
        """)
    
    # Launch the demo
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )


if __name__ == "__main__":
    run_gradio_demo()
