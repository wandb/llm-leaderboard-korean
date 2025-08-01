#!/usr/bin/env python
"""
MLOps Integration Examples for HRET

This file demonstrates how to integrate HRET with various MLOps platforms
and workflows, including MLflow, Weights & Biases, and custom monitoring systems.
"""

import os
import json
import time
from typing import Dict, Any, List
from pathlib import Path

# Import HRET
import sys
sys.path.append('..')
import hret


class ModelTrainingPipeline:
    """Example MLOps pipeline that integrates HRET for evaluation."""
    
    def __init__(self, experiment_name: str = "llm_training"):
        self.experiment_name = experiment_name
        self.checkpoints: List[str] = []
        self.evaluation_results: Dict[str, Any] = {}
    
    def train_model(self, epochs: int = 10):
        """Simulate model training with periodic evaluation."""
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            # Simulate training
            print(f"Training epoch {epoch}/{epochs}...")
            time.sleep(0.1)  # Simulate training time
            
            # Save checkpoint every 3 epochs
            if epoch % 3 == 0:
                checkpoint_name = f"checkpoint_epoch_{epoch}"
                self.checkpoints.append(checkpoint_name)
                
                # Evaluate checkpoint using HRET
                self.evaluate_checkpoint(checkpoint_name, epoch)
        
        print("Training completed!")
        self.generate_training_report()
    
    def evaluate_checkpoint(self, checkpoint_name: str, epoch: int):
        """Evaluate a model checkpoint using HRET."""
        
        def model_function(input_text: str) -> str:
            # Simulate model inference
            # In practice, you would load the actual checkpoint here
            return f"Epoch {epoch} model response: {input_text[:50]}..."
        
        # Configure HRET for this evaluation
        hret.configure(
            default_dataset="kmmlu",
            mlflow_tracking=True,
            wandb_tracking=True,
            output_dir=f"./results/{self.experiment_name}"
        )
        
        # Run evaluation with MLOps integration
        with hret.evaluation_context(
            run_name=f"{checkpoint_name}_evaluation",
            model_backend_name="huggingface"
        ) as ctx:
            # Add MLOps integrations
            ctx.log_to_mlflow(experiment_name=self.experiment_name)
            ctx.log_to_wandb(project_name="llm_evaluation")
            
            # Add custom metadata
            ctx.metrics_tracker.run_metadata.update({
                "checkpoint": checkpoint_name,
                "epoch": epoch,
                "experiment": self.experiment_name
            })
            
            # Run evaluation
            result = ctx.evaluate(model_function)
            
            # Store results
            self.evaluation_results[checkpoint_name] = {
                "epoch": epoch,
                "metrics": result.metrics,
                "timestamp": time.time()
            }
            
            print(f"Checkpoint {checkpoint_name} evaluated: {result.metrics}")
    
    def generate_training_report(self):
        """Generate a comprehensive training report."""
        report = {
            "experiment_name": self.experiment_name,
            "total_checkpoints": len(self.checkpoints),
            "evaluation_results": self.evaluation_results,
            "best_checkpoint": self.find_best_checkpoint(),
            "training_summary": self.get_training_summary()
        }
        
        # Save report
        report_path = Path(f"./results/{self.experiment_name}/training_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Training report saved to: {report_path}")
        return report
    
    def find_best_checkpoint(self) -> Dict[str, Any]:
        """Find the best performing checkpoint."""
        if not self.evaluation_results:
            return {}
        
        best_checkpoint = None
        best_score = -1
        
        for checkpoint, data in self.evaluation_results.items():
            # Use accuracy as the primary metric (adjust as needed)
            score = data["metrics"].get("accuracy", 0)
            if score > best_score:
                best_score = score
                best_checkpoint = checkpoint
        
        return {
            "checkpoint": best_checkpoint,
            "score": best_score,
            "epoch": self.evaluation_results[best_checkpoint]["epoch"] if best_checkpoint else None
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the training process."""
        if not self.evaluation_results:
            return {}
        
        scores = [data["metrics"].get("accuracy", 0) for data in self.evaluation_results.values()]
        
        return {
            "total_evaluations": len(scores),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "best_score": max(scores) if scores else 0,
            "worst_score": min(scores) if scores else 0,
            "improvement": scores[-1] - scores[0] if len(scores) > 1 else 0
        }


class HyperparameterTuning:
    """Example hyperparameter tuning with HRET evaluation."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def tune_hyperparameters(self):
        """Run hyperparameter tuning with evaluation."""
        
        # Define hyperparameter space
        hyperparams = [
            {"learning_rate": 0.001, "batch_size": 16, "dropout": 0.1},
            {"learning_rate": 0.01, "batch_size": 32, "dropout": 0.2},
            {"learning_rate": 0.005, "batch_size": 64, "dropout": 0.15},
        ]
        
        for i, params in enumerate(hyperparams):
            print(f"Testing hyperparameters {i+1}/{len(hyperparams)}: {params}")
            
            # Train model with these hyperparameters (simulated)
            model_function = self.create_model_with_params(params)
            
            # Evaluate using HRET
            result = self.evaluate_hyperparameters(params, model_function, run_id=i+1)
            self.results.append(result)
        
        # Find best hyperparameters
        best_result = max(self.results, key=lambda x: x["metrics"].get("accuracy", 0))
        print(f"Best hyperparameters: {best_result['hyperparameters']}")
        print(f"Best score: {best_result['metrics'].get('accuracy', 0)}")
        
        return best_result
    
    def create_model_with_params(self, params: Dict[str, Any]):
        """Create a model function with specific hyperparameters."""
        def model_function(input_text: str) -> str:
            # Simulate model with specific hyperparameters
            lr = params["learning_rate"]
            bs = params["batch_size"]
            dropout = params["dropout"]
            
            return f"Model(lr={lr}, bs={bs}, dropout={dropout}): {input_text[:30]}..."
        
        return model_function
    
    def evaluate_hyperparameters(self, params: Dict[str, Any], model_function, run_id: int):
        """Evaluate a model with specific hyperparameters."""
        
        with hret.evaluation_context(
            dataset_name="kmmlu",
            run_name=f"hyperparam_run_{run_id}"
        ) as ctx:
            # Add hyperparameters as metadata
            ctx.metrics_tracker.run_metadata.update({
                "hyperparameters": params,
                "run_id": run_id
            })
            
            # Add MLOps integration
            ctx.log_to_mlflow(experiment_name="hyperparameter_tuning")
            
            # Run evaluation
            result = ctx.evaluate(model_function)
            
            return {
                "run_id": run_id,
                "hyperparameters": params,
                "metrics": result.metrics,
                "timestamp": time.time()
            }


class ModelComparison:
    """Example model comparison across different architectures."""
    
    def compare_model_architectures(self):
        """Compare different model architectures."""
        
        # Define different model architectures
        models = {
            "transformer": self.create_transformer_model,
            "lstm": self.create_lstm_model,
            "cnn": self.create_cnn_model,
        }
        
        # Run comparison using HRET
        with hret.evaluation_context(
            dataset_name="kmmlu",
            run_name="architecture_comparison"
        ) as ctx:
            # Add MLOps integration
            ctx.log_to_mlflow(experiment_name="model_comparison")
            ctx.log_to_wandb(project_name="architecture_study")
            
            # Evaluate each architecture
            results = {}
            for arch_name, model_creator in models.items():
                print(f"Evaluating {arch_name} architecture...")
                
                model_function = model_creator()
                result = ctx.evaluate(model_function)
                results[arch_name] = result
                
                print(f"{arch_name} results: {result.metrics}")
            
            # Generate comparison report
            self.generate_comparison_report(results)
            
            return results
    
    def create_transformer_model(self):
        """Create a transformer-based model function."""
        def model_function(input_text: str) -> str:
            return f"Transformer model output: {input_text[:40]}..."
        return model_function
    
    def create_lstm_model(self):
        """Create an LSTM-based model function."""
        def model_function(input_text: str) -> str:
            return f"LSTM model output: {input_text[:40]}..."
        return model_function
    
    def create_cnn_model(self):
        """Create a CNN-based model function."""
        def model_function(input_text: str) -> str:
            return f"CNN model output: {input_text[:40]}..."
        return model_function
    
    def generate_comparison_report(self, results: Dict[str, Any]):
        """Generate a model comparison report."""
        report = {
            "comparison_timestamp": time.time(),
            "models_compared": list(results.keys()),
            "detailed_results": {}
        }
        
        for model_name, result in results.items():
            report["detailed_results"][model_name] = {
                "metrics": result.metrics,
                "sample_count": len(result.samples)
            }
        
        # Find best model
        best_model = max(results.keys(), 
                        key=lambda x: results[x].metrics.get("accuracy", 0))
        report["best_model"] = {
            "name": best_model,
            "score": results[best_model].metrics.get("accuracy", 0)
        }
        
        # Save report
        report_path = Path("./results/model_comparison_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Comparison report saved to: {report_path}")


class ContinuousEvaluation:
    """Example continuous evaluation system."""
    
    def __init__(self, model_endpoint: str):
        self.model_endpoint = model_endpoint
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def run_continuous_evaluation(self, interval_hours: int = 24):
        """Run continuous evaluation (simulated)."""
        print(f"Starting continuous evaluation every {interval_hours} hours...")
        
        # Simulate multiple evaluation runs
        for run_number in range(1, 4):  # Simulate 3 runs
            print(f"Running evaluation #{run_number}...")
            
            result = self.run_single_evaluation(run_number)
            self.evaluation_history.append(result)
            
            # Check for performance degradation
            self.check_performance_degradation()
            
            time.sleep(0.1)  # Simulate time between runs
        
        print("Continuous evaluation completed!")
    
    def run_single_evaluation(self, run_number: int):
        """Run a single evaluation."""
        
        def production_model(input_text: str) -> str:
            # Simulate calling production model endpoint
            return f"Production model run {run_number}: {input_text[:30]}..."
        
        with hret.evaluation_context(
            dataset_name="kmmlu",
            run_name=f"continuous_eval_run_{run_number}"
        ) as ctx:
            # Add production metadata
            ctx.metrics_tracker.run_metadata.update({
                "model_endpoint": self.model_endpoint,
                "run_number": run_number,
                "evaluation_type": "continuous"
            })
            
            # Add monitoring integration
            ctx.log_to_mlflow(experiment_name="continuous_evaluation")
            
            # Run evaluation
            result = ctx.evaluate(production_model)
            
            return {
                "run_number": run_number,
                "timestamp": time.time(),
                "metrics": result.metrics,
                "model_endpoint": self.model_endpoint
            }
    
    def check_performance_degradation(self):
        """Check for performance degradation."""
        if len(self.evaluation_history) < 2:
            return
        
        current_score = self.evaluation_history[-1]["metrics"].get("accuracy", 0)
        previous_score = self.evaluation_history[-2]["metrics"].get("accuracy", 0)
        
        degradation_threshold = 0.05  # 5% degradation threshold
        
        if current_score < previous_score - degradation_threshold:
            print(f"⚠️  Performance degradation detected!")
            print(f"Previous: {previous_score:.3f}, Current: {current_score:.3f}")
            
            # In practice, you would send alerts here
            self.send_alert(current_score, previous_score)
    
    def send_alert(self, current_score: float, previous_score: float):
        """Send performance degradation alert."""
        alert_message = f"""
        Performance Degradation Alert
        Model Endpoint: {self.model_endpoint}
        Previous Score: {previous_score:.3f}
        Current Score: {current_score:.3f}
        Degradation: {previous_score - current_score:.3f}
        """
        
        print(alert_message)
        # In practice, send to Slack, email, PagerDuty, etc.


def main():
    """Run all MLOps integration examples."""
    print("HRET MLOps Integration Examples")
    print("=" * 50)
    
    # Example 1: Model Training Pipeline
    print("\n1. Model Training Pipeline Example:")
    pipeline = ModelTrainingPipeline("llm_experiment_1")
    pipeline.train_model(epochs=9)
    
    # Example 2: Hyperparameter Tuning
    print("\n2. Hyperparameter Tuning Example:")
    tuner = HyperparameterTuning()
    best_params = tuner.tune_hyperparameters()
    
    # Example 3: Model Architecture Comparison
    print("\n3. Model Architecture Comparison Example:")
    comparator = ModelComparison()
    comparison_results = comparator.compare_model_architectures()
    
    # Example 4: Continuous Evaluation
    print("\n4. Continuous Evaluation Example:")
    continuous_eval = ContinuousEvaluation("https://api.example.com/model")
    continuous_eval.run_continuous_evaluation()
    
    print("\nAll MLOps integration examples completed!")


if __name__ == "__main__":
    main()