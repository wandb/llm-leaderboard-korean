#!/usr/bin/env python
"""
HRET (Haerae Evaluation Toolkit) Usage Examples

This file demonstrates various ways to use the HRET facade API
for MLOps-friendly LLM evaluation.
"""

import hret
from typing import Dict, Any


# Example 1: Simple decorator-based evaluation
@hret.evaluate(dataset="kmmlu", model="huggingface")
def my_simple_model(input_text: str) -> str:
    """A simple model function that just returns the input (dummy example)."""
    # In practice, this would call your actual model
    return f"Generated response for: {input_text}"


# Example 2: Benchmark multiple models
@hret.benchmark(dataset="kmmlu")
def compare_my_models():
    """Compare multiple model variants."""
    return {
        "baseline": lambda x: f"Baseline response: {x}",
        "improved": lambda x: f"Improved response: {x}",
        "experimental": lambda x: f"Experimental response: {x}"
    }


# Example 3: Track specific metrics
@hret.track_metrics(["accuracy", "latency"])
def custom_evaluation() -> Dict[str, Any]:
    """Custom evaluation with metric tracking."""
    # Your custom evaluation logic here
    return {
        "accuracy": 0.85,
        "latency": 120.5,
        "other_metric": "not_tracked"
    }


# Example 4: Context manager usage
def context_manager_example():
    """Example using context manager for more control."""
    
    def my_model(input_text: str) -> str:
        return f"Model output: {input_text}"
    
    # Basic context manager usage
    with hret.evaluation_context(dataset="kmmlu", run_name="my_experiment") as ctx:
        result = ctx.evaluate(my_model)
        print(f"Evaluation completed with accuracy: {result.metrics.get('accuracy', 'N/A')}")
    
    # Context manager with MLOps integration
    with hret.evaluation_context(dataset="kmmlu", run_name="mlops_experiment") as ctx:
        # Add MLOps integrations
        ctx.log_to_mlflow(experiment_name="hret_experiments")
        ctx.log_to_wandb(project_name="llm_evaluation")
        
        # Run evaluation
        result = ctx.evaluate(my_model)
        
        # Results are automatically logged to MLflow and W&B


# Example 5: Quick evaluation for simple cases
def quick_evaluation_example():
    """Example of quick evaluation function."""
    
    def my_model(input_text: str) -> str:
        return f"Quick model response: {input_text}"
    
    # One-liner evaluation
    result = hret.quick_eval(my_model, dataset="kmmlu")
    print(f"Quick eval result: {result.metrics}")


# Example 6: Model comparison
def model_comparison_example():
    """Example of comparing multiple models."""
    
    models = {
        "gpt_style": lambda x: f"GPT-style: {x}",
        "bert_style": lambda x: f"BERT-style: {x}",
        "custom_model": lambda x: f"Custom: {x}"
    }
    
    results = hret.compare_models(models, dataset="kmmlu")
    
    # Print comparison
    for model_name, result in results.items():
        print(f"{model_name}: {result.metrics}")


# Example 7: Configuration and advanced usage
def advanced_configuration_example():
    """Example of advanced configuration and usage."""
    
    # Configure global settings
    hret.configure(
        default_dataset="kmmlu",
        default_model="huggingface",
        mlflow_tracking=True,
        wandb_tracking=True,
        output_dir="./my_results",
        auto_save_results=True
    )
    
    # Load configuration from file
    # hret.load_config("config.yaml")
    
    def my_advanced_model(input_text: str) -> str:
        return f"Advanced model: {input_text}"
    
    # Use configured defaults
    with hret.evaluation_context(run_name="advanced_experiment") as ctx:
        result = ctx.evaluate(my_advanced_model)
        
        # Save results manually if needed
        filepath = ctx.save_results("custom_results.json")
        print(f"Results saved to: {filepath}")


# Example 8: Metrics tracking and comparison
def metrics_tracking_example():
    """Example of metrics tracking across multiple runs."""
    
    def model_v1(input_text: str) -> str:
        return f"V1: {input_text}"
    
    def model_v2(input_text: str) -> str:
        return f"V2: {input_text}"
    
    # Run multiple evaluations
    with hret.evaluation_context(run_name="model_v1") as ctx:
        ctx.evaluate(model_v1)
    
    with hret.evaluation_context(run_name="model_v2") as ctx:
        ctx.evaluate(model_v2)
    
    # Compare metrics across runs
    history = hret.get_metrics_history()
    print(f"Total runs: {len(history)}")
    
    # Compare specific metric
    accuracy_comparison = hret.compare_metric("accuracy")
    print(f"Accuracy comparison: {accuracy_comparison}")


if __name__ == "__main__":
    print("HRET Examples")
    print("=" * 50)
    
    print("\n3. Context manager example:")
    context_manager_example()
    
    print("\n4. Quick evaluation example:")
    quick_evaluation_example()
    
    print("\n5. Model comparison example:")
    model_comparison_example()
    
    print("\n6. Advanced configuration example:")
    advanced_configuration_example()
    
    print("\n7. Metrics tracking example:")
    metrics_tracking_example()
    
    print("\nAll examples completed!")