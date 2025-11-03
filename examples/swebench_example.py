"""
SWE-bench Integration Example

This example demonstrates how to use the integrated SWE-bench dataset and evaluator
within the HRET framework.
"""

from llm_eval.evaluator import Evaluator

# Example 1: Basic usage with Evaluator API
def example_basic():
    """Basic SWE-bench evaluation using the Evaluator API"""

    evaluator = Evaluator()

    results = evaluator.run(
        # Dataset configuration
        dataset="swebench",
        split="test",
        dataset_params={
            "artifacts_path": "horangi/horangi4-dataset/swebench_verified_official_80:v4",
            "dataset_dir": ".",
            "max_samples": 10,  # Limit for testing
        },

        # Model configuration
        model="litellm",  # or "openai", "huggingface", etc.
        model_params={
            "model_name_or_path": "gpt-4o-2024-11-20",
            "temperature": 0.0,
            "max_tokens": 16000,
        },

        # Evaluation configuration
        evaluation_method="swebench",
        evaluator_params={
            "api_endpoint": "https://api.nejumi-swebench.org/",
            "api_key": None,  # Optional, uses SWE_API_KEY env var if not provided
            "namespace": "swebench",
            "tag": "latest",
            "timeout_sec": 1800,
            "concurrency": 2,
        },
    )

    print("Results:")
    print(f"  Resolved: {results['metrics']['resolved']}")
    print(f"  Resolved Rate: {results['metrics']['resolved_rate']:.2%}")
    print(f"  Total Samples: {results['metrics']['total_samples']}")


# Example 2: Using HRET decorator API
def example_hret_api():
    """Using HRET's decorator-based API for SWE-bench"""

    import llm_eval.hret as hret

    # Configure HRET
    hret.load_config({
        "default_dataset": "swebench",
        "default_model": "litellm",
        "mlflow_tracking": True,
        "wandb_tracking": True,
    })

    # Use context manager for evaluation
    with hret.evaluation_context(
        dataset="swebench",
        dataset_params={
            "artifacts_path": "horangi/horangi4-dataset/swebench_verified_official_80:v4",
            "max_samples": 10,
        },
        evaluation_method="swebench",
        evaluator_params={
            "api_endpoint": "https://api.nejumi-swebench.org/",
            "concurrency": 2,
        }
    ) as ctx:
        # Your model function
        def my_model_function(input_text):
            # Your model inference logic here
            # This is just a placeholder
            return "<patch>\nExample patch content\n</patch>"

        # Run evaluation
        result = ctx.evaluate(my_model_function)

        # Log to MLOps platforms
        ctx.log_to_mlflow(experiment_name="swebench_experiments")
        ctx.log_to_wandb(project_name="swebench_eval")

        print(f"Resolved rate: {result.metrics.get('resolved_rate', 0):.2%}")


# Example 3: Config-based evaluation
def example_config_based():
    """Using a YAML config file for SWE-bench evaluation"""

    from llm_eval.evaluator import run_from_config

    # Create a config file (swebench_config.yaml):
    """
    dataset:
      name: swebench
      split: test
      params:
        artifacts_path: "horangi/horangi4-dataset/swebench_verified_official_80:v4"
        dataset_dir: "."
        max_samples: 10

    model:
      name: litellm
      params:
        model_name_or_path: "gpt-4o-2024-11-20"
        temperature: 0.0
        max_tokens: 16000

    evaluation:
      method: swebench
      params:
        api_endpoint: "https://api.nejumi-swebench.org/"
        concurrency: 2
        timeout_sec: 1800
    """

    # Run from config
    result = run_from_config("swebench_config.yaml")

    print(f"Results: {result['metrics']}")


# Example 4: Advanced - Using with custom preprocessing
def example_advanced():
    """Advanced usage with custom preprocessing"""

    from llm_eval.datasets import load_datasets
    from llm_eval.models import get_model
    from llm_eval.evaluation import get_evaluator

    # Load dataset
    dataset = load_datasets(
        name="swebench",
        split="test",
        artifacts_path="horangi/horangi4-dataset/swebench_verified_official_80:v4",
        max_samples=5,
    )
    samples = dataset.load()

    print(f"Loaded {len(samples)} samples")

    # Load model
    model = get_model(
        "litellm",
        model_name_or_path="gpt-4o-2024-11-20",
        temperature=0.0,
    )

    # Generate predictions
    predictions = model.generate_batch(samples)

    # Custom post-processing on predictions if needed
    for pred in predictions:
        # Add any custom metadata
        pred["model_name"] = "gpt-4o-2024-11-20"
        # You can also modify the prediction here
        # pred["prediction"] = custom_postprocess(pred["prediction"])

    # Evaluate
    evaluator = get_evaluator(
        "swebench",
        api_endpoint="https://api.nejumi-swebench.org/",
        concurrency=2,
    )

    result = evaluator.evaluate(
        data=predictions,
        model=model,
        subsets=None
    )

    print(f"Metrics: {result['metrics']}")

    # Access individual sample results
    for sample in result['samples']:
        print(f"  {sample['instance_id']}: {sample.get('patch_valid', 'N/A')}")


if __name__ == "__main__":
    print("=" * 60)
    print("SWE-bench Integration Examples")
    print("=" * 60)

    # Run the basic example
    print("\n>>> Example 1: Basic Usage")
    try:
        example_basic()
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("\nFor other examples, uncomment and run:")
    print("  - example_hret_api()")
    print("  - example_config_based()")
    print("  - example_advanced()")
