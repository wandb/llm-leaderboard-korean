# HRET API Guide - MLOps-Friendly Evaluation Interface

HRET (Haerae Evaluation Toolkit) provides a deepeval-style decorator-based API designed for seamless integration with MLOps pipelines. This high-level interface enables LLM evaluation with minimal boilerplate code while maintaining full compatibility with the existing PipelineRunner.

## 🚀 Quick Start

```python
import llm_eval.hret as hret

# Simple decorator-based evaluation
@hret.evaluate(dataset="kmmlu", model="huggingface")
def my_model(input_text: str) -> str:
    return model.generate(input_text)

# Run evaluation
result = my_model()
print(f"Accuracy: {result.metrics['accuracy']}")
```

## 📋 Core Features

### 1. Decorator-Based API

#### `@hret.evaluate()`
Evaluate a single model function with minimal setup.

```python
@hret.evaluate(
    dataset="kmmlu", 
    model="huggingface", 
    evaluation_method="string_match"
)
def my_model(input_text: str) -> str:
    return model.generate(input_text)

result = my_model()  # Returns EvaluationResult
```

**Parameters:**
- `dataset`: Dataset name (e.g., "kmmlu", "haerae", "benchhub")
- `model`: Model backend name (e.g., "huggingface", "openai")
- `evaluation_method`: Evaluation method (e.g., "string_match", "llm_judge")
- `**kwargs`: Additional configuration parameters

#### `@hret.benchmark()`
Compare multiple models on the same dataset.

```python
@hret.benchmark(dataset="kmmlu")
def compare_models():
    return {
        "gpt-4": lambda x: gpt4_model.generate(x),
        "claude-3": lambda x: claude_model.generate(x),
        "custom": lambda x: custom_model.generate(x)
    }

results = compare_models()  # Returns Dict[str, EvaluationResult]
```

#### `@hret.track_metrics()`
Track specific metrics from evaluation functions.

```python
@hret.track_metrics(["accuracy", "latency", "cost"])
def custom_evaluation():
    # Your evaluation logic
    return {
        "accuracy": 0.85,
        "latency": 120.5,
        "cost": 0.02,
        "other_metric": "not_tracked"
    }
```

### 2. Context Manager Interface

For more granular control over the evaluation process:

```python
with hret.evaluation_context(
    dataset="kmmlu", 
    run_name="my_experiment"
) as ctx:
    # Configure MLOps integrations
    ctx.log_to_mlflow(experiment_name="llm_experiments")
    ctx.log_to_wandb(project_name="model_evaluation")
    
    # Run evaluation
    result = ctx.evaluate(my_model_function)
    
    # Save results
    ctx.save_results("experiment_results.json")
```

### 3. Convenience Functions

#### Quick Evaluation
```python
result = hret.quick_eval(my_model_function, dataset="kmmlu")
```

#### Model Comparison
```python
models = {
    "baseline": lambda x: baseline_model.generate(x),
    "improved": lambda x: improved_model.generate(x)
}
results = hret.compare_models(models, dataset="kmmlu")
```

## 🔧 Configuration Management

### Global Configuration

```python
hret.configure(
    default_dataset="kmmlu",
    default_model="huggingface",
    mlflow_tracking=True,
    wandb_tracking=True,
    output_dir="./results",
    auto_save_results=True,
    log_level="INFO"
)
```

### Configuration Files

Create a `hret_config.yaml` file:

```yaml
# HRET Configuration
default_dataset: "kmmlu"
default_model: "huggingface"
default_split: "test"
default_evaluation_method: "string_match"

# MLOps Integration
mlflow_tracking: true
wandb_tracking: false
tensorboard_tracking: false

# Output Settings
output_dir: "./hret_results"
auto_save_results: true
log_level: "INFO"

# Performance Settings
batch_size: 32
max_workers: 4
```

Load configuration:
```python
hret.load_config("hret_config.yaml")
```

## 🔗 MLOps Integration

### MLflow Integration

```python
with hret.evaluation_context(dataset="kmmlu") as ctx:
    ctx.log_to_mlflow(experiment_name="my_experiments")
    result = ctx.evaluate(my_model)
    # Results automatically logged to MLflow
```

### Weights & Biases Integration

```python
with hret.evaluation_context(dataset="kmmlu") as ctx:
    ctx.log_to_wandb(project_name="llm_evaluation")
    result = ctx.evaluate(my_model)
    # Results automatically logged to W&B
```

### Custom Integration

```python
def custom_logger(run_result, results):
    # Send to your monitoring system
    send_to_monitoring_system(run_result, results)

with hret.evaluation_context(dataset="kmmlu") as ctx:
    ctx.add_mlops_integration(custom_logger)
    result = ctx.evaluate(my_model)
```

## 📊 Metrics Tracking and Analysis

### Cross-Run Metrics Tracking

```python
# View all evaluation history
history = hret.get_metrics_history()
print(f"Total runs: {len(history)}")

# Compare specific metrics across runs
accuracy_comparison = hret.compare_metric("accuracy")
print(f"Best accuracy: {accuracy_comparison['_stats']['best']}")
print(f"Average accuracy: {accuracy_comparison['_stats']['average']}")
```

### Advanced Metrics Tracking

```python
tracker = hret.MetricsTracker()

# Run multiple experiments
for model_name, model_func in models.items():
    tracker.start_run(run_name=f"eval_{model_name}")
    
    result = hret.quick_eval(model_func, dataset="kmmlu")
    tracker.log_metrics(result.metrics)
    
    tracker.end_run()

# Compare results
comparison = tracker.compare_runs("accuracy")
```

## 🏭 MLOps Pipeline Integration

### Training Pipeline Integration

```python
class ModelTrainingPipeline:
    def train_and_evaluate(self, epochs=10):
        for epoch in range(1, epochs + 1):
            # Train model
            self.train_epoch(epoch)
            
            # Evaluate checkpoint
            if epoch % 3 == 0:
                self.evaluate_checkpoint(epoch)
    
    def evaluate_checkpoint(self, epoch):
        @hret.evaluate(dataset="kmmlu", model="huggingface")
        def checkpoint_model(input_text):
            return self.model.generate(input_text)
        
        with hret.evaluation_context(
            run_name=f"checkpoint_epoch_{epoch}"
        ) as ctx:
            ctx.log_to_mlflow(experiment_name="training")
            result = ctx.evaluate(checkpoint_model)
            
            # Performance monitoring
            if self.detect_degradation(result):
                self.send_alert(epoch, result)
```

### Hyperparameter Tuning

```python
def hyperparameter_tuning():
    hyperparams = [
        {"lr": 0.001, "batch_size": 16, "dropout": 0.1},
        {"lr": 0.01, "batch_size": 32, "dropout": 0.2},
        {"lr": 0.005, "batch_size": 64, "dropout": 0.15},
    ]
    
    best_result = None
    best_score = 0
    
    for i, params in enumerate(hyperparams):
        model_func = create_model_with_params(params)
        
        with hret.evaluation_context(
            run_name=f"hyperparam_run_{i}"
        ) as ctx:
            # Add hyperparameters as metadata
            ctx.metrics_tracker.run_metadata.update({
                "hyperparameters": params
            })
            ctx.log_to_mlflow(experiment_name="hyperparameter_tuning")
            
            result = ctx.evaluate(model_func)
            
            if result.metrics["accuracy"] > best_score:
                best_score = result.metrics["accuracy"]
                best_result = (params, result)
    
    return best_result
```

### Continuous Evaluation

```python
class ContinuousEvaluation:
    def run_continuous_evaluation(self):
        def production_model(input_text):
            return call_production_api(input_text)
        
        with hret.evaluation_context(
            run_name=f"continuous_eval_{int(time.time())}"
        ) as ctx:
            ctx.log_to_mlflow(experiment_name="continuous_evaluation")
            
            result = ctx.evaluate(production_model)
            
            # Performance monitoring
            if self.detect_degradation(result):
                self.send_performance_alert(result)
```

## 🎯 Advanced Usage Patterns

### Multi-Dataset Evaluation

```python
datasets = ["kmmlu", "haerae", "benchhub"]
results = {}

for dataset in datasets:
    with hret.evaluation_context(dataset_name=dataset) as ctx:
        ctx.log_to_mlflow(experiment_name="multi_dataset_eval")
        results[dataset] = ctx.evaluate(my_model)

# Compare performance across datasets
for dataset, result in results.items():
    print(f"{dataset}: {result.metrics['accuracy']:.3f}")
```

### A/B Testing

```python
@hret.benchmark(dataset="kmmlu")
def ab_test():
    return {
        "model_a": lambda x: model_a.generate(x),
        "model_b": lambda x: model_b.generate(x)
    }

results = ab_test()

# Statistical significance testing
from scipy import stats
scores_a = [s['accuracy'] for s in results['model_a'].samples]
scores_b = [s['accuracy'] for s in results['model_b'].samples]
t_stat, p_value = stats.ttest_ind(scores_a, scores_b)

print(f"A/B Test Results:")
print(f"Model A: {results['model_a'].metrics['accuracy']:.3f}")
print(f"Model B: {results['model_b'].metrics['accuracy']:.3f}")
print(f"P-value: {p_value:.4f}")
```

### Custom Evaluation Metrics

```python
def custom_evaluator(predictions, references):
    # Implement your custom evaluation logic
    custom_scores = []
    for pred, ref in zip(predictions, references):
        score = calculate_custom_metric(pred, ref)
        custom_scores.append(score)
    
    return {
        "custom_metric": sum(custom_scores) / len(custom_scores),
        "custom_std": np.std(custom_scores)
    }

with hret.evaluation_context(
    dataset="kmmlu",
    evaluation_method_name="custom",
    evaluator_params={"custom_evaluator": custom_evaluator}
) as ctx:
    result = ctx.evaluate(my_model)
```

## 📚 API Reference

### Configuration Classes

- **`HRETConfig`**: Global configuration management
  - `default_dataset`: Default dataset name
  - `default_model`: Default model backend
  - `mlflow_tracking`: Enable MLflow integration
  - `wandb_tracking`: Enable W&B integration
  - `output_dir`: Results output directory
  - `auto_save_results`: Automatically save results

### Core Classes

- **`EvaluationContext`**: Evaluation session management
  - `evaluate(model_function)`: Evaluate a model function
  - `benchmark(model_functions)`: Compare multiple models
  - `log_to_mlflow()`: Add MLflow integration
  - `log_to_wandb()`: Add W&B integration
  - `save_results()`: Save evaluation results

- **`MetricsTracker`**: Cross-run metrics tracking
  - `start_run()`: Start a new evaluation run
  - `log_metrics()`: Log metrics for current run
  - `end_run()`: End current run and save results
  - `compare_runs()`: Compare metrics across runs

### Decorators

- **`@evaluate()`**: Single model evaluation decorator
- **`@benchmark()`**: Multi-model comparison decorator
- **`@track_metrics()`**: Metrics tracking decorator

### Utility Functions

- **`configure()`**: Set global configuration
- **`load_config()`**: Load configuration from file
- **`quick_eval()`**: Quick model evaluation
- **`compare_models()`**: Compare multiple models
- **`evaluation_context()`**: Create evaluation context
- **`get_metrics_history()`**: Get evaluation history
- **`compare_metric()`**: Compare specific metric across runs

## 🔄 Migration from PipelineRunner

HRET maintains full backward compatibility with the existing PipelineRunner:

```python
# Existing code (still works)
from llm_eval.runner import PipelineRunner
runner = PipelineRunner(
    dataset_name="kmmlu", 
    model_backend_name="huggingface"
)
result = runner.run()

# New HRET approach
import llm_eval.hret as hret
result = hret.quick_eval(my_model_function, dataset="kmmlu")
```

## 🛠️ Best Practices

### 1. Use Configuration Files
```python
# Create hret_config.yaml for your project
hret.load_config("hret_config.yaml")
```

### 2. Implement Proper Error Handling
```python
@hret.evaluate(dataset="kmmlu")
def robust_model(input_text: str) -> str:
    try:
        return model.generate(input_text)
    except Exception as e:
        logger.error(f"Model generation failed: {e}")
        return ""  # Return empty string on error
```

### 3. Use Context Managers for Complex Workflows
```python
with hret.evaluation_context(dataset="kmmlu") as ctx:
    ctx.log_to_mlflow()
    # Multiple evaluations in same context
    result1 = ctx.evaluate(model1)
    result2 = ctx.evaluate(model2)
```

### 4. Implement Monitoring and Alerting
```python
def performance_monitor(run_result, results):
    accuracy = run_result["metrics"].get("accuracy", 0)
    if accuracy < PERFORMANCE_THRESHOLD:
        send_alert(f"Performance degradation: {accuracy}")

with hret.evaluation_context() as ctx:
    ctx.add_mlops_integration(performance_monitor)
```

## 📖 Examples

Complete examples are available in the `examples/` directory:

- `examples/hret_examples.py`: Basic usage examples
- `examples/mlops_integration_example.py`: MLOps integration patterns
- `examples/hret_config.yaml`: Configuration file template

## 🤝 Contributing

HRET is designed to be extensible. You can contribute by:

1. Adding new MLOps integrations
2. Implementing custom evaluation metrics
3. Creating new decorator patterns
4. Improving documentation and examples

For detailed contribution guidelines, see [07-contribution-guide.md](07-contribution-guide.md).

---

HRET makes LLM evaluation simple, powerful, and MLOps-ready. Start with the decorators for quick evaluations, then leverage context managers and configuration files for production deployments!