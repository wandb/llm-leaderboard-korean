# HRET (Haerae Evaluation Toolkit) API

HRET는 MLOps 환경에서 쉽게 통합할 수 있는 deepeval 스타일의 데코레이터 기반 API를 제공합니다. 이 API는 LLM 평가를 위한 고수준 인터페이스를 제공하며, 최소한의 보일러플레이트 코드로 평가를 수행할 수 있습니다.

## 🚀 Quick Start

```python
import hret

# 간단한 데코레이터 기반 평가
@hret.evaluate(dataset="kmmlu", model="huggingface")
def my_model(input_text: str) -> str:
    return model.generate(input_text)

# 평가 실행
result = my_model()
print(f"Accuracy: {result.metrics['accuracy']}")
```

## 📋 주요 기능

### 1. 데코레이터 기반 API

#### `@hret.evaluate()`
단일 모델 함수를 평가합니다.

```python
@hret.evaluate(dataset="kmmlu", model="huggingface", evaluation_method="string_match")
def my_model(input_text: str) -> str:
    return model.generate(input_text)

result = my_model()  # EvaluationResult 반환
```

#### `@hret.benchmark()`
여러 모델을 비교 평가합니다.

```python
@hret.benchmark(dataset="kmmlu")
def compare_models():
    return {
        "gpt-4": lambda x: gpt4_model.generate(x),
        "claude-3": lambda x: claude_model.generate(x),
        "custom": lambda x: custom_model.generate(x)
    }

results = compare_models()  # Dict[str, EvaluationResult] 반환
```

#### `@hret.track_metrics()`
특정 메트릭을 추적합니다.

```python
@hret.track_metrics(["accuracy", "latency"])
def custom_evaluation():
    return {
        "accuracy": 0.85,
        "latency": 120.5,
        "other_metric": "not_tracked"
    }
```

### 2. 컨텍스트 매니저

더 세밀한 제어가 필요한 경우 컨텍스트 매니저를 사용할 수 있습니다.

```python
with hret.evaluation_context(dataset="kmmlu", run_name="my_experiment") as ctx:
    # MLOps 통합 설정
    ctx.log_to_mlflow(experiment_name="llm_experiments")
    ctx.log_to_wandb(project_name="model_evaluation")
    
    # 평가 실행
    result = ctx.evaluate(my_model_function)
    
    # 결과 저장
    ctx.save_results("experiment_results.json")
```

### 3. 편의 함수

#### 빠른 평가
```python
result = hret.quick_eval(my_model_function, dataset="kmmlu")
```

#### 모델 비교
```python
models = {
    "model_a": lambda x: model_a.generate(x),
    "model_b": lambda x: model_b.generate(x)
}
results = hret.compare_models(models, dataset="kmmlu")
```

## 🔧 설정 관리

### 글로벌 설정

```python
hret.configure(
    default_dataset="kmmlu",
    default_model="huggingface",
    mlflow_tracking=True,
    wandb_tracking=True,
    output_dir="./my_results",
    auto_save_results=True
)
```

### 설정 파일 사용

```python
# YAML 또는 JSON 설정 파일 로드
hret.load_config("hret_config.yaml")
```

설정 파일 예시 (`hret_config.yaml`):
```yaml
default_dataset: "kmmlu"
default_model: "huggingface"
mlflow_tracking: true
wandb_tracking: true
output_dir: "./results"
auto_save_results: true
log_level: "INFO"
```

## 🔗 MLOps 통합

### MLflow 통합

```python
with hret.evaluation_context(dataset="kmmlu") as ctx:
    ctx.log_to_mlflow(experiment_name="my_experiments")
    result = ctx.evaluate(my_model)
    # 결과가 자동으로 MLflow에 로깅됩니다
```

### Weights & Biases 통합

```python
with hret.evaluation_context(dataset="kmmlu") as ctx:
    ctx.log_to_wandb(project_name="llm_evaluation")
    result = ctx.evaluate(my_model)
    # 결과가 자동으로 W&B에 로깅됩니다
```

### 커스텀 통합

```python
def custom_logger(run_result, results):
    # 커스텀 로깅 로직
    send_to_monitoring_system(run_result, results)

with hret.evaluation_context(dataset="kmmlu") as ctx:
    ctx.add_mlops_integration(custom_logger)
    result = ctx.evaluate(my_model)
```

## 📊 메트릭 추적

### 실행 기록 조회

```python
# 모든 실행 기록 조회
history = hret.get_metrics_history()

# 특정 메트릭 비교
accuracy_comparison = hret.compare_metric("accuracy")
print(f"Best accuracy: {accuracy_comparison['_stats']['best']}")
```

### 메트릭 추적기 사용

```python
tracker = hret.MetricsTracker()

tracker.start_run("experiment_1")
tracker.log_metrics({"accuracy": 0.85, "f1": 0.82})
result = tracker.end_run()

# 여러 실행 비교
comparison = tracker.compare_runs("accuracy")
```

## 🏭 MLOps 파이프라인 예시

### 모델 훈련 파이프라인

```python
class ModelTrainingPipeline:
    def train_and_evaluate(self, epochs=10):
        for epoch in range(1, epochs + 1):
            # 모델 훈련
            self.train_epoch(epoch)
            
            # 체크포인트 평가
            if epoch % 3 == 0:
                self.evaluate_checkpoint(epoch)
    
    def evaluate_checkpoint(self, epoch):
        def model_function(input_text):
            return self.model.generate(input_text)
        
        with hret.evaluation_context(
            run_name=f"checkpoint_epoch_{epoch}"
        ) as ctx:
            ctx.log_to_mlflow(experiment_name="training")
            result = ctx.evaluate(model_function)
            
            # 성능 저하 감지
            if self.is_performance_degraded(result):
                self.send_alert(epoch, result)
```

### 하이퍼파라미터 튜닝

```python
def hyperparameter_tuning():
    hyperparams = [
        {"lr": 0.001, "batch_size": 16},
        {"lr": 0.01, "batch_size": 32},
        {"lr": 0.005, "batch_size": 64},
    ]
    
    best_result = None
    best_score = 0
    
    for i, params in enumerate(hyperparams):
        model_function = create_model_with_params(params)
        
        with hret.evaluation_context(
            run_name=f"hyperparam_run_{i}"
        ) as ctx:
            ctx.metrics_tracker.run_metadata.update({
                "hyperparameters": params
            })
            ctx.log_to_mlflow(experiment_name="hyperparameter_tuning")
            
            result = ctx.evaluate(model_function)
            
            if result.metrics["accuracy"] > best_score:
                best_score = result.metrics["accuracy"]
                best_result = (params, result)
    
    return best_result
```

### 지속적 평가

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
            
            # 성능 저하 감지 및 알림
            if self.detect_degradation(result):
                self.send_alert(result)
```

## 🎯 고급 사용법

### 배치 처리

```python
# 대용량 데이터셋을 위한 배치 처리
hret.configure(batch_size=32, max_workers=4)

with hret.evaluation_context(dataset="large_dataset") as ctx:
    result = ctx.evaluate(my_model)
```

### 커스텀 평가 메트릭

```python
def custom_evaluator(predictions, references):
    # 커스텀 평가 로직
    return {"custom_metric": calculate_custom_score(predictions, references)}

with hret.evaluation_context(
    dataset="kmmlu",
    evaluation_method_name="custom",
    evaluator_params={"custom_evaluator": custom_evaluator}
) as ctx:
    result = ctx.evaluate(my_model)
```

### 다중 데이터셋 평가

```python
datasets = ["kmmlu", "haerae", "benchhub"]
results = {}

for dataset in datasets:
    with hret.evaluation_context(dataset_name=dataset) as ctx:
        ctx.log_to_mlflow(experiment_name="multi_dataset_eval")
        results[dataset] = ctx.evaluate(my_model)

# 데이터셋별 성능 비교
for dataset, result in results.items():
    print(f"{dataset}: {result.metrics['accuracy']:.3f}")
```

## 📚 API 레퍼런스

### 주요 클래스

- `HRETConfig`: 글로벌 설정 관리
- `EvaluationContext`: 평가 컨텍스트 관리
- `MetricsTracker`: 메트릭 추적 및 비교

### 주요 함수

- `configure(**kwargs)`: 글로벌 설정
- `load_config(path)`: 설정 파일 로드
- `evaluation_context()`: 평가 컨텍스트 생성
- `quick_eval()`: 빠른 평가
- `compare_models()`: 모델 비교
- `get_metrics_history()`: 실행 기록 조회
- `compare_metric()`: 메트릭 비교

### 데코레이터

- `@evaluate()`: 단일 모델 평가
- `@benchmark()`: 다중 모델 비교
- `@track_metrics()`: 메트릭 추적

## 🔍 예시 코드

전체 예시 코드는 다음 파일들을 참조하세요:

- `examples/hret_examples.py`: 기본 사용법 예시
- `examples/mlops_integration_example.py`: MLOps 통합 예시
- `examples/hret_config.yaml`: 설정 파일 예시

## 🤝 기존 코드와의 호환성

HRET는 기존의 `PipelineRunner`와 완전히 호환됩니다. 기존 코드를 수정하지 않고도 HRET API를 점진적으로 도입할 수 있습니다.

```python
# 기존 방식 (여전히 작동)
from llm_eval.runner import PipelineRunner
runner = PipelineRunner(dataset_name="kmmlu", model_backend_name="huggingface")
result = runner.run()

# 새로운 HRET 방식
import hret
result = hret.quick_eval(my_model_function, dataset="kmmlu")
```

## 🚀 시작하기

1. HRET 모듈 import
2. 모델 함수 정의
3. 데코레이터 또는 컨텍스트 매니저 사용
4. MLOps 통합 설정 (선택사항)
5. 평가 실행 및 결과 분석

HRET를 사용하면 복잡한 LLM 평가 파이프라인을 간단하고 직관적인 API로 구축할 수 있습니다!