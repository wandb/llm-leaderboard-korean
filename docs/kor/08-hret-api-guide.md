# HRET API 가이드 - MLOps 친화적 평가 인터페이스

HRET (Haerae Evaluation Toolkit)는 MLOps 파이프라인과의 원활한 통합을 위해 설계된 deepeval 스타일의 데코레이터 기반 API를 제공합니다. 이 고수준 인터페이스는 기존 PipelineRunner와의 완전한 호환성을 유지하면서 최소한의 보일러플레이트 코드로 LLM 평가를 가능하게 합니다.

## 🚀 빠른 시작

```python
import llm_eval.hret as hret

# 간단한 데코레이터 기반 평가
@hret.evaluate(dataset="kmmlu", model="huggingface")
def my_model(input_text: str) -> str:
    return model.generate(input_text)

# 평가 실행
result = my_model()
print(f"정확도: {result.metrics['accuracy']}")
```

## 📋 핵심 기능

### 1. 데코레이터 기반 API

#### `@hret.evaluate()`
최소한의 설정으로 단일 모델 함수를 평가합니다.

```python
@hret.evaluate(
    dataset="kmmlu", 
    model="huggingface", 
    evaluation_method="string_match"
)
def my_model(input_text: str) -> str:
    return model.generate(input_text)

result = my_model()  # EvaluationResult 반환
```

**매개변수:**
- `dataset`: 데이터셋 이름 (예: "kmmlu", "haerae", "benchhub")
- `model`: 모델 백엔드 이름 (예: "huggingface", "openai")
- `evaluation_method`: 평가 방법 (예: "string_match", "llm_judge")
- `**kwargs`: 추가 설정 매개변수

#### `@hret.benchmark()`
동일한 데이터셋에서 여러 모델을 비교합니다.

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
평가 함수에서 특정 메트릭을 추적합니다.

```python
@hret.track_metrics(["accuracy", "latency", "cost"])
def custom_evaluation():
    # 평가 로직
    return {
        "accuracy": 0.85,
        "latency": 120.5,
        "cost": 0.02,
        "other_metric": "추적되지_않음"
    }
```

### 2. 컨텍스트 매니저 인터페이스

평가 프로세스에 대한 더 세밀한 제어가 필요한 경우:

```python
with hret.evaluation_context(
    dataset="kmmlu", 
    run_name="my_experiment"
) as ctx:
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
    "baseline": lambda x: baseline_model.generate(x),
    "improved": lambda x: improved_model.generate(x)
}
results = hret.compare_models(models, dataset="kmmlu")
```

## 🔧 설정 관리

### 전역 설정

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

### 설정 파일

`hret_config.yaml` 파일 생성:

```yaml
# HRET 설정
default_dataset: "kmmlu"
default_model: "huggingface"
default_split: "test"
default_evaluation_method: "string_match"

# MLOps 통합
mlflow_tracking: true
wandb_tracking: false
tensorboard_tracking: false

# 출력 설정
output_dir: "./hret_results"
auto_save_results: true
log_level: "INFO"

# 성능 설정
batch_size: 32
max_workers: 4
```

설정 로드:
```python
hret.load_config("hret_config.yaml")
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
    # 모니터링 시스템으로 전송
    send_to_monitoring_system(run_result, results)

with hret.evaluation_context(dataset="kmmlu") as ctx:
    ctx.add_mlops_integration(custom_logger)
    result = ctx.evaluate(my_model)
```

## 📊 메트릭 추적 및 분석

### 실행 간 메트릭 추적

```python
# 모든 평가 기록 조회
history = hret.get_metrics_history()
print(f"총 실행 횟수: {len(history)}")

# 실행 간 특정 메트릭 비교
accuracy_comparison = hret.compare_metric("accuracy")
print(f"최고 정확도: {accuracy_comparison['_stats']['best']}")
print(f"평균 정확도: {accuracy_comparison['_stats']['average']}")
```

### 고급 메트릭 추적

```python
tracker = hret.MetricsTracker()

# 여러 실험 실행
for model_name, model_func in models.items():
    tracker.start_run(run_name=f"eval_{model_name}")
    
    result = hret.quick_eval(model_func, dataset="kmmlu")
    tracker.log_metrics(result.metrics)
    
    tracker.end_run()

# 결과 비교
comparison = tracker.compare_runs("accuracy")
```

## 🏭 MLOps 파이프라인 통합

### 훈련 파이프라인 통합

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
        @hret.evaluate(dataset="kmmlu", model="huggingface")
        def checkpoint_model(input_text):
            return self.model.generate(input_text)
        
        with hret.evaluation_context(
            run_name=f"checkpoint_epoch_{epoch}"
        ) as ctx:
            ctx.log_to_mlflow(experiment_name="training")
            result = ctx.evaluate(checkpoint_model)
            
            # 성능 모니터링
            if self.detect_degradation(result):
                self.send_alert(epoch, result)
```

### 하이퍼파라미터 튜닝

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
            # 하이퍼파라미터를 메타데이터로 추가
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
            
            # 성능 모니터링
            if self.detect_degradation(result):
                self.send_performance_alert(result)
```

## 🎯 고급 사용 패턴

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

### A/B 테스트

```python
@hret.benchmark(dataset="kmmlu")
def ab_test():
    return {
        "model_a": lambda x: model_a.generate(x),
        "model_b": lambda x: model_b.generate(x)
    }

results = ab_test()

# 통계적 유의성 검정
from scipy import stats
scores_a = [s['accuracy'] for s in results['model_a'].samples]
scores_b = [s['accuracy'] for s in results['model_b'].samples]
t_stat, p_value = stats.ttest_ind(scores_a, scores_b)

print(f"A/B 테스트 결과:")
print(f"모델 A: {results['model_a'].metrics['accuracy']:.3f}")
print(f"모델 B: {results['model_b'].metrics['accuracy']:.3f}")
print(f"P-값: {p_value:.4f}")
```

### 커스텀 평가 메트릭

```python
def custom_evaluator(predictions, references):
    # 커스텀 평가 로직 구현
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

## 📚 API 레퍼런스

### 설정 클래스

- **`HRETConfig`**: 전역 설정 관리
  - `default_dataset`: 기본 데이터셋 이름
  - `default_model`: 기본 모델 백엔드
  - `mlflow_tracking`: MLflow 통합 활성화
  - `wandb_tracking`: W&B 통합 활성화
  - `output_dir`: 결과 출력 디렉토리
  - `auto_save_results`: 결과 자동 저장

### 핵심 클래스

- **`EvaluationContext`**: 평가 세션 관리
  - `evaluate(model_function)`: 모델 함수 평가
  - `benchmark(model_functions)`: 여러 모델 비교
  - `log_to_mlflow()`: MLflow 통합 추가
  - `log_to_wandb()`: W&B 통합 추가
  - `save_results()`: 평가 결과 저장

- **`MetricsTracker`**: 실행 간 메트릭 추적
  - `start_run()`: 새 평가 실행 시작
  - `log_metrics()`: 현재 실행의 메트릭 로깅
  - `end_run()`: 현재 실행 종료 및 결과 저장
  - `compare_runs()`: 실행 간 메트릭 비교

### 데코레이터

- **`@evaluate()`**: 단일 모델 평가 데코레이터
- **`@benchmark()`**: 다중 모델 비교 데코레이터
- **`@track_metrics()`**: 메트릭 추적 데코레이터

### 유틸리티 함수

- **`configure()`**: 전역 설정 지정
- **`load_config()`**: 파일에서 설정 로드
- **`quick_eval()`**: 빠른 모델 평가
- **`compare_models()`**: 여러 모델 비교
- **`evaluation_context()`**: 평가 컨텍스트 생성
- **`get_metrics_history()`**: 평가 기록 조회
- **`compare_metric()`**: 실행 간 특정 메트릭 비교

## 🔄 PipelineRunner에서 마이그레이션

HRET는 기존 PipelineRunner와 완전한 하위 호환성을 유지합니다:

```python
# 기존 코드 (여전히 작동)
from llm_eval.runner import PipelineRunner
runner = PipelineRunner(
    dataset_name="kmmlu", 
    model_backend_name="huggingface"
)
result = runner.run()

# 새로운 HRET 방식
import llm_eval.hret as hret
result = hret.quick_eval(my_model_function, dataset="kmmlu")
```

## 🛠️ 모범 사례

### 1. 설정 파일 사용
```python
# 프로젝트용 hret_config.yaml 생성
hret.load_config("hret_config.yaml")
```

### 2. 적절한 오류 처리 구현
```python
@hret.evaluate(dataset="kmmlu")
def robust_model(input_text: str) -> str:
    try:
        return model.generate(input_text)
    except Exception as e:
        logger.error(f"모델 생성 실패: {e}")
        return ""  # 오류 시 빈 문자열 반환
```

### 3. 복잡한 워크플로우에는 컨텍스트 매니저 사용
```python
with hret.evaluation_context(dataset="kmmlu") as ctx:
    ctx.log_to_mlflow()
    # 동일한 컨텍스트에서 여러 평가
    result1 = ctx.evaluate(model1)
    result2 = ctx.evaluate(model2)
```

### 4. 모니터링 및 알림 구현
```python
def performance_monitor(run_result, results):
    accuracy = run_result["metrics"].get("accuracy", 0)
    if accuracy < PERFORMANCE_THRESHOLD:
        send_alert(f"성능 저하: {accuracy}")

with hret.evaluation_context() as ctx:
    ctx.add_mlops_integration(performance_monitor)
```

## 📖 예시

완전한 예시는 `examples/` 디렉토리에서 확인할 수 있습니다:

- `examples/hret_examples.py`: 기본 사용법 예시
- `examples/mlops_integration_example.py`: MLOps 통합 패턴
- `examples/hret_config.yaml`: 설정 파일 템플릿

## 🤝 기여하기

HRET는 확장 가능하도록 설계되었습니다. 다음과 같은 방법으로 기여할 수 있습니다:

1. 새로운 MLOps 통합 추가
2. 커스텀 평가 메트릭 구현
3. 새로운 데코레이터 패턴 생성
4. 문서 및 예시 개선

자세한 기여 가이드라인은 [07-contribution-guide.md](07-contribution-guide.md)를 참조하세요.

---

HRET는 LLM 평가를 간단하고 강력하며 MLOps 준비가 된 상태로 만듭니다. 빠른 평가를 위해서는 데코레이터로 시작하고, 프로덕션 배포를 위해서는 컨텍스트 매니저와 설정 파일을 활용하세요!