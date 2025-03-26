## LLM 평가 프레임워크 Contribution Guide

본 가이드는 Haerae-evaluation-toolkit에 기여하고자 하는 개발자분들이 Dataset, Evaluator, Scaling, 그리고 Backend 영역에서 새 기능을 추가하거나 개선할 때 따라야 할 절차와 사례를 상세하게 안내합니다. 이 contribution guide을 따르면 코드의 일관성을 유지하고, 전체 코드베이스와 원활하게 통합되며, 유지보수와 확장이 용이한 기여를 할 수 있습니다.

---

### 전반적인 원칙

#### • 코드 스타일과 일관성
- 프로젝트 내 기존 코드의 네이밍 규칙, 들여쓰기, 주석 스타일, 에러 처리 방식을 그대로 따르세요.(PEP8)
- 새로 작성하는 모든 클래스, 함수, 메서드에는 기능, 인자, 반환값 등을 상세하게 설명하는 Docstring이나 주석을 추가해 주세요.

#### • 문서화와 테스트
- 기여하는 코드에는 사용법과 예제를 포함한 문서화를 반드시 하세요.
- 각 모듈의 역할과 사용법이 명확하게 드러나도록 하여, 다른 개발자나 사용자가 쉽게 이해할 수 있도록 해주세요.
- 변경사항이 기존 기능에 영향을 미치지 않도록 충분한 테스트를 작성하거나 기존 테스트를 업데이트하세요.
- 새로 추가한 기능은 최소한의 단위 테스트를 포함시키는 것을 권장합니다.

#### • 레지스트리 패턴 준수
- dataset, evaluation, scaling_method, backend 각 영역은 데코레이터 기반의 레지스트리 시스템을 사용해 확장 가능하도록 설계되어 있습니다.
- 새로운 클래스를 추가할 때 반드시 해당 모듈의 레지스트리에 등록되었는지 확인하세요.
  - 예: dataset 기여 시 `@register_dataset("이름")`, 평가 기여 시 `@register_evaluator("이름")`

#### • 코드 재사용과 모듈화
- 각 기여 항목은 가능한 한 다른 모듈과 독립적이며 재사용 가능하도록 작성되어야 합니다.
- 기능별로 역할을 명확하게 분리(데이터 로딩, 모델 추론, 디코딩/스케일링, 평가 등)하여, 파이프라인 내에서 쉽게 교체하거나 수정할 수 있도록 해주세요.

---

### 2. Dataset 기여 가이드

#### • BaseDataset 상속 및 구현
- 새 데이터셋 클래스를 구현할 때는 반드시 `BaseDataset`을 상속받으세요.
- 생성자에서는 `dataset_name`, `split`, `subset` (필요시) 및 추가 파라미터(HuggingFace 옵션 등)를 받아 내부 변수에 저장합니다.
- `load()` 메서드를 구현하여, 데이터를 불러온 후 각 샘플을 `{ "input": ..., "reference": ..., (추가 필드) }` 형태의 리스트로 반환합니다.
- 예: subset이 None이면 전체 데이터를, 문자열이면 해당 subset만, 리스트이면 여러 subset의 데이터를 합치는 로직을 구현

#### • 선택적 메서드
- `get_raw_samples()`: 원본 데이터를 그대로 반환 (디버깅/분석 시 유용)
- `info()`: 전체 샘플 수, 하위 태스크 정보 등의 메타 정보를 반환

#### • 레지스트리 등록
- `@register_dataset("데이터셋_이름")` 데코레이터를 사용해 등록하세요.
- CLI나 다른 모듈에서 `load_dataset("데이터셋_이름")`으로 호출 가능

#### • 팁
- 인증 문제, 네트워크 오류 등 예외 상황에 대한 에러 처리를 꼼꼼히 하세요.
- 각 데이터셋의 필드 이름이 다를 수 있으므로, 필드 매핑을 주석으로 명확히 남기세요.

---

### 3. Evaluation 기여 가이드

#### • BaseEvaluator 상속 및 구현
- `BaseEvaluator`를 상속받아 새로운 평가 로직을 구현하세요.
- 필요 시 `prepare_prompt()`, `parse_prediction()` 메서드를 오버라이드하여 프롬프트 수정, 출력 파싱 수행
- `evaluate_predictions(samples)`를 구현하여 메트릭을 계산하고 `{"metric_name": value, ...}` 형태로 반환
- `evaluate(data, model)` 메서드는 전체 평가 흐름 관리 (MultiModel 사용 시 `judge_batch()` 포함 가능)

#### • 커스텀 CoT Parser 지원
- 평가 클래스에서 CoT parser 함수를 받을 수 있도록 인터페이스 제공
- parser 함수는 `(chain_of_thought, final_answer)` 형태의 튜플 반환 필수
- 주석에 parser 사용 방식 상세히 명시

#### • 레지스트리 등록
- `@register_evaluator("평가_이름")` 데코레이터를 사용해 등록하세요.

#### • 팁
- 다양한 메트릭 예제를 참고하여 기존 평가 방법과 일관성 있게 작성하세요.
- 디버깅을 위해 중간 출력(파싱 결과, 점수 계산 등)을 로깅으로 남기세요.

---

### 4. Scaling Method 기여 가이드

#### • BaseScalingMethod 상속 및 구현
- `BaseScalingMethod`를 상속받아 새로운 스케일링 전략 구현
- `apply(data)` 메서드에서 Beam Search, Best-of-N, Majority Voting 등 전략을 적용
- 필요 시 `set_params()`를 구현해 파라미터 동적 업데이트 지원

#### • 레지스트리 등록
- `@register_scaling_method("스케일링_방법_이름")` 데코레이터를 사용해 등록하세요.

#### • 팁
- 각 선택 기준 (log prob 누적, beam pruning 기준 등)을 주석으로 명확히 작성
- 중복 제거, EOS 처리 등 추가 기능도 고려해 구현하세요.

---

### 5. Backend 기여 가이드

#### • BaseModel, BaseJudge, BaseRewardModel 상속 및 구현
- 각각 `BaseModel`, `BaseJudge`, `BaseRewardModel`을 상속하여 구현
- `BaseModel`: `generate_batch(inputs, return_logits=False)` 구현
- `BaseJudge`: `judge_batch(inputs)` 구현
- `BaseRewardModel`: `score_batch(inputs)` 구현

#### • 레지스트리 등록
- `@register_model("백엔드_이름")` 데코레이터를 사용해 등록하세요.

#### • 팁
- 대규모 모델이나 외부 API 사용 시 성능 최적화, 에러 처리 중요
- 설정값은 환경변수/설정 파일로 관리하고, 주석으로 명시
- HuggingFace 등 외부 라이브러리 사용 시 버전 호환성 유의

---

### 추가 참고사항 및 팁

#### • Custom CoT Parser
- `(chain_of_thought, final_answer)` 튜플을 반환하는 파서 함수 구현 가능
- Evaluator나 PipelineRunner에서 `custom_cot_parser`로 전달 가능
- CLI에서는 `--cot_parser` 인자로 "패키지.모듈.함수명" 문자열 전달

#### • CLI 및 API 연동
- 기여 코드는 CLI와 API 양쪽에서 재사용 가능하도록 설계
- JSON 문자열 파라미터는 `_parse_json_str()` 등 헬퍼 함수 활용
- `PipelineRunner`, `Evaluator` 객체 구조를 API로도 쉽게 노출 가능

#### • 테스트와 디버깅
- 기능 추가 후 반드시 단위 테스트, 통합 테스트 작성
- 각 단계(데이터 로딩, 추론, 평가 등)의 중간 결과는 로깅 활용
- 코드 리뷰를 통해 다른 기여자들과 개선점 논의

---

### 마무리

기여를 시작하기 전에 이 가이드를 숙지해 주시고, 궁금한 사항이나 개선 제안이 있으시면 프로젝트 관리자(quick_start.md 참고)에게 문의해 주시기 바랍니다.

