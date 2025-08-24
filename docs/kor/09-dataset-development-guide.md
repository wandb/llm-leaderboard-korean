# 데이터셋 개발 가이드

환영합니다! 이 문서는 Haerae Evaluation Toolkit에 새 데이터셋을 추가하는 방법을 설명합니다. 설계 철학(레지스트리/퍼사드), 핵심 인터페이스, 단계별 구현 절차와 모범 사례를 다룹니다.

## 설계 철학

본 프로젝트는 유연성과 확장성을 최우선으로 합니다. 핵심 원칙은 다음과 같습니다.

1) 레지스트리 패턴(Extensibility)
   - 데이터셋/모델/스케일링/평가기 모두 데코레이터로 등록합니다.
   - 신규 구성 요소는 작은 클래스를 만들고 `@register_*` 데코레이터만 붙이면 됩니다.
   - 침습적 수정 없이 손쉽게 확장 가능합니다.

2) 퍼사드 패턴(Usability)
   - Evaluator와 PipelineRunner가 퍼사드 역할을 하여 데이터 로딩, 추론, 스케일링, 평가를 오케스트레이션합니다.
   - 기여자는 자신의 컴포넌트 구현에만 집중하면 되며 퍼사드가 나머지를 연결합니다.

이 철학 덕분에 팀 간 결합도를 낮추고, 각자 독립적으로 기여하기 수월합니다.

## 데이터셋 인터페이스 요약

- 베이스 클래스: `llm_eval.datasets.base.BaseDataset`
- 필수 메서드:
  - `load() -> List[Dict[str, Any]]`: 표준화된 샘플 리스트 반환
    - 각 샘플에는 최소한 다음 키 필요:
      - `input`: 모델이 받는 전체 입력 문자열
      - `reference`: 정답/라벨 문자열
    - 선택적 키(자주 사용): `_subset_name`, `options`, `metadata`
- 선택 메서드:
  - `get_raw_samples()`: 원본 데이터 접근
  - `info() -> Dict[str, Any]`: 데이터셋 메타데이터

## 레지스트리 사용

- 파일: `llm_eval/datasets/__init__.py`
- 데코레이터: `@register_dataset("키")`
- 레지스트리에 등록된 키로 로더를 얻습니다:

```python
from llm_eval.datasets import load_datasets
loader = load_datasets(name="your_key", subset="your_subset", split="test")
samples = loader.load()
```

## 표준 출력 스키마

아래와 같은 리스트를 반환하세요:

```python
[
  {
    "input": "...",               # 모델 입력(지시문 포함 권장)
    "reference": "...",           # 정답 문자열
    "_subset_name": "subset_a",   # 선택: 서브셋별 지표 분리용
    "options": ["(A)", ...],      # 선택: 객관식 옵션
    "metadata": { ... }            # 선택: 부가 정보
  },
]
```

`input`은 자기완결적이게(지시문/템플릿 포함) 구성하는 것을 권장합니다.

## 프롬프트 템플릿

- `base_prompt_template`를 도입하면 `input` 포맷팅이 쉬워집니다.
- 합리적인 기본값을 제공하고, `dataset_params`로 덮어쓸 수 있게 하세요.

예:
```python
default_template = "최종 답은 'Answer:' 뒤에 한 줄로 제시하세요.\n\n{question}"
formatted = default_template.format(question=raw)
```

## 서브셋과 스플릿

- `subset`은 None | str | List[str] 모두 지원하는 것을 권장합니다.
- HF Hub의 경우 제공 스플릿이 제한적일 수 있습니다. `self.split` → `test` → `validation` → `train` 순으로 폴백을 구현해 주세요.
- 서브셋이 있다면 `_subset_name`을 꼭 채워서 분석 모듈이 서브셋별 지표를 계산할 수 있게 하세요.

## 허용 평가기 설정

- 특정 평가기만 허용하려면 `info()`에 명시하세요:

```python
return { "evaluation_only": ["string_match", "math_match"] }
```

- 제한이 없으면 None 또는 키 자체를 생략하세요. PipelineRunner가 유효성 검사를 수행합니다.

## 단계별 구현 절차

1) 새 파일 생성: `llm_eval/datasets/my_dataset.py`
2) `BaseDataset` 상속 클래스 작성, `@register_dataset("키")` 부착
3) `load()`에서 표준 샘플 포맷으로 변환하여 반환
4) 레지스트리 연결: `llm_eval/datasets/__init__.py`에 `from .my_dataset import MyDataset` 추가
5) (선택) `examples/`에 예시 설정 파일 추가
6) (선택) 문서/기여가이드 업데이트

## 예시: 로컬 파일 로더

`dataset_loader.py`의 `generic_file`도 활용 가능합니다. 직접 구현 예시는:

```python
@register_dataset("my_local")
class MyLocalDataset(BaseDataset):
    def load(self):
        return [{"input": row["question"], "reference": row["answer"]} for row in rows]
```

## 예시: HF Hub 데이터셋

```python
from datasets import load_dataset

@register_dataset("my_hf")
class MyHFDataset(BaseDataset):
    def load(self):
        ds = load_dataset("owner/name", split=self.split)
        out = []
        for it in ds:
            q = it.get("question", "")
            a = str(it.get("answer", ""))
            inp = self.base_prompt_template.format(question=q) if self.base_prompt_template else q
            out.append({"input": inp, "reference": a, "_subset_name": "default"})
        return out
```

## 모범 사례

- 간결하고 읽기 쉬운 코드, 작은 헬퍼 활용
- 불필요한 필드는 지양하고 표준 스키마 유지
- 가능하면 결정적 순서 보장
- 모델 프롬프트 또는 평가기 선택을 통해 정규화/형식 일관성 확보
- 기존 평가기 재사용 우선: `string_match`, `math_match`, `partial_match`, `llm_judge` 등

## 테스트

- `llm_eval/test/test_datasets.py`가 레지스트리 및 `load()` 반환 형식을 점검합니다.
- 네트워크 의존(HF Hub) 데이터셋은 일시 오류를 견딜 수 있도록 테스트가 관대하게 설계되어 있습니다.

## 문의

- 기여 워크플로우: `docs/kor/07-contribution-guide.md`
- GitHub 이슈/디스커션으로 문의해 주세요.
