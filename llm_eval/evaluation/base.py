# llm_eval/evaluation/base.py
from typing import List, Dict, Any
from llm_eval.models.base import BaseModel 
from llm_eval.utils.metrics import * # 기타 메트릭들을 import

class BaseEvaluator:
    """
    (개발 용이를 위해 개발 단계에서는 한국어로 작성, 이후 영어로 대체)
    모든 Evaluator Class가 상속해야 할 BaseClass.

    주요 컨셉:
    - prepare_prompt: (선택) 입력 prompt를 수정하거나 CoT를 삽입하는 등의 로직을 수행.
    - parse_prediction: (선택) 모델의 raw output을 파싱(후처리)해 정답 비교가 용이하도록 가공.
    - evaluate_predictions: 실제 점수를 계산하는 메트릭 계산 로직.
    - evaluate: (상위 로직) 데이터를 순회하며 prompt 준비 -> 모델 호출 -> 예측 파싱 -> 점수 계산의 전체 플로우.
    
    필요시 requires_logits, requires_chain_of_thought 등 속성을 둬서 
    모델을 호출할 때 logits, CoT를 요청하는 등의 분기 처리가 가능.
    # TODO : requires_chain_of_thought 추가, scaling method와의 연동
    """

    name: str = "base"
    requires_logits: bool = False
    requires_chain_of_thought: bool = False

    def prepare_prompt(self, input_text: str) -> str:
        """
        (옵션) 'input' 텍스트를 받아, CoT 지시문이나 추가 시스템 메세지 등을 붙여
        모델에게 전달할 최종 프롬프트를 생성.
        기본 구현은 아무것도 하지 않고 그대로 반환.
        """
        return input_text

    def parse_prediction(self, raw_output: str) -> Any:
        """
        (옵션) 모델의 raw output(문자열 등)을 받아 정답 비교가 용이하도록 전처리/파싱.
        예: JSON 포맷으로 응답이 온다면 파싱, 
            '답변: ...' 형태면 '...'만 추출 등.
        기본 구현은 raw_output을 그대로 반환.
        """
        return raw_output

    def evaluate_predictions(
        self, 
        samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        (필수 구현) 
        이미 'prediction' 필드를 갖고 있는 샘플들(=모델 출력이 반영된 상태)에 대해
        실제 메트릭을 계산하고 결과(점수)를 반환.
        
        Args:
            samples: 각 원소가 
                {
                    "input": str,
                    "reference": str, 
                    "prompt": str,               # prepare_prompt 후 실제 모델에 들어간 프롬프트
                    "prediction": Any,           # parse_prediction 후 결과
                    "logits" (optional): Any,    # requires_logits=True인 경우, 모델이 반환한 logits 정보
                    ...
                } 
                형태를 갖는 리스트.

        Returns:
            Dict[str, float]: {"metric_name": metric_value, ...} 
            예: {"accuracy": 0.85, "f1": 0.76}
        """
        raise NotImplementedError("Subclasses must implement evaluate_predictions().")

    def evaluate(
        self, 
        data: List[Dict[str, Any]], 
        model: BaseModel
    ) -> Dict[str, Any]:
        """
        전체 평가 프로세스:
        1) 각 sample에 대해 prepare_prompt로 prompt 생성
        2) 모델에 입력 -> (logits, raw_output) 획득 (requires_logits에 따라 옵션 분기)
        3) parse_prediction으로 결과 파싱
        4) 모든 샘플에 대해 evaluate_predictions()로 최종 점수 계산
        5) 결과 리턴
        
        Args:
            data: [{"input":..., "reference":...}, ...] 형태의 전처리된 데이터.
            model: BaseModel을 상속한 모델 객체.
        Returns:
            {"metrics": { ... }, "samples": [ ... ] }
        """
        processed_samples = []

        # 1) prompt 준비
        for sample in data:
            prompt_text = self.prepare_prompt(sample["input"])
            sample["prompt"] = prompt_text  # 어떤 프롬프트로 모델을 호출했는지 기록
            processed_samples.append(sample)

        # 2) 모델 호출
        #    requires_logits=True인 경우, model.generate_batch 등에서 logits도 반환하도록 구현 필요
        #    여기서는 pseudo-code 형태로, 이후에 model 구현 후 이에 맞게 변형 필요
        if self.requires_logits:
            # 모델에서 logits까지 함께 반환한다고 가정
            predictions = model.generate_batch(
                processed_samples, 
                return_logits=True
            )
        else:
            predictions = model.generate_batch(
                processed_samples
            )

        # (generate_batch가 predictions에 raw_output, logits 등을 채워넣어 반환한다고 가정)
        # 예: [{"prompt":"...", "reference":"...", "raw_output":"...", "logits":..., ...}, ...]

        # 3) parse_prediction
        for sample in predictions:
            raw_output = sample.get("raw_output", "")
            parsed_pred = self.parse_prediction(raw_output)
            sample["prediction"] = parsed_pred

        # 4) 메트릭 계산
        metrics = self.evaluate_predictions(predictions)

        # 5) 결과 리턴
        return {
            "metrics": metrics,
            "samples": predictions
        }
