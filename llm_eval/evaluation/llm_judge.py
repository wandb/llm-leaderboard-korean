from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import logging
import time
from enum import Enum
from ..models.multi import MultiModel
from .base import BaseEvaluator
from ..utils.prompt_template import JUDGE_PROMPTS


# LLM 응답을 평가하기 위한 세 가지 평가 유형을 정의
class JudgeType(Enum):
    """LLM 응답 평가 유형.
    
    Attributes:
        RUBRIC_AND_RESPONSE: 루브릭 기준 기반 평가
        RUBRIC_RESPONSE_AND_GOLD: 루브릭, 응답, 정답 기반 평가
        RESPONSE_COMPARISON: 두 응답 간 비교 평가
    """
    RUBRIC_AND_RESPONSE = "rubric_and_response"          # 루브릭 기반 응답 평가
    RUBRIC_RESPONSE_AND_GOLD = "rubric_response_and_gold"  # 루브릭, 응답, 정답 기반 평가
    RESPONSE_COMPARISON = "response_comparison"           # 두 응답 간 비교 평가


# 평가에 필요한 입력 데이터를 정의하는 데이터 클래스
@dataclass
class JudgeInput:
    """평가에 필요한 입력 데이터를 정의하는 데이터 클래스.
    
    Attributes:
        judge_type: 수행할 평가 유형
        model_response: 평가할 모델의 응답
        rubric: 평가 기준 (선택사항)
        gold_response: 비교를 위한 정답 (선택사항)
        model_response_b: 비교를 위한 두 번째 모델 응답 (선택사항)
    """
    judge_type: JudgeType
    model_response: str
    rubric: Optional[str] = None          # 평가 기준
    gold_response: Optional[str] = None    # 정답
    model_response_b: Optional[str] = None # 비교할 다른 모델의 응답


class ResponseParser:
    """응답 파싱을 위한 기본 클래스."""
    
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        """LLM 응답을 파싱합니다.
        
        Args:
            response: LLM으로부터의 원본 응답 문자열
            model_name: 모델 이름 (선택사항)

        Returns:
            파싱된 결과를 담은 딕셔너리

        Raises:
            NotImplementedError: 하위 클래스에서 구현해야 함
        """
        raise NotImplementedError


# 루브릭 기반 점수 평가를 위한 파서
class RubricScoreParser(ResponseParser):
    """루브릭 기반 점수 평가를 위한 파서."""
    
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        """[[score: X]] 형식의 점수를 추출합니다.
        
        Args:
            response: 점수가 포함된 원본 응답 문자열
            model_name: 모델 이름 (선택사항)

        Returns:
            점수와 모델 이름이 포함된 딕셔너리

        Raises:
            ValueError: 응답이 None이거나 유효한 점수를 찾을 수 없는 경우
        """
        if not response:
            raise ValueError("Response is None")
        import re
        score_pattern = r"\[\[score:\s*(\d+\.?\d*)\]\]"
        match = re.search(score_pattern, response)
        if not match:
            raise ValueError(f"No valid score found in response: {response}")
        
        return {
            "score": float(match.group(1)),
            "model_name": model_name if model_name else "unknown"
        }


# 두 응답을 비교하여 승자를 결정하는 파서
class PairwiseComparisonParser(ResponseParser):
    """[[A]], [[B]], [[C]](tie) 형식으로 된 승자 판정을 추출"""
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        if not response:
            raise ValueError("Response is None")
        result = {}
        if "[[A]]" in response:
            result = {"winner": "A"}
        elif "[[B]]" in response:
            result = {"winner": "B"}
        elif "[[C]]" in response:
            result = {"winner": "tie"}
        else:
            raise ValueError(f"No valid verdict found in response: {response}")
        
        result["model_name"] = model_name if model_name else "unknown"
        return result


# 정답과 비교하여 정확성을 평가하는 파서
class GoldComparisonParser(ResponseParser):
    """[[true]] 또는 [[false]] 형식으로 된 정확성 판정을 추출"""
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        if not response:
            raise ValueError("Response is None")
        if "[[true]]" in response.lower():
            return {
                "correct": True,
                "step": -1,
                "model_name": model_name if model_name else "unknown"
            }
        elif "[[false]]" in response.lower():
            import re
            step_pattern = r"step:\s*\[(\d+)\]"
            match = re.search(step_pattern, response)
            if match:
                return {
                    "correct": False,
                    "step": int(match.group(1)),
                    "model_name": model_name if model_name else "unknown"
                }
        raise ValueError(f"No valid verdict found in response: {response}")


# 여러 LLM을 사용하여 평가를 수행하는 메인 클래스
class MultiLLMJudge:
    """여러 LLM을 사용하여 평가를 수행하는 메인 클래스.
    
    Args:
        models_config: 모델 구성 목록
        max_retries: 최대 재시도 횟수
        retry_delay: 재시도 간 대기 시간(초)
        aggregation_strategy: 결과 집계 전략 (majority/first/all)
        logger: 로거 인스턴스 (선택사항)

    Attributes:
        multi_model: 다중 모델 처리를 위한 MultiModel 인스턴스
        parsers: 평가 유형별 파서 매핑 딕셔너리
        prompt_templates: 평가 유형별 프롬프트 템플릿
    """
    def __init__(
        self,
        models_config: List[Dict[str, Any]],
        max_retries: int = 3,             # 재시도 최대 횟수
        retry_delay: float = 1.0,         # 재시도 간 대기 시간
        aggregation_strategy: str = "majority",  # 결과 집계 전략 (majority/first/all)
        logger: Optional[logging.Logger] = None
    ):
        self.multi_model = MultiModel(items_config=models_config)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.aggregation_strategy = aggregation_strategy
        self.logger = logger or logging.getLogger(__name__)
        
        self.parsers = {
            JudgeType.RUBRIC_AND_RESPONSE: RubricScoreParser(),
            JudgeType.RUBRIC_RESPONSE_AND_GOLD: GoldComparisonParser(),
            JudgeType.RESPONSE_COMPARISON: PairwiseComparisonParser()
        }
        
        self.prompt_templates = JUDGE_PROMPTS

    def _format_prompt(self, judge_input: JudgeInput) -> str:
        """평가 입력에 따라 프롬프트 템플릿을 포맷팅합니다.
        
        Args:
            judge_input: 평가를 위한 입력 데이터

        Returns:
            포맷팅된 프롬프트 문자열

        Raises:
            ValueError: 지원하지 않는 평가 유형인 경우
        """
        if judge_input.judge_type == JudgeType.RUBRIC_AND_RESPONSE:
            return self.prompt_templates["RUBRIC_AND_RESPONSE"].format(
                rubric=judge_input.rubric,
                response=judge_input.model_response
            )
        
        elif judge_input.judge_type == JudgeType.RESPONSE_COMPARISON:
            return self.prompt_templates["RESPONSE_COMPARISON"].format(
                response_a=judge_input.model_response,
                response_b=judge_input.model_response_b
            )
        
        elif judge_input.judge_type == JudgeType.RUBRIC_RESPONSE_AND_GOLD:
            return self.prompt_templates["RUBRIC_RESPONSE_AND_GOLD"].format(
                rubric=judge_input.rubric,
                gold_response=judge_input.gold_response,
                response=judge_input.model_response
            )
        
        raise ValueError(f"Unsupported judge type: {judge_input.judge_type}")

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """여러 모델의 평가 결과를 집계합니다.
        
        Args:
            results: 개별 평가 결과 목록

        Returns:
            전략에 따라 집계된 결과

        Raises:
            ValueError: 집계할 결과가 없거나 지원하지 않는 전략인 경우
        """
        if not results:
            raise ValueError("No results to aggregate")

        if self.aggregation_strategy == "majority":
            if "score" in results[0]:  # Rubric scoring
                scores = [result.get("score", 0) for result in results if "score" in result]
                models = [result.get("model_name", "unknown") for result in results if "score" in result]
                if not scores:
                    raise ValueError("No valid scores found in results")
                average_score = sum(scores) / len(scores)
                return {
                    "score": average_score,
                    "individual_scores": scores,
                    "model_names": models,
                    "num_models": len(scores)
                }
            elif "winner" in results[0]:  # Pairwise comparison
                winners = [result.get("winner") for result in results if "winner" in result]
                models = [result.get("model_name", "unknown") for result in results if "winner" in result]
                from collections import Counter
                winner_counts = Counter(winners)
                majority_winner = max(winner_counts.items(), key=lambda x: x[1])[0]
                return {
                    "winner": majority_winner,
                    "model_names": models
                }
            elif "correct" in results[0]:  # Gold comparison
                corrects = [result.get("correct", False) for result in results if "correct" in result]
                models = [result.get("model_name", "unknown") for result in results if "correct" in result]
                majority_correct = sum(corrects) > len(corrects) / 2
                return {
                    "correct": majority_correct,
                    "model_names": models
                }
        
        elif self.aggregation_strategy == "first":
            return results[0]
        
        elif self.aggregation_strategy == "all":
            return {"results": results}
        
        raise ValueError(f"Unsupported aggregation strategy: {self.aggregation_strategy}")

    def judge(self, judge_input: JudgeInput) -> Dict[str, Any]:
        """여러 LLM을 사용하여 평가를 수행합니다.
        
        이 메서드는 다음과 같은 전체 평가 프로세스를 처리합니다:
        1. 적절한 프롬프트 생성
        2. 여러 모델에 평가 요청
        3. 응답 파싱
        4. 결과 집계

        Args:
            judge_input: 평가를 위한 입력 데이터

        Returns:
            집계된 평가 결과

        Raises:
            ValueError: 최대 재시도 후에도 유효한 결과를 얻지 못한 경우
        """
        prompt = self._format_prompt(judge_input)
        parser = self.parsers[judge_input.judge_type]
        
        batch_input = [{
            "input": prompt,
            "judge_type": judge_input.judge_type.value
        }]
        
        all_results = []
        for attempt in range(self.max_retries):
            try:
                responses = self.multi_model.generate_batch(batch_input)
                self.logger.debug(f"Raw responses from models: {responses}")
                
                for response in responses:
                    self.logger.debug(f"Processing response: {response}")
                    
                    if "prediction" in response:
                        try:
                            result = parser.parse(
                                response["prediction"],
                                model_name=response.get("model_name", "unknown")
                            )
                            all_results.append(result)
                        except ValueError as e:
                            self.logger.warning(f"Failed to parse direct prediction: {str(e)}")
                    
                    if "multi_outputs" in response:
                        for model_output in response["multi_outputs"]:
                            try:
                                result = parser.parse(
                                    model_output["prediction"],
                                    model_name=model_output.get("model_name", "unknown")
                                )
                                all_results.append(result)
                            except ValueError as e:
                                self.logger.warning(f"Failed to parse response from {model_output.get('model_name', 'unknown')}: {str(e)}")
                
                if all_results:
                    return self._aggregate_results(all_results)
                
                self.logger.warning("No valid results were parsed from model responses")
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise ValueError(f"Failed to get valid judgments after {self.max_retries} attempts")
        
        raise ValueError("No valid results obtained from any model")