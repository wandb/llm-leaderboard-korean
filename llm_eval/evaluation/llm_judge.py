from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass
import json
from enum import Enum
import logging
import time
from abc import ABC, abstractmethod
from openai import OpenAI
import os

class JudgeType(Enum):
    RUBRIC_AND_RESPONSE = "rubric_and_response"
    RUBRIC_RESPONSE_AND_GOLD = "rubric_response_and_gold"
    RESPONSE_COMPARISON = "response_comparison"

@dataclass
class JudgeInput:
    judge_type: JudgeType
    model_response: str
    rubric: Optional[str] = None
    gold_response: Optional[str] = None
    model_response_b: Optional[str] = None

class ResponseParser(ABC):
    @abstractmethod
    def parse(self, response: str) -> Dict[str, Any]:
        pass

class PairwiseComparisonParser(ResponseParser):
    def parse(self, response: str) -> Dict[str, Any]:
        """
        [[A]], [[B]], 또는 [[C]] 형식을 찾아 응답을 파싱합니다
        """
        if "[[A]]" in response:
            return {"winner": "A"}
        elif "[[B]]" in response:
            return {"winner": "B"}
        elif "[[C]]" in response:
            return {"winner": "tie"}
        raise ValueError("No valid verdict found in response")

class RubricScoreParser(ResponseParser):
    def parse(self, response: str) -> Dict[str, Any]:
        """
        지정된 형식의 점수를 찾아 응답을 파싱합니다
        예시: [[score: 8.5]]
        """
        import re
        score_pattern = r"\[\[score:\s*(\d+\.?\d*)\]\]"
        match = re.search(score_pattern, response)
        if match:
            return {"score": float(match.group(1))}
        raise ValueError("No valid score found in response")

class GoldComparisonParser(ResponseParser):
    def parse(self, response: str) -> Dict[str, Any]:
        """
        [[true]] 또는 [[false]]와 단계 번호를 찾아 응답을 파싱합니다
        """
        if "[[true]]" in response:
            return {"correct": True, "step": -1}
        elif "[[false]]" in response:
            import re
            step_pattern = r"step:\s*\[(\d+)\]"
            match = re.search(step_pattern, response)
            if match:
                return {"correct": False, "step": int(match.group(1))}
        raise ValueError("No valid verdict found in response")

class OpenAIClient:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4",  # 또는 "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # 평가는 일관성이 중요하므로 temperature를 0으로 설정
        )
        return response.choices[0].message.content

class LLMJudge:
    def __init__(
        self,
        llm_client: Any,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logger or logging.getLogger(__name__)
        
        # Register parsers for different judge types
        self.parsers = {
            JudgeType.RUBRIC_AND_RESPONSE: RubricScoreParser(),
            JudgeType.RUBRIC_RESPONSE_AND_GOLD: GoldComparisonParser(),
            JudgeType.RESPONSE_COMPARISON: PairwiseComparisonParser()
        }
        
        # Template prompts for different judge types
        self.prompt_templates = {
            JudgeType.RUBRIC_AND_RESPONSE: """
Please evaluate the following response according to the provided rubric.
Rate on a scale of 0-10 and provide your verdict in this format: [[score: X]]

Rubric:
{rubric}

Response to evaluate:
{response}
""",
            JudgeType.RUBRIC_RESPONSE_AND_GOLD: """
Please evaluate the following response against the gold standard answer.
Compare step by step and provide your verdict as [[true]] if correct or [[false]] step: [X] if incorrect.

Question:
{rubric}

Gold Answer:
{gold_response}

Model Response:
{response}
""",
            JudgeType.RESPONSE_COMPARISON: """
Please compare the following two responses and choose the better one.
Provide your verdict as [[A]] for first response, [[B]] for second response, or [[C]] for tie.

Response A:
{response_a}

Response B:
{response_b}
"""
        }

    def _format_prompt(self, judge_input: JudgeInput) -> str:
        template = self.prompt_templates[judge_input.judge_type]
        
        format_args = {
            "rubric": judge_input.rubric,
            "response": judge_input.model_response,
            "gold_response": judge_input.gold_response,
            "response_a": judge_input.model_response,
            "response_b": judge_input.model_response_b
        }
        
        # Remove None values
        format_args = {k: v for k, v in format_args.items() if v is not None}
        
        return template.format(**format_args)

    def _call_llm(self, prompt: str) -> str:
        """
        재시도 로직을 포함하여 LLM을 호출합니다
        특정 LLM 클라이언트에 맞게 구현해야 합니다
        """
        for attempt in range(self.max_retries):
            try:
                # This should be adapted based on your LLM client's API
                response = self.llm_client.generate(prompt)
                return response
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise

    def judge(self, judge_input: JudgeInput) -> Dict[str, Any]:
        """
        LLM으로부터 판단을 얻기 위한 주요 메서드
        """
        prompt = self._format_prompt(judge_input)
        parser = self.parsers[judge_input.judge_type]
        
        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(prompt)
                result = parser.parse(response)
                return result
            except ValueError as e:
                self.logger.warning(f"Parsing attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise ValueError(f"Failed to parse LLM response after {self.max_retries} attempts")

