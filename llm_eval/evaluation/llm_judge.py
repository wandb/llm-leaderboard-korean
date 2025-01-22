from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import logging
import time
from enum import Enum
from ..models.multi import MultiModel
from .base import BaseEvaluator
from ..utils.prompt_template import JUDGE_PROMPTS


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


class ResponseParser:
    def parse(self, response: str) -> Dict[str, Any]:
        """Base parser interface"""
        raise NotImplementedError


class PairwiseComparisonParser(ResponseParser):
    def parse(self, response: str) -> Dict[str, Any]:
        if not response:  # 널 체크 추가
            raise ValueError("Response is None")
        if "[[A]]" in response:
            return {"winner": "A"}
        elif "[[B]]" in response:
            return {"winner": "B"}
        elif "[[C]]" in response:
            return {"winner": "tie"}
        raise ValueError(f"No valid verdict found in response: {response}")


class RubricScoreParser(ResponseParser):
    def parse(self, response: str) -> Dict[str, Any]:
        if not response:  # 널 체크 추가
            raise ValueError("Response is None")
        import re
        score_pattern = r"\[\[score:\s*(\d+\.?\d*)\]\]"
        match = re.search(score_pattern, response)
        if match:
            return {"score": float(match.group(1))}
        raise ValueError(f"No valid score found in response: {response}")


class GoldComparisonParser(ResponseParser):
    def parse(self, response: str) -> Dict[str, Any]:
        if not response:  # 널 체크 추가
            raise ValueError("Response is None")
        if "[[true]]" in response.lower():
            return {"correct": True, "step": -1}
        elif "[[false]]" in response.lower():
            import re
            step_pattern = r"step:\s*\[(\d+)\]"
            match = re.search(step_pattern, response)
            if match:
                return {"correct": False, "step": int(match.group(1))}
        raise ValueError(f"No valid verdict found in response: {response}")


class MultiLLMJudge:
    def __init__(
        self,
        models_config: List[Dict[str, Any]],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        aggregation_strategy: str = "majority",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize MultiLLMJudge with multiple model configurations
        
        Args:
            models_config: List of model configurations for MultiModel
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            aggregation_strategy: Strategy to aggregate multiple judge results
                                Options: 'majority', 'average', 'first'
            logger: Optional logger instance
        """
        self.multi_model = MultiModel(items_config=models_config)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.aggregation_strategy = aggregation_strategy
        self.logger = logger or logging.getLogger(__name__)
        
        # Register parsers for different judge types
        self.parsers = {
            JudgeType.RUBRIC_AND_RESPONSE: RubricScoreParser(),
            JudgeType.RUBRIC_RESPONSE_AND_GOLD: GoldComparisonParser(),
            JudgeType.RESPONSE_COMPARISON: PairwiseComparisonParser()
        }
        
        # Template prompts for different judge types
        self.prompt_templates = JUDGE_PROMPTS

    def _format_prompt(self, judge_input: JudgeInput) -> str:
        """Format prompt based on judge type and input"""
        # Using formatted string templates directly based on judge type
        if judge_input.judge_type == JudgeType.RUBRIC_AND_RESPONSE:
            return f"""You are an expert evaluator. Your task is to evaluate the given response based on the rubric and provide a score.

IMPORTANT: You must format your response exactly like this example:
Based on the rubric, this response deserves [[score: 7]].

Rubric:
{judge_input.rubric}

Response to evaluate:
{judge_input.model_response}

Provide your evaluation with the score in the specified format:"""
            
        elif judge_input.judge_type == JudgeType.RESPONSE_COMPARISON:
            return f"""You are an expert evaluator. Your task is to compare two responses and choose the better one.

IMPORTANT: You must format your verdict exactly like this:
- Use [[A]] to choose the first response
- Use [[B]] to choose the second response
- Use [[C]] if they are equally good

Response A:
{judge_input.model_response}

Response B:
{judge_input.model_response_b}

Provide your verdict in the specified format:"""
            
        elif judge_input.judge_type == JudgeType.RUBRIC_RESPONSE_AND_GOLD:
            return f"""You are an expert evaluator. Please evaluate if the following response matches the gold standard answer.
Compare step by step and provide your verdict as [[true]] if correct or [[false]] step: [X] if incorrect.

Rubric:
{judge_input.rubric}

Gold Response:
{judge_input.gold_response}

Model Response to evaluate:
{judge_input.model_response}

Provide your evaluation in the specified format:"""
        
        raise ValueError(f"Unsupported judge type: {judge_input.judge_type}")

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple models"""
        if not results:
            raise ValueError("No results to aggregate")

        if self.aggregation_strategy == "majority":
            # Handle different types of results based on judge type
            if "score" in results[0]:  # Rubric scoring
                scores = [result.get("score", 0) for result in results if "score" in result]
                if not scores:
                    raise ValueError("No valid scores found in results")
                average_score = sum(scores) / len(scores)
                return {
                    "score": average_score,
                    "individual_scores": scores,
                    "num_models": len(scores)
                }
            elif "winner" in results[0]:  # Pairwise comparison
                winners = [result.get("winner") for result in results if "winner" in result]
                from collections import Counter
                winner_counts = Counter(winners)
                majority_winner = max(winner_counts.items(), key=lambda x: x[1])[0]
                return {"winner": majority_winner}
            elif "correct" in results[0]:  # Gold comparison
                corrects = [result.get("correct", False) for result in results if "correct" in result]
                majority_correct = sum(corrects) > len(corrects) / 2
                return {"correct": majority_correct}
        
        elif self.aggregation_strategy == "first":
            return results[0]
        
        elif self.aggregation_strategy == "all":
            return {"results": results}
        
        raise ValueError(f"Unsupported aggregation strategy: {self.aggregation_strategy}")

    def judge(self, judge_input: JudgeInput) -> Dict[str, Any]:
        """Get judgment from multiple LLMs and aggregate results"""
        prompt = self._format_prompt(judge_input)
        parser = self.parsers[judge_input.judge_type]
        
        # Prepare batch input for MultiModel
        batch_input = [{
            "input": prompt,
            "judge_type": judge_input.judge_type.value
        }]
        
        all_results = []
        for attempt in range(self.max_retries):
            try:
                # Generate responses from all models
                responses = self.multi_model.generate_batch(batch_input)
                
                # Log raw responses for debugging
                self.logger.debug(f"Raw responses from models: {responses}")
                
                for response in responses:
                    self.logger.debug(f"Processing response: {response}")
                    
                    # Handle response with direct prediction
                    if "prediction" in response:
                        try:
                            result = parser.parse(response["prediction"])
                            result["model_name"] = response.get("model_name", "unknown")
                            all_results.append(result)
                        except ValueError as e:
                            self.logger.warning(f"Failed to parse direct prediction: {str(e)}")
                    
                    # Handle response with multi_outputs
                    if "multi_outputs" in response:
                        for model_output in response["multi_outputs"]:
                            try:
                                result = parser.parse(model_output["prediction"])
                                result["model_name"] = model_output["model_name"]
                                all_results.append(result)
                            except ValueError as e:
                                self.logger.warning(f"Failed to parse response from {model_output['model_name']}: {str(e)}")
                
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