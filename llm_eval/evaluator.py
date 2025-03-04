#!/usr/bin/env python
"""
Evaluator CLI 

This module provides a high-level interface to run the full LLM evaluation pipeline,
which includes dataset loading, model inference (with optional scaling),
evaluation, and additional post-processing such as language penalization.
"""

import argparse
import json
import logging
from typing import Any, Dict, List, Optional, Union

from llm_eval.runner import PipelineRunner
from llm_eval.utils.logging import get_logger
from llm_eval.utils.util import EvaluationResult

logger = get_logger(name="evaluator", level=logging.INFO)


def _parse_json_str(json_str: Optional[str]) -> Dict[str, Any]:
    """
    Parses a JSON string into a dictionary.

    If the input string is None or empty, returns an empty dictionary.
    If parsing fails due to a JSONDecodeError, logs a warning and returns {}.
    """
    if not json_str:
        return {}
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON string: {json_str}, error: {e}")
        return {}


class Evaluator:
    """
    High-level interface for running the full LLM evaluation pipeline.
    This class leverages PipelineRunner internally to connect dataset, model,
    scaling method, and evaluator components. It supports judge/reward backends
    via MultiModel if specified.
    """

    def __init__(
        self,
        default_model_backend: str = "huggingface",
        default_judge_backend: Optional[str] = None,
        default_reward_backend: Optional[str] = None,
        default_scaling_method: Optional[str] = None,
        default_evaluation_method: str = "string_match",
        default_split: str = "test",
    ):
        """
        Initializes the Evaluator with default backend configurations.

        Args:
            default_model_backend (str): Default generation model backend.
            default_judge_backend (Optional[str]): Default judge model backend (if any).
            default_reward_backend (Optional[str]): Default reward model backend (if any).
            default_scaling_method (Optional[str]): Default scaling method (e.g., "beam_search").
            default_evaluation_method (str): Default evaluation method (e.g., "string_match").
            default_split (str): Default dataset split (e.g., "test").
        """
        self.default_model_backend = default_model_backend
        self.default_judge_backend = default_judge_backend
        self.default_reward_backend = default_reward_backend
        self.default_scaling_method = default_scaling_method
        self.default_evaluation_method = default_evaluation_method
        self.default_split = default_split

    def run(
        self,
        model: Optional[str] = None,
        judge_model: Optional[str] = None,
        reward_model: Optional[str] = None,
        dataset: str = "haerae_bench",
        subset: Union[str, List[str], None] = None,
        split: Optional[str] = None,
        scaling_method: Optional[str] = None,
        evaluation_method: Optional[str] = None,
        dataset_params: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        judge_params: Optional[Dict[str, Any]] = None,
        reward_params: Optional[Dict[str, Any]] = None,
        scaling_params: Optional[Dict[str, Any]] = None,
        evaluator_params: Optional[Dict[str, Any]] = None,
        language_penalize: bool = True,
    ) -> EvaluationResult:
        """
        Runs the full LLM evaluation pipeline, which involves:
          1. Loading the dataset,
          2. Loading the model (or constructing a MultiModel if judge/reward are provided),
          3. Applying the scaling method (if specified),
          4. Evaluating the generated predictions,
          5. Optionally applying a language penalizer,
          6. Returning the final results as an EvaluationResult.

        Args:
            model (Optional[str]): Main model backend name. Defaults to default_model_backend.
            judge_model (Optional[str]): Judge model backend name.
            reward_model (Optional[str]): Reward model backend name.
            dataset (str): Dataset name (registry key). Defaults to "haerae_bench".
            subset (Union[str, List[str], None]): Dataset subset/config name.
            split (Optional[str]): Dataset split (e.g., "train", "test"). Defaults to default_split.
            scaling_method (Optional[str]): Scaling method name (registry key). Defaults to default_scaling_method.
            evaluation_method (Optional[str]): Evaluation method name (registry key). Defaults to default_evaluation_method.
            dataset_params (Optional[Dict[str, Any]]): Additional parameters for dataset loading.
            model_params (Optional[Dict[str, Any]]): Additional parameters for the main model.
            judge_params (Optional[Dict[str, Any]]): Additional parameters for the judge model.
            reward_params (Optional[Dict[str, Any]]): Additional parameters for the reward model.
            scaling_params (Optional[Dict[str, Any]]): Additional parameters for the scaling method.
            evaluator_params (Optional[Dict[str, Any]]): Additional parameters for the evaluator.
            language_penalize (bool): If True, apply the language penalizer to predictions.

        Returns:
            EvaluationResult: Object containing evaluation metrics, sample outputs, and run info.
        """
        from llm_eval.utils.util import EvaluationResult

        model_backend_name = model or self.default_model_backend
        judge_backend_name = judge_model or self.default_judge_backend
        reward_backend_name = reward_model or self.default_reward_backend

        final_split = split if split is not None else self.default_split
        final_scaling = scaling_method or self.default_scaling_method
        final_eval = evaluation_method or self.default_evaluation_method

        # Determine if MultiModel is needed
        use_multi = (judge_backend_name is not None) or (reward_backend_name is not None)

        if use_multi:
            multi_config = {
                "generate_model": None,
                "judge_model": None,
                "reward_model": None,
            }
            if model_backend_name:
                multi_config["generate_model"] = {
                    "name": model_backend_name,
                    "params": model_params or {},
                }
            if judge_backend_name:
                multi_config["judge_model"] = {
                    "name": judge_backend_name,
                    "params": judge_params or {},
                }
            if reward_backend_name:
                multi_config["reward_model"] = {
                    "name": reward_backend_name,
                    "params": reward_params or {},
                }
            runner = PipelineRunner(
                dataset_name=dataset,
                subset=subset,
                split=final_split,
                model_backend_name="multi",
                scaling_method_name=final_scaling,
                evaluation_method_name=final_eval,
                dataset_params=dataset_params or {},
                model_backend_params=multi_config,
                scaling_params=scaling_params or {},
                evaluator_params=evaluator_params or {},
                language_penalize=language_penalize,
            )
        else:
            runner = PipelineRunner(
                dataset_name=dataset,
                subset=subset,
                split=final_split,
                model_backend_name=model_backend_name,
                scaling_method_name=final_scaling,
                evaluation_method_name=final_eval,
                dataset_params=dataset_params or {},
                model_backend_params=model_params or {},
                scaling_params=scaling_params or {},
                evaluator_params=evaluator_params or {},
                language_penalize=language_penalize,
            )

        eval_result: EvaluationResult = runner.run()
        return eval_result


def main():
    """
    Command-line entry point for the Evaluator CLI.
    Supports configuration via JSON parameters for dataset, model, scaling, and evaluator.
    The final evaluation results are output as JSON, either to stdout or to a specified file.
    """
    parser = argparse.ArgumentParser(description="LLM Evaluator CLI (Supports Judge/Reward)")
    parser.add_argument("--model", type=str, default=None, help="Main model backend name")
    parser.add_argument("--judge_model", type=str, default=None, help="Judge model backend name")
    parser.add_argument("--reward_model", type=str, default=None, help="Reward model backend name")
    parser.add_argument("--dataset", type=str, default="haerae_bench", help="Dataset name (registry key)")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset/config name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (e.g., 'train', 'test')")
    parser.add_argument("--scaling_method", type=str, default=None, help="Scaling method (registry key)")
    parser.add_argument("--evaluation_method", type=str, default="string_match", help="Evaluation method (registry key)")
    parser.add_argument("--output_file", type=str, default=None, help="File path to save JSON results")

    # Additional parameters passed as JSON strings
    parser.add_argument("--dataset_params", type=str, default=None, help="JSON string for dataset parameters")
    parser.add_argument("--model_params", type=str, default=None, help="JSON string for main model parameters")
    parser.add_argument("--judge_params", type=str, default=None, help="JSON string for judge model parameters")
    parser.add_argument("--reward_params", type=str, default=None, help="JSON string for reward model parameters")
    parser.add_argument("--scaling_params", type=str, default=None, help="JSON string for scaling method parameters")
    parser.add_argument("--evaluator_params", type=str, default=None, help="JSON string for evaluator parameters")

    # Language penalizer flags and target language parameter
    parser.add_argument("--language_penalize", action="store_true", help="Enable language penalizer")
    parser.add_argument("--no_language_penalize", action="store_true", help="Disable language penalizer")
    parser.add_argument("--target_lang", type=str, default="ko", help="Target language code for penalizer (e.g., 'ko')")

    args = parser.parse_args()

    # Determine language penalize flag based on arguments
    language_penalize_flag = True
    if args.no_language_penalize:
        language_penalize_flag = False
    elif args.language_penalize:
        language_penalize_flag = True


    dataset_params = _parse_json_str(args.dataset_params)
    model_params = _parse_json_str(args.model_params)
    judge_params = _parse_json_str(args.judge_params)
    reward_params = _parse_json_str(args.reward_params)
    scaling_params = _parse_json_str(args.scaling_params)
    evaluator_params = _parse_json_str(args.evaluator_params)

    evaluator = Evaluator(
        default_model_backend="huggingface",
        default_judge_backend=None,
        default_reward_backend=None,
        default_scaling_method=None,
        default_evaluation_method="string_match",
        default_split="test",
    )

    eval_result = evaluator.run(
        model=args.model,
        judge_model=args.judge_model,
        reward_model=args.reward_model,
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        scaling_method=args.scaling_method,
        evaluation_method=args.evaluation_method,
        dataset_params=dataset_params,
        model_params=model_params,
        judge_params=judge_params,
        reward_params=reward_params,
        scaling_params=scaling_params,
        evaluator_params=evaluator_params,
        language_penalize=language_penalize_flag,
    )

    result_dict = eval_result.to_dict()

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_file}")
    else:
        print(json.dumps(result_dict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
