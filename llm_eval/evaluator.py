import argparse
import json
import logging
from typing import Any, Dict, List, Optional, Union

from llm_eval.runner import PipelineRunner
from llm_eval.utils.logging import get_logger

logger = get_logger(name="evaluator", level=logging.INFO)


def _parse_json_str(json_str: Optional[str]) -> Dict[str, Any]:
    """
    Helper function:
      - If `json_str` is not None, parse it as JSON and return the dictionary.
      - If parsing fails or `json_str` is None, return an empty dict.
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
    A high-level Evaluator class providing a simple interface for running the LLM
    evaluation pipeline. It can optionally handle a main generation model, a
    judge model, and a reward model.

    If both judge_model and reward_model are None, it will load a single
    model backend directly. Otherwise, it will create a 'multi' backend
    (MultiModel) with the specified generate, judge, and reward models.
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
        Args:
            default_model_backend (str): Name of the default generation model backend.
            default_judge_backend (str | None): Name of the default judge model backend.
            default_reward_backend (str | None): Name of the default reward model backend.
            default_scaling_method (str | None): Name of the default scaling method (e.g., beam_search).
            default_evaluation_method (str): Name of the default evaluation method (e.g., string_match).
            default_split (str): Default dataset split (train/valid/test).
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
    ) -> Dict[str, Any]:
        """
        Runs the full LLM evaluation pipeline:
          1) Load the dataset
          2) Load the model (or MultiModel if judge/reward models are provided)
          3) Apply the scaling method (if any)
          4) Perform evaluation
          5) Return the results (metrics, samples, etc.)

        Args:
            model (str | None): Generation model backend name (e.g., "huggingface")
            judge_model (str | None): Judge model backend name (e.g., "hf_judge")
            reward_model (str | None): Reward model backend name (e.g., "hf_reward")
            dataset (str): Dataset identifier (from DATASET_REGISTRY)
            subset (str | list[str] | None): Dataset subset or config
            split (str): Dataset split (train/valid/test)
            scaling_method (str | None): Name of the scaling/decoding method (e.g., best_of_n)
            evaluation_method (str): Name of the evaluation method (e.g., string_match, llm_judge)
            dataset_params (dict): Additional dataset loader parameters
            model_params (dict): Additional generation model parameters
            judge_params (dict): Additional judge model parameters
            reward_params (dict): Additional reward model parameters
            scaling_params (dict): Additional scaling method parameters
            evaluator_params (dict): Additional evaluation parameters

        Returns:
            Dict[str, Any]: Dictionary containing the final evaluation results, e.g.:
                {
                  "metrics": {...},
                  "samples": [...],
                  "info": {...}
                }
        """
        model_backend_name = model or self.default_model_backend
        judge_backend_name = judge_model or self.default_judge_backend
        reward_backend_name = reward_model or self.default_reward_backend

        final_split = split if split is not None else self.default_split
        final_scaling = scaling_method or self.default_scaling_method
        final_eval = evaluation_method or self.default_evaluation_method

        # Determine if MultiModel is needed
        use_multi = (judge_backend_name is not None) or (reward_backend_name is not None)

        if use_multi:
            # Construct MultiModel config
            multi_config = {
                "generate_model": None,
                "judge_model": None,
                "reward_model": None
            }

            # Generate model
            if model_backend_name:
                multi_config["generate_model"] = {
                    "name": model_backend_name,
                    "params": model_params or {}
                }
            # Judge model
            if judge_backend_name:
                multi_config["judge_model"] = {
                    "name": judge_backend_name,
                    "params": judge_params or {}
                }
            # Reward model
            if reward_backend_name:
                multi_config["reward_model"] = {
                    "name": reward_backend_name,
                    "params": reward_params or {}
                }

            # Use "multi" backend in the runner
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
            )
        else:
            # Use a single model backend
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
            )

        # Execute the pipeline
        results = runner.run()
        return results


def main():
    parser = argparse.ArgumentParser(description="LLM Evaluator CLI (Supports Judge/Reward)")
    parser.add_argument("--model", type=str, default=None, help="Main model backend name")
    parser.add_argument("--judge_model", type=str, default=None, help="Judge model backend name")
    parser.add_argument("--reward_model", type=str, default=None, help="Reward model backend name")

    parser.add_argument("--dataset", type=str, default="haerae_bench")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--scaling_method", type=str, default=None)
    parser.add_argument("--evaluation_method", type=str, default="string_match")
    parser.add_argument("--output_file", type=str, default=None)

    # JSON parameters
    parser.add_argument("--dataset_params", type=str, default=None)
    parser.add_argument("--model_params", type=str, default=None)
    parser.add_argument("--judge_params", type=str, default=None)
    parser.add_argument("--reward_params", type=str, default=None)
    parser.add_argument("--scaling_params", type=str, default=None)
    parser.add_argument("--evaluator_params", type=str, default=None)

    args = parser.parse_args()

    # Parse the JSON strings
    dataset_params = _parse_json_str(args.dataset_params)
    model_params = _parse_json_str(args.model_params)
    judge_params = _parse_json_str(args.judge_params)
    reward_params = _parse_json_str(args.reward_params)
    scaling_params = _parse_json_str(args.scaling_params)
    evaluator_params = _parse_json_str(args.evaluator_params)

    evaluator = Evaluator()
    results = evaluator.run(
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
    )

    # Output the results
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_file}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
