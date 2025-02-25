import argparse
import json
import logging
from typing import Any, Dict, List, Optional, Union

from llm_eval.runner import PipelineRunner
from llm_eval.utils.logging import get_logger
from llm_eval.utils.util import EvaluationResult

logger = get_logger(name="evaluator", level=logging.INFO)


def _parse_json_str(json_str: Optional[str]) -> Dict[str, Any]:
    """Parses a JSON string into a dictionary.

    If the input string is None or empty, returns an empty dictionary.
    If parsing fails due to `json.JSONDecodeError`, logs a warning and returns an empty dictionary.

    Args:
        json_str (Optional[str]): The JSON string to parse.

    Returns:
        Dict[str, Any]: The parsed dictionary. Empty if input is None or parsing fails.
    """
    if not json_str:
        return {}
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON string: {json_str}, error: {e}")
        return {}


class Evaluator:
    """High-level interface for running the LLM evaluation pipeline.

    This class can optionally handle:
      - A main generation model,
      - A judge model (LLM-as-a-Judge), and
      - A reward model (for reward scoring).

    If both `judge_model` and `reward_model` are `None`, a single model backend
    is loaded. Otherwise, it constructs a 'multi' backend (MultiModel) that
    contains the generate, judge, and/or reward models.

    Attributes:
        default_model_backend (str): Name of the default generation model backend.
        default_judge_backend (Optional[str]): Name of the default judge model backend.
        default_reward_backend (Optional[str]): Name of the default reward model backend.
        default_scaling_method (Optional[str]): Name of the default scaling method (e.g., 'beam_search').
        default_evaluation_method (str): Name of the default evaluation method (e.g., 'string_match').
        default_split (str): Default dataset split ('train', 'valid', or 'test').
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
        """Initializes the Evaluator with default backend configurations.

        Args:
            default_model_backend (str, optional):
                Name of the default generation model backend. Defaults to "huggingface".
            default_judge_backend (Optional[str], optional):
                Name of the default judge model backend. If None, no judge backend is used by default.
            default_reward_backend (Optional[str], optional):
                Name of the default reward model backend. If None, no reward backend is used by default.
            default_scaling_method (Optional[str], optional):
                Name of the default scaling method (e.g., "beam_search"). Defaults to None.
            default_evaluation_method (str, optional):
                Name of the default evaluation method. Defaults to "string_match".
            default_split (str, optional):
                Default dataset split. Typically "train", "valid", or "test". Defaults to "test".
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
    ) -> EvaluationResult:
        """Runs the full LLM evaluation pipeline.

        This involves:
          1) Loading the dataset,
          2) Loading the model (or constructing a MultiModel if judge/reward models are provided),
          3) Applying the scaling method (if any),
          4) Performing the evaluation,
          5) Returning the final results wrapped in an `EvaluationResult` object.

        Args:
            model (Optional[str], optional):
                The main model backend name to use. If None, uses `default_model_backend`.
            judge_model (Optional[str], optional):
                The judge model backend name. If provided, triggers MultiModel usage.
            reward_model (Optional[str], optional):
                The reward model backend name. If provided, triggers MultiModel usage.
            dataset (str, optional):
                The dataset name in the registry. Defaults to "haerae_bench".
            subset (Union[str, List[str], None], optional):
                The subset or config for the dataset (e.g., "csat_geo"). Defaults to None.
            split (Optional[str], optional):
                Which data split to use. If None, uses `default_split`. Defaults to None.
            scaling_method (Optional[str], optional):
                Name of the scaling method in the registry (e.g., "beam_search").
                If None, uses `default_scaling_method`. Defaults to None.
            evaluation_method (Optional[str], optional):
                Name of the evaluation method in the registry (e.g., "string_match").
                If None, uses `default_evaluation_method`. Defaults to None.
            dataset_params (Optional[Dict[str, Any]], optional):
                Additional parameters for the dataset loader.
            model_params (Optional[Dict[str, Any]], optional):
                Additional parameters for the main model. Only used if not in MultiModel mode.
            judge_params (Optional[Dict[str, Any]], optional):
                Additional parameters for the judge model in MultiModel mode.
            reward_params (Optional[Dict[str, Any]], optional):
                Additional parameters for the reward model in MultiModel mode.
            scaling_params (Optional[Dict[str, Any]], optional):
                Additional parameters for the scaling method (e.g., beam width, number of samples).
            evaluator_params (Optional[Dict[str, Any]], optional):
                Additional parameters for the evaluator (e.g., CoT flags).

        Returns:
            EvaluationResult:
                An object containing the evaluation metrics, sample-level outputs,
                and possibly extra information about the run.

        Raises:
            RuntimeError: If any required configuration is invalid or missing.
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
            # Construct MultiModel config
            multi_config = {
                "generate_model": None,
                "judge_model": None,
                "reward_model": None
            }

            # Fill the multi_config dictionary
            if model_backend_name:
                multi_config["generate_model"] = {
                    "name": model_backend_name,
                    "params": model_params or {}
                }
            if judge_backend_name:
                multi_config["judge_model"] = {
                    "name": judge_backend_name,
                    "params": judge_params or {}
                }
            if reward_backend_name:
                multi_config["reward_model"] = {
                    "name": reward_backend_name,
                    "params": reward_params or {}
                }

            # Use "multi" backend in the pipeline runner
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

        eval_result: EvaluationResult = runner.run()
        return eval_result


def main():
    """Command-line entry point for the Evaluator.

    This CLI interface supports:
      - A main model, a judge model, and a reward model (each optional except the main model).
      - A dataset name, subset, and split.
      - A scaling method and evaluation method.
      - Additional JSON-encoded parameters for dataset, model, judge, reward, scaling, and evaluator.

    The pipeline results are saved to a JSON file if `--output_file` is specified,
    otherwise printed to stdout.

    Raises:
        SystemExit: If argparse fails or invalid command-line options are given.
    """
    parser = argparse.ArgumentParser(description="LLM Evaluator CLI (Supports Judge/Reward)")
    parser.add_argument("--model", type=str, default=None, help="Main model backend name")
    parser.add_argument("--judge_model", type=str, default=None, help="Judge model backend name")
    parser.add_argument("--reward_model", type=str, default=None, help="Reward model backend name")

    parser.add_argument("--dataset", type=str, default="haerae_bench", help="Dataset name (registry key)")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset/config name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (e.g. 'train', 'test')")
    parser.add_argument("--scaling_method", type=str, default=None, help="Scaling method (registry key)")
    parser.add_argument("--evaluation_method", type=str, default="string_match", help="Evaluation method (registry key)")
    parser.add_argument("--output_file", type=str, default=None, help="Where to store results (JSON)")

    # JSON parameters
    parser.add_argument("--dataset_params", type=str, default=None, help="JSON string for dataset params")
    parser.add_argument("--model_params", type=str, default=None, help="JSON string for main model params")
    parser.add_argument("--judge_params", type=str, default=None, help="JSON string for judge model params")
    parser.add_argument("--reward_params", type=str, default=None, help="JSON string for reward model params")
    parser.add_argument("--scaling_params", type=str, default=None, help="JSON string for scaling method params")
    parser.add_argument("--evaluator_params", type=str, default=None, help="JSON string for evaluator params")

    args = parser.parse_args()

    # Parse the JSON strings
    dataset_params = _parse_json_str(args.dataset_params)
    model_params = _parse_json_str(args.model_params)
    judge_params = _parse_json_str(args.judge_params)
    reward_params = _parse_json_str(args.reward_params)
    scaling_params = _parse_json_str(args.scaling_params)
    evaluator_params = _parse_json_str(args.evaluator_params)

    evaluator = Evaluator()

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
    )

    # Convert EvaluationResult to dict for serialization
    result_dict = eval_result.to_dict()

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_file}")
    else:
        print(json.dumps(result_dict, ensure_ascii=False, indent=2))
