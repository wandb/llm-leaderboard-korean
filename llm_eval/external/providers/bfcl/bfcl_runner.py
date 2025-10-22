# import argparse  # CLI interface moved to evaluator.py
import statistics
from collections import defaultdict
import os
import sys
from pathlib import Path

# Add bfcl directory to Python path so it can find bfcl_eval
_bfcl_path = Path(__file__).parent
if str(_bfcl_path) not in sys.path:
    sys.path.insert(0, str(_bfcl_path))

from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.constants.eval_config import *
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.eval_checker.eval_runner_helper import *
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
    multi_turn_irrelevance_checker,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    is_empty_execute_response,
)
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import parse_prompt_variation_params
from bfcl_eval.utils import *

# from dotenv import load_dotenv  # CLI interface moved to evaluator.py
from tqdm import tqdm
from llm_eval.wandb_controller import WeaveEvalsController
import wandb
import pandas as pd


def get_handler(model_name: str) -> BaseHandler:
    if model_name not in MODEL_CONFIG_MAPPING:
        all_models = list(MODEL_CONFIG_MAPPING.keys())
        raise ValueError(f"Model '{model_name}' is not found in MODEL_CONFIG_MAPPING.\n" f"Available models:\n{all_models}")
    config = MODEL_CONFIG_MAPPING[model_name]
    registry_name = config.model_name.replace("/", "_")
    is_fc_model = config.is_fc_model
    handler: BaseHandler = config.model_handler(model_name, temperature=0, registry_name=registry_name, is_fc_model=is_fc_model)  # Temperature doesn't matter for evaluation
    return handler


def _evaluate_single_agentic_entry(
    handler: BaseHandler,
    index,
    model_result_list,
    possible_answer_item,
    prompt_entry,
    model_name,
    test_category,
):
    """Helper method to process a single agentic entry."""
    # Remove the function doc from the score file for better readability
    if "function" in prompt_entry:
        del prompt_entry["function"]

    # Agentic test is a single-turn multi-step test, so the model result should be a list of one element
    if type(model_result_list) != list or len(model_result_list) != 1:
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": {
                "error_message": ["Error during inference phase. Model did not output a list of model responses."],
                "error_type": "agentic:inference_error",
            },
            "prompt": prompt_entry,
            "model_result": model_result_list,
            "possible_answer": possible_answer_item,
        }

    # Try decoding the model results into executable function calls
    # Note: We only care about the last non-function-call message, which should fail to get decoded.
    # We don't care about the function calls in the middle of the conversation.
    # We only check if the expected answer is mentioned in the last message.
    # decode_execute returns a list of strings
    model_result_list_decoded: list[list[str]] = []
    last_unsuccessful_decoding_message = None

    for model_result_item in model_result_list[0]:
        # model_result_item is per step
        try:
            decoded_result: list[str] = handler.decode_execute(model_result_item, has_tool_call_tag=False)
            if is_empty_execute_response(decoded_result):
                last_unsuccessful_decoding_message = model_result_item
                continue
            model_result_list_decoded.append(decoded_result)
        except Exception as e:
            last_unsuccessful_decoding_message = model_result_item
            continue

    if not last_unsuccessful_decoding_message:
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": {
                "error_message": ["Cannot find the last chat message that is not a function call."],
                "error_type": "agentic:no_last_message",
            },
            "prompt": prompt_entry,
            "model_result": model_result_list,
            "model_result_decoded": model_result_list_decoded,
            "possible_answer": possible_answer_item,
        }

    # Check if the model output contains the expected answer
    accuracy_checker_result = agentic_checker(
        last_unsuccessful_decoding_message,
        possible_answer_item,
    )

    if not accuracy_checker_result["valid"]:
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": accuracy_checker_result.pop("valid"),
            "error": accuracy_checker_result,
            "prompt": prompt_entry["question"],
            "model_result_raw": model_result_list,
            "last_non_fc_message": last_unsuccessful_decoding_message,
            "possible_answer": possible_answer_item,
        }

    return {"valid": True}


def _evaluate_single_multi_turn_entry(
    handler: BaseHandler,
    test_entry_id,
    model_result_list,
    ground_truth_list,
    prompt_entry,
    model_name,
    test_category,
):
    """Helper method to process a single multi-turn entry."""
    # Remove the function doc from the score file for better readability
    if "function" in prompt_entry:
        del prompt_entry["function"]

    if type(model_result_list) != list:
        return {
            "id": test_entry_id,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": {
                "error_message": ["Error during inference phase. Model did not output a list of model responses."],
                "error_type": "multi_turn:inference_error",
            },
            "prompt": prompt_entry,
            "model_result": model_result_list,
            "possible_answer": ground_truth_list,
        }

    # Check if force-terminated during inference phase.
    # This happens when the model has retried too many times and still haven't figured out the answer.
    # When force-terminated, no further evaluation is needed. This whole entry will be failed.
    if len(model_result_list) != len(ground_truth_list):
        return {
            "id": test_entry_id,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": {
                "error_message": [
                    f"Model was force-terminated during inference phase. The length of the model result turns ({len(model_result_list)}) does not match the length of the ground truth turns ({len(ground_truth_list)})."
                ],
                "error_type": "multi_turn:force_terminated",
            },
            "prompt": prompt_entry,
            "model_result": model_result_list,
            "possible_answer": ground_truth_list,
        }

    # decode_execute returns a list of strings
    multi_turn_model_result_list_decoded: list[list[list[str]]] = []
    # Try decoding the model results into executable function calls
    for single_turn_model_result_list in model_result_list:
        single_turn_model_result_list_decoded = []
        for model_result_item in single_turn_model_result_list:
            # model_result_item is per step
            try:
                decoded_result: list[str] = handler.decode_execute(model_result_item, has_tool_call_tag=False)
                if is_empty_execute_response(decoded_result):
                    # Empty output is not considered as a valid function call
                    continue
                single_turn_model_result_list_decoded.append(decoded_result)
            except Exception as e:
                # Ignore any failed decoding and continue to the next message
                # We only care about the decoded function call, not the error message or if the model is chatting
                continue
        multi_turn_model_result_list_decoded.append(single_turn_model_result_list_decoded)

    # Check if the model output the correct function calls
    accuracy_checker_result = multi_turn_checker(
        multi_turn_model_result_list_decoded,
        ground_truth_list,
        prompt_entry,
        test_category,
        model_name,
    )

    if not accuracy_checker_result["valid"]:
        return {
            "id": test_entry_id,
            "model_name": model_name,
            "test_category": test_category,
            "valid": accuracy_checker_result.pop("valid"),
            "error": accuracy_checker_result,
            "prompt": prompt_entry,
            "model_result_raw": model_result_list,
            "model_result_decoded": multi_turn_model_result_list_decoded,
            "possible_answer": ground_truth_list,
        }

    return {"valid": True}


def _evaluate_single_relevance_entry(
    handler: BaseHandler,
    index,
    model_result_item,
    prompt_entry,
    model_name,
    test_category,
):
    """Helper method to process a single relevance/irrelevance entry."""
    contain_func_call = False
    decoded_result = None
    decode_error = None

    try:
        decoded_result = handler.decode_ast(model_result_item, language=ReturnFormat.PYTHON, has_tool_call_tag=False)
        # Decode successfully, which means the model output is in valid function call format
        contain_func_call = True
        if is_empty_output(decoded_result):
            # Empty output is not considered as a valid function call
            contain_func_call = False
    except Exception as e:
        # Decode failed, which means the model output is not in valid function call format
        contain_func_call = False
        decode_error = str(e)

    # irrelevance test means no function call outputted
    if "irrelevance" in test_category:
        success = not contain_func_call
    else:
        success = contain_func_call

    if not success:
        temp = {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": success,
            "prompt": prompt_entry,
            "model_result": model_result_item,
            "decoded_result": decoded_result,
        }
        if "irrelevance" in test_category:
            temp["error"] = ["Valid syntax. Successfully decode AST when it should not."]
            temp["error_type"] = "irrelevance_error:decoder_success"
        else:
            temp["error"] = [f"Invalid syntax. Failed to decode AST when it should have. {decode_error}"]
            temp["error_type"] = "relevance_error:decoder_failed"
        return temp

    return {"valid": True}


def _evaluate_single_ast_entry(
    handler: BaseHandler,
    index,
    model_result_item,
    possible_answer_item,
    prompt_entry,
    model_name,
    test_category,
    language: Language,
    return_format: ReturnFormat,
    has_tool_call_tag=False,
):
    """Helper method to process a single AST entry."""
    prompt_function = prompt_entry["function"]

    try:
        model_result_item_raw = model_result_item
        model_result_item = handler.decode_ast(model_result_item, return_format, has_tool_call_tag)
    except Exception as e:
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": [f"Invalid syntax. Failed to decode AST. {str(e)}"],
            "error_type": "ast_decoder:decoder_failed",
            "prompt": prompt_entry,
            "model_result_raw": model_result_item_raw,
            "possible_answer": possible_answer_item,
        }

    decoder_output_valid = is_function_calling_format_output(model_result_item)
    if not decoder_output_valid:
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": False,
            "error": ["Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."],
            "error_type": "ast_decoder:decoder_wrong_output_format",
            "prompt": prompt_entry,
            "model_result_raw": str(model_result_item_raw),
            "model_result_decoded": str(model_result_item),
            "possible_answer": possible_answer_item,
        }

    checker_result = ast_checker(
        prompt_function,
        model_result_item,
        possible_answer_item,
        language,
        test_category,
        model_name,
    )

    if not checker_result["valid"]:
        return {
            "id": index,
            "model_name": model_name,
            "test_category": test_category,
            "valid": checker_result["valid"],
            "error": checker_result["error"],
            "error_type": checker_result["error_type"],
            "prompt": prompt_entry,
            "model_result_raw": model_result_item_raw,
            "model_result_decoded": model_result_item,
            "possible_answer": possible_answer_item,
        }
    return {"valid": True}


def format_sensitivity_runner(
    handler: BaseHandler,
    model_result,
    prompt,
    possible_answer,
    model_name,
    test_category,
    score_dir,
):
    if not len(model_result) == len(prompt) == len(possible_answer):
        breakpoint()

    assert (
        len(model_result) == len(prompt) == len(possible_answer)
    ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

    # The format sensitivity tests are all single-turn tests, so we use a similar logic to the ast_file_runner to evaluate them.

    result = []
    correct_count = 0
    # Track stats per format sensitivity configuration
    config_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for i in range(len(model_result)):
        index = model_result[i]["id"]
        model_result_item = model_result[i]["result"]
        prompt_entry = prompt[i]
        possible_answer_item = possible_answer[i]["ground_truth"]

        assert ":" in index and len(index.split(":")) == 3, f"Test entry ID {index} should contain exactly two colons, since they are supposed to be the format sensitivity ids."

        format_sensitivity_config = index.split(":")[1]
        (
            return_format,
            has_tool_call_tag,
            function_doc_format,
            prompt_format,
            prompt_style,
        ) = parse_prompt_variation_params(format_sensitivity_config)

        return_format = ReturnFormat(return_format)

        entry_result = _evaluate_single_ast_entry(
            handler,
            index,
            model_result_item,
            possible_answer_item,
            prompt_entry,
            model_name,
            test_category,
            # Format sensitivity tests are all python tests
            language=Language.PYTHON,
            return_format=return_format,
            has_tool_call_tag=has_tool_call_tag,
        )

        # Update stats for this configuration
        config_stats[format_sensitivity_config]["total"] += 1
        if entry_result["valid"]:
            correct_count += 1
            config_stats[format_sensitivity_config]["correct"] += 1
        else:
            result.append(entry_result)

    # Compute accuracy per configuration
    accuracy_by_config = {
        cfg: {
            "accuracy": stats["correct"] / stats["total"],
            "correct_count": stats["correct"],
            "total_count": stats["total"],
        }
        for cfg, stats in config_stats.items()
    }

    # Calculate statistics across different prompt configurations
    config_accuracies = [v["accuracy"] for v in accuracy_by_config.values()]
    if len(config_accuracies) > 1:
        accuracy_variance = round(statistics.variance(config_accuracies) * 100**2, 2)
        accuracy_std = round(statistics.stdev(config_accuracies) * 100, 2)
        accuracy_max_delta = round((max(config_accuracies) - min(config_accuracies)) * 100, 2)
    else:
        accuracy_variance = 0.0
        accuracy_std = 0.0
        accuracy_max_delta = 0.0

    extra_header_fields = {
        "accuracy_max_delta": accuracy_max_delta,
        "accuracy_variance": accuracy_variance,
        "accuracy_std": accuracy_std,
        **accuracy_by_config,
    }

    return save_eval_results(
        result,
        correct_count,
        model_result,
        test_category,
        model_name,
        score_dir,
        extra_header_fields=extra_header_fields,
    )


def agentic_runner(
    handler: BaseHandler,
    model_result,
    prompt,
    possible_answer,
    model_name,
    test_category,
    score_dir,
):
    assert (
        len(model_result) == len(prompt) == len(possible_answer)
    ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        index = model_result[i]["id"]
        model_result_list = model_result[i]["result"]
        possible_answer_item = possible_answer[i]["ground_truth"]
        test_entry = prompt[i]

        entry_result = _evaluate_single_agentic_entry(
            handler,
            index,
            model_result_list,
            possible_answer_item,
            test_entry,
            model_name,
            test_category,
        )

        if entry_result["valid"]:
            correct_count += 1
        else:
            entry_result["inference_log"] = model_result[i].get("inference_log", "")
            result.append(entry_result)

    return save_eval_results(result, correct_count, model_result, test_category, model_name, score_dir)


def multi_turn_runner(
    handler: BaseHandler,
    model_result,
    prompt,
    possible_answer,
    model_name,
    test_category,
    score_dir,
):
    assert (
        len(model_result) == len(prompt) == len(possible_answer)
    ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        index = model_result[i]["id"]
        multi_turn_model_result_list = model_result[i]["result"]
        multi_turn_ground_truth_list = possible_answer[i]["ground_truth"]
        test_entry = prompt[i]

        entry_result = _evaluate_single_multi_turn_entry(
            handler,
            index,
            multi_turn_model_result_list,
            multi_turn_ground_truth_list,
            test_entry,
            model_name,
            test_category,
        )

        if entry_result["valid"]:
            correct_count += 1
        else:
            entry_result["inference_log"] = model_result[i].get("inference_log", "")
            result.append(entry_result)

    return save_eval_results(result, correct_count, model_result, test_category, model_name, score_dir)


def relevance_file_runner(handler: BaseHandler, model_result, prompt, model_name, test_category, score_dir):
    # This function serves for both relevance and irrelevance tests, which share the exact opposite logic.
    # If `test_category` is "irrelevance", the model is expected to output no function call.
    # No function call means either the AST decoding fails (a error message is generated) or the decoded AST does not contain any function call (such as a empty list, `[]`).
    # If `test_category` is "relevance", the model is expected to output to a function call, and empty list doesn't count as a function call.
    result = []
    correct_count = 0
    for i in range(len(model_result)):
        index = model_result[i]["id"]
        model_result_item = model_result[i]["result"]
        prompt_entry = prompt[i]

        entry_result = _evaluate_single_relevance_entry(handler, index, model_result_item, prompt_entry, model_name, test_category)

        if entry_result["valid"]:
            correct_count += 1
        else:
            result.append(entry_result)

    return save_eval_results(result, correct_count, model_result, test_category, model_name, score_dir)


def ast_file_runner(
    handler: BaseHandler,
    model_result,
    prompt,
    possible_answer,
    test_category,
    model_name,
    score_dir,
):
    assert (
        len(model_result) == len(prompt) == len(possible_answer)
    ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

    if is_java(test_category):
        language = Language.JAVA
        return_format = ReturnFormat.JAVA
    elif is_js(test_category):
        language = Language.JAVASCRIPT
        return_format = ReturnFormat.JAVASCRIPT
    else:
        language = Language.PYTHON
        return_format = ReturnFormat.PYTHON

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        index = model_result[i]["id"]
        model_result_item = model_result[i]["result"]
        prompt_entry = prompt[i]
        possible_answer_item = possible_answer[i]["ground_truth"]

        entry_result = _evaluate_single_ast_entry(
            handler,
            index,
            model_result_item,
            possible_answer_item,
            prompt_entry,
            model_name,
            test_category,
            language=language,
            return_format=return_format,
            has_tool_call_tag=False,
        )

        if entry_result["valid"]:
            correct_count += 1
        else:
            result.append(entry_result)

    return save_eval_results(result, correct_count, model_result, test_category, model_name, score_dir)


def _subset_entries_by_model_ids(
    model_result_entries: list[dict],
    prompt_entries: list[dict],
    ground_truth_entries: list[dict] = None,  # Irrelevance entries don't have ground truth
    allow_missing: bool = False,
):
    """
    Filter the prompt and ground truth entries so that its order/length matches the IDs present in `model_result`. When `allow_missing` is False, all IDs must be present; otherwise, any missing IDs are silently ignored.
    """
    if not model_result_entries:
        return [], []

    if not allow_missing and (len(model_result_entries) != len(prompt_entries)):
        raise ValueError(
            f"Length of model result ({len(model_result_entries)}) does not match length of test entries ({len(prompt_entries)}). If you intended to run only on a subset (eg. entries present in the model result), please pass the `--partial-eval` flag."
        )

    all_present_ids = {entry["id"]: entry for entry in model_result_entries}

    # Align prompt and ground-truth using the *index* of the prompt entry. Some
    # ground-truth items use a different ID format, but the order between the
    # prompt list and the ground-truth list is guaranteed to be identical. We
    # therefore keep the element at index *i* in both lists whenever the
    # prompt entry at that index has an ID present in the model results.
    filtered_prompt_entries: list[dict] = []
    filtered_ground_truth_entries: list[dict] = []
    for idx, prompt_entry in enumerate(prompt_entries):
        if prompt_entry["id"] in all_present_ids:
            filtered_prompt_entries.append(prompt_entry)
            # ground_truth_entries and prompt_entries are aligned by index.
            if ground_truth_entries is not None:
                filtered_ground_truth_entries.append(ground_truth_entries[idx])

    return filtered_prompt_entries, filtered_ground_truth_entries


#### Main runner function ####
def evaluate_task(
    test_category,
    result_dir,
    score_dir,
    model_result,
    model_name,
    handler,
    leaderboard_table,
    allow_missing: bool = True,  # If True, missing IDs are silently ignored
):
    print(f"🔍 Running test: {test_category}")

    record_cost_latency(leaderboard_table, model_name, model_result)

    # Find the corresponding prompt entries
    prompt = load_dataset_entry(test_category, include_prereq=False, include_language_specific_hint=False)

    if is_relevance_or_irrelevance(test_category):
        prompt, _ = _subset_entries_by_model_ids(model_result, prompt, None, allow_missing=allow_missing)
        assert len(prompt) == len(model_result), f"Length of prompt ({len(prompt)}) does not match length of model result ({len(model_result)}). Please check the input files for completeness."

        accuracy, total_count = relevance_file_runner(handler, model_result, prompt, model_name, test_category, score_dir)

    else:
        # Find the corresponding possible answer entries
        possible_answer = load_ground_truth_entry(test_category)
        # Sanity: prompt and ground truth should be 1:1

        prompt, possible_answer = _subset_entries_by_model_ids(model_result, prompt, possible_answer, allow_missing=allow_missing)
        assert len(prompt) == len(possible_answer), f"Length of ground truth ({len(possible_answer)}) should match prompt entries ({len(prompt)})."

        if is_format_sensitivity(test_category):
            accuracy, total_count = format_sensitivity_runner(
                handler,
                model_result,
                prompt,
                possible_answer,
                model_name,
                test_category,
                score_dir,
            )

        elif is_multi_turn(test_category):
            accuracy, total_count = multi_turn_runner(
                handler,
                model_result,
                prompt,
                possible_answer,
                model_name,
                test_category,
                score_dir,
            )

        elif is_agentic(test_category):
            accuracy, total_count = agentic_runner(
                handler,
                model_result,
                prompt,
                possible_answer,
                model_name,
                test_category,
                score_dir,
            )
        # Single turn test
        else:
            accuracy, total_count = ast_file_runner(
                handler,
                model_result,
                prompt,
                possible_answer,
                test_category,
                model_name,
                score_dir,
            )

    record_result(leaderboard_table, model_name, test_category, accuracy, total_count)

    print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy:.2%}")

    # # Weave Evals logging
    # try:
    #     # Build samples from model_result for logging
    #     samples = []
    #     for i, result_item in enumerate(model_result):
    #         sample = {
    #             "input": result_item.get("prompt", ""),
    #             "prediction": result_item.get("result", ""),
    #             "reference": None,  # BFCL doesn't have explicit references in this format
    #             "evaluation": {
    #                 "test_category": test_category,
    #                 "id": result_item.get("id", i),
    #             },
    #         }
    #         samples.append(sample)

    #     # Extract metrics
    #     metrics = {
    #         "accuracy": accuracy,
    #         "total_count": total_count,
    #         "correct_count": int(accuracy * total_count),
    #     }

    #     # Get model name for logging (convert _ back to / for display)
    #     model_name_display = model_name.replace("_", "/")

    #     WeaveEvalsController.log(
    #         dataset_name="bfcl",
    #         subset=test_category,
    #         split="test",
    #         model_backend_name="bfcl",
    #         model_name=model_name_display,
    #         scaling_method_name=None,
    #         evaluation_method_name="BFCL-Evaluation",
    #         language_penalize=False,
    #         target_lang="en",  # BFCL is primarily English
    #         samples=samples,
    #         metrics=metrics,
    #         wandb_params={},
    #     )
    # except Exception as e:
    #     print(f"[Weave] BFCL {test_category} logging skipped: {e}")

    # Wandb logging (similar to HalluLens)
    try:
        # Create summary metrics for wandb
        summary_metrics = {
            "model": model_name,
            "test_category": test_category,
            "accuracy": accuracy,
            "total_count": total_count,
            "correct_count": int(accuracy * total_count),
        }
        summary_df = pd.DataFrame([summary_metrics])
        # _log_to_wandb(summary_df, f"bfcl_{test_category}")
    except Exception as e:
        print(f"[Wandb] BFCL {test_category} summary logging skipped: {e}")

    return leaderboard_table


def _log_to_wandb(result_df: pd.DataFrame, model_task_name: str) -> None:
    """Log evaluation summary to Weights & Biases if configured."""
    from llm_eval.wandb_singleton import WandbConfigSingleton

    leaderboard_table = wandb.Table(dataframe=result_df)

    WandbConfigSingleton.collect_leaderboard_table(model_task_name, result_df)
    run = WandbConfigSingleton.get_instance().run
    run.log({model_task_name + "_leaderboard_table": leaderboard_table})


def runner(model_names, test_categories, result_dir, score_dir):
    # A dictionary to store the evaluation scores.
    # Key is model name, value is a dictionary with keys as test category
    # and values as a dictionary with accuracy and total count.
    leaderboard_table = defaultdict(dict)

    # Get a list of all entries in the folder
    entries = result_dir.iterdir()
    # Filter out the subdirectories
    subdirs = [entry for entry in entries if entry.is_dir()]

    # Traverse each subdirectory
    for subdir in tqdm(subdirs, desc="Number of models evaluated"):
        model_name = subdir.relative_to(result_dir).name
        if model_names is not None and model_name not in model_names:
            continue

        model_name_escaped = model_name.replace("_", "/")
        print(f"🦍 Model: {model_name}")

        # Find and process all result JSON files recursively in the subdirectory
        for model_result_json in subdir.rglob(RESULT_FILE_PATTERN):
            test_category = extract_test_category(model_result_json)
            if test_category not in test_categories:
                continue

            handler = get_handler(model_name_escaped)
            # We don't evaluate the following categories in the current iteration of the benchmark
            if (is_chatable(test_category) or is_sql(test_category) or is_executable(test_category) or is_memory_prereq(test_category)):
                continue

            model_result = load_file(model_result_json, sort_by_id=True)

            leaderboard_table = evaluate_task(
                test_category,
                result_dir,
                score_dir,
                model_result,
                model_name,
                handler,
                leaderboard_table,
            )

    # This function reads all the score files from local folder and updates the leaderboard table.
    # This is helpful when you only want to run the evaluation for a subset of models and test categories.
    update_leaderboard_table_with_local_score_file(leaderboard_table, score_dir)
    # Write the leaderboard table to a file
    generate_leaderboard_csv(leaderboard_table, score_dir)
    
    if model_names is not None:
        leaderboard_table = {k: v for k, v in leaderboard_table.items() if k in model_names}

    # Log overall results to wandb (per-category and overall model-wise)
    try:
        # Convert leaderboard_table to DataFrame for wandb logging (per category)
        wandb_data = []
        for model_name, model_results in leaderboard_table.items():
            for test_category, result in model_results.items():
                if isinstance(result, dict) and "accuracy" in result:
                    wandb_data.append(
                        {
                            "model": model_name,
                            "test_category": test_category,
                            "accuracy": result["accuracy"],
                            "total_count": result.get("total_count", 0),
                            "correct_count": result.get("correct_count", 0),
                        }
                    )
        if wandb_data:
            overall_df = pd.DataFrame(wandb_data)
            _log_to_wandb(overall_df, "bfcl_overall")

        # Additionally, make model-wise overall bfcl_leaderboard_table (aggregate over all categories)
        summary_rows = []
        for model_name, model_results in leaderboard_table.items():
            total_correct = 0
            total_count = 0
            for result in model_results.values():
                if isinstance(result, dict) and "accuracy" in result:
                    correct_count = result.get("correct_count")
                    if correct_count is not None:
                        total_correct += correct_count
                    else:
                        # fallback, compute from accuracy * total_count
                        try:
                            total_correct += int(result.get("accuracy", 0) * result.get("total_count", 0))
                        except Exception:
                            pass
                    total_count += result.get("total_count", 0)
            overall_accuracy = (total_correct / total_count) if total_count > 0 else 0.0
            summary_rows.append(
                {
                    "model_name": model_name,
                    "AVG": overall_accuracy,
                    "total_count": total_count,
                    "correct_count": total_correct,
                }
            )
        if summary_rows:
            leaderboard_df = pd.DataFrame(summary_rows)
            _log_to_wandb(leaderboard_df, "bfcl")
    except Exception as e:
        print(f"[Wandb] BFCL overall logging skipped: {e}")

    # --- Weave Evals logging across all models/categories (aggregate and per-sample) ---
    try:
        from llm_eval.wandb_controller import WeaveEvalsController

        # First, log aggregate/summary metrics per model across all categories
        for model_name, model_results in leaderboard_table.items():
            total_correct = 0
            total_count = 0
            for result in model_results.values():
                if isinstance(result, dict) and "accuracy" in result:
                    correct_count = result.get("correct_count")
                    if correct_count is not None:
                        total_correct += correct_count
                    else:
                        # fallback: compute from accuracy * total_count
                        try:
                            total_correct += int(result.get("accuracy", 0) * result.get("total_count", 0))
                        except Exception:
                            pass
                    total_count += result.get("total_count", 0)
            overall_accuracy = (total_correct / total_count) if total_count > 0 else 0.0
            summary_metrics = {
                "AVG": overall_accuracy,
                "total_count": total_count,
                "correct_count": total_correct,
            }
            model_name_display = model_name.replace("_", "/")
            # Log overall metrics for this model (no samples, just summary)
            WeaveEvalsController.log(
                dataset_name="bfcl",
                subset="overall",  # Special value to indicate overall summary
                split="test",
                model_backend_name="bfcl",
                model_name=model_name_display,
                scaling_method_name=None,
                evaluation_method_name="BFCL-Evaluation",
                language_penalize=False,
                target_lang="en",
                samples=[],
                metrics=summary_metrics,
                wandb_params={},
            )
    except Exception as e:
        print(f"[Weave] BFCL all logging skipped: {e}")

    return leaderboard_table
