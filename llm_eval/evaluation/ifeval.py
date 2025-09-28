from .base import BaseEvaluator
from . import register_evaluator
from .ifeval_ko.utils import (
    InputExample,
    test_instruction_following_strict,
    test_instruction_following_loose,
)


@register_evaluator("ifeval_strict")
class IFEvalStrictEvaluator(BaseEvaluator):
    name = "ifeval_strict"

    def evaluate_predictions(self, samples):
        if not samples:
            return {
                "prompt_level_strict_accuracy": 0.0,
                "instruction_level_strict_accuracy": 0.0,
            }

        prompt_correct = 0
        total_prompts = len(samples)
        inst_correct = 0
        inst_total = 0

        for sample in samples:
            metadata = sample.get("metadata", {})
            doc = {
                "key": metadata.get("key", -1),
                "instruction_id_list": metadata.get("instruction_id_list", []),
                "prompt": metadata.get("prompt", sample.get("input", "")),
                "kwargs": metadata.get("kwargs", []),
            }

            input_example = InputExample(**doc)
            response = sample.get("prediction", "") or ""
            response = str(response)

            out = test_instruction_following_strict(input_example, response)

            prompt_correct += 1 if out.follow_all_instructions else 0
            inst_correct += sum(out.follow_instruction_list)
            inst_total += len(out.follow_instruction_list)

            sample["evaluation"] = {
                "prompt_level_strict_acc": out.follow_all_instructions,
                "inst_level_strict_acc": out.follow_instruction_list,
            }

        metrics = {
            "prompt_level_strict_accuracy": prompt_correct / total_prompts,
        }

        if inst_total > 0:
            metrics["instruction_level_strict_accuracy"] = inst_correct / inst_total
        else:
            metrics["instruction_level_strict_accuracy"] = 0.0

        return metrics


@register_evaluator("ifeval_loose")
class IFEvalLooseEvaluator(BaseEvaluator):
    name = "ifeval_loose"

    def evaluate_predictions(self, samples):
        if not samples:
            return {
                "prompt_level_loose_accuracy": 0.0,
                "instruction_level_loose_accuracy": 0.0,
            }

        prompt_correct = 0
        total_prompts = len(samples)
        inst_correct = 0
        inst_total = 0

        for sample in samples:
            metadata = sample.get("metadata", {})
            doc = {
                "key": metadata.get("key", -1),
                "instruction_id_list": metadata.get("instruction_id_list", []),
                "prompt": metadata.get("prompt", sample.get("input", "")),
                "kwargs": metadata.get("kwargs", []),
            }

            input_example = InputExample(**doc)
            response = sample.get("prediction", "") or ""
            response = str(response)

            out = test_instruction_following_loose(input_example, response)

            prompt_correct += 1 if out.follow_all_instructions else 0
            inst_correct += sum(out.follow_instruction_list)
            inst_total += len(out.follow_instruction_list)

            sample["evaluation"] = sample.get("evaluation", {})
            sample["evaluation"].update({
                "prompt_level_loose_acc": out.follow_all_instructions,
                "inst_level_loose_acc": out.follow_instruction_list,
            })

        metrics = {
            "prompt_level_loose_accuracy": prompt_correct / total_prompts,
        }

        if inst_total > 0:
            metrics["instruction_level_loose_accuracy"] = inst_correct / inst_total
        else:
            metrics["instruction_level_loose_accuracy"] = 0.0

        return metrics
