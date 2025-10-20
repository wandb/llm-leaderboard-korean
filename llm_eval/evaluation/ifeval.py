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

    def evaluate_predictions(self, subsets, samples):
        if not samples:
            return {
                "prompt_level_strict_accuracy": 0.0,
                "instruction_level_strict_accuracy": 0.0,
            }

        prompt_correct = 0
        total_prompts = len(samples)
        inst_correct = 0
        inst_total = 0

        # subset 별 집계를 위한 통계 구조
        # 요청된 subsets 기준 초기화
        subset_stats = {}
        if subsets:
            for s in subsets:
                subset_stats[s] = {
                    "prompt_correct": 0,
                    "total_prompts": 0,
                    "inst_correct": 0,
                    "inst_total": 0,
                }

        for sample in samples:
            subset_name = sample.get("_subset_name")
            if subset_name and subset_name not in subset_stats:
                subset_stats[subset_name] = {
                    "prompt_correct": 0,
                    "total_prompts": 0,
                    "inst_correct": 0,
                    "inst_total": 0,
                }

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

            # 전체 통계 업데이트
            prompt_correct += 1 if out.follow_all_instructions else 0
            inst_correct += sum(out.follow_instruction_list)
            inst_total += len(out.follow_instruction_list)

            # subset 통계 업데이트
            if subset_name:
                subset_stats[subset_name]["prompt_correct"] += 1 if out.follow_all_instructions else 0
                subset_stats[subset_name]["total_prompts"] += 1
                subset_stats[subset_name]["inst_correct"] += sum(out.follow_instruction_list)
                subset_stats[subset_name]["inst_total"] += len(out.follow_instruction_list)

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

        metrics["AVG"] = (metrics["prompt_level_strict_accuracy"] + metrics["instruction_level_strict_accuracy"]) / 2

        # subsets 전달 여부에 따라 분기
        if subsets:
            for sname, st in subset_stats.items():
                s_prompt_acc = (st["prompt_correct"] / st["total_prompts"]) if st["total_prompts"] > 0 else 0.0
                s_inst_acc = (st["inst_correct"] / st["inst_total"]) if st["inst_total"] > 0 else 0.0
                s_avg = (s_prompt_acc + s_inst_acc) / 2
                metrics[f"{sname}/strict_accuracy"] = s_avg

        return metrics


@register_evaluator("ifeval_loose")
class IFEvalLooseEvaluator(BaseEvaluator):
    name = "ifeval_loose"

    def evaluate_predictions(self, subsets, samples):
        if not samples:
            return {
                "prompt_level_loose_accuracy": 0.0,
                "instruction_level_loose_accuracy": 0.0,
            }

        prompt_correct = 0
        total_prompts = len(samples)
        inst_correct = 0
        inst_total = 0

        # subset 별 집계를 위한 통계 구조
        # 요청된 subsets 기준 초기화
        subset_stats = {}
        if subsets:
            for s in subsets:
                subset_stats[s] = {
                    "prompt_correct": 0,
                    "total_prompts": 0,
                    "inst_correct": 0,
                    "inst_total": 0,
                }

        for sample in samples:
            subset_name = sample.get("_subset_name")
            if subset_name and subset_name not in subset_stats:
                subset_stats[subset_name] = {
                    "prompt_correct": 0,
                    "total_prompts": 0,
                    "inst_correct": 0,
                    "inst_total": 0,
                }
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

            # 전체 통계 업데이트
            prompt_correct += 1 if out.follow_all_instructions else 0
            inst_correct += sum(out.follow_instruction_list)
            inst_total += len(out.follow_instruction_list)

            # subset 통계 업데이트
            if subset_name:
                subset_stats[subset_name]["prompt_correct"] += 1 if out.follow_all_instructions else 0
                subset_stats[subset_name]["total_prompts"] += 1
                subset_stats[subset_name]["inst_correct"] += sum(out.follow_instruction_list)
                subset_stats[subset_name]["inst_total"] += len(out.follow_instruction_list)

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

        metrics["final_score"] = (metrics["prompt_level_loose_accuracy"] + metrics["instruction_level_loose_accuracy"]) / 2

        # subsets 전달 여부에 따라 분기
        if subsets:
            for sname, st in subset_stats.items():
                s_prompt_acc = (st["prompt_correct"] / st["total_prompts"]) if st["total_prompts"] > 0 else 0.0
                s_inst_acc = (st["inst_correct"] / st["inst_total"]) if st["inst_total"] > 0 else 0.0
                s_avg = (s_prompt_acc + s_inst_acc) / 2
                metrics[f"{sname}/loose_accuracy"] = s_avg

        return metrics
