from __future__ import annotations

from typing import Any, Dict, List, Optional, Iterable, Union
import os
import json

from .base import BaseDataset
from . import register_dataset
from llm_eval.utils.logging import get_logger

logger = get_logger(name="arc_agi", level="INFO")

# Lightweight aliases to keep type hints simple in artifact-only mode
Dataset = Any
DatasetDict = Any

# DEFAULT_ARC_PROMPT_TEMPLATE = (
#     # "You are given ARC tasks. Grids use digits 0-9 only.\n"
#     # "Return only the final grid as comma-separated rows, with no extra text.\n\n"
# )
DEFAULT_ARC_PROMPT_TEMPLATE = """
ARC 문제(Task)가 주어집니다. 그리드는 0~9 숫자만 사용합니다.
문제를 해결하세요. 사고 과정은 내부에서만 진행하고, 절대 외부로 드러내지 마세요.
- 출력 형식 (OUTPUT FORMAT):
  - 숫자와 콤마로 이루어진 그리드만 출력하세요.
  - 출력 전후에 어떤 텍스트도 추가하지 마세요.
  - 따옴표, 코드 블록, 설명을 절대 포함하지 마세요.
**숫자·콤마·줄바꿈으로 이루어진 그리드 이외의 문자가 출력되면 오답으로 간주됩니다.**
**그리드만 그대로 출력하세요.**
{examples}\n\n
{query_block}\n
"""


Grid = List[List[int]]


def _serialize_grid(g: Grid) -> str:
    return "\n".join(",".join(str(c) for c in row) for row in g)


def _build_examples(train_pairs: List[Dict[str, Any]]) -> str:
    if not train_pairs:
        return ""
    blocks = []
    for i, pair in enumerate(train_pairs, start=1):
        tin = _serialize_grid(pair["input"])
        tout = _serialize_grid(pair["output"])
        blocks.append(f"[Example {i}]\n[Input]\n{tin}\n[Output]\n{tout}\n")
    return "\n".join(blocks).strip()


def _build_query_block(test_input_grid: Grid) -> str:
    return "[Query Input]\n" + _serialize_grid(test_input_grid)


def _task_to_prompt(
    task: Dict[str, Any],
    test_input_grid: Grid,
    base_prompt_template: str,
) -> str:
    examples = _build_examples(task.get("train", []))
    query_block = _build_query_block(test_input_grid)
    template = base_prompt_template or DEFAULT_ARC_PROMPT_TEMPLATE
    if not examples:
        template = template.replace("{examples}\n", "")
    return template.format(examples=examples, query_block=query_block)


@register_dataset("arc_agi")
class ARCAGIDataset(BaseDataset):
    """
    ARC-AGI loader (artifact-only):
      - Loads local artifact file arc_agi.json downloaded via WandB artifact utilities.

    Expected per-task JSON format:
    {
      "train": [ {"input": [[...]], "output": [[...]]}, ... ],
      "test":  [ {"input": [[...]], "output": [[...]]}, ... ]
    }
    """

    def __init__(
        self,
        dataset_name: str = "arc_agi",
        split: str = "evaluation",
        subset: Optional[Union[str, list]] = "default",
        base_prompt_template: Optional[str] = DEFAULT_ARC_PROMPT_TEMPLATE,
        **kwargs: Any
    ):
        super().__init__(
            dataset_name,
            split=split,
            subset=subset,
            base_prompt_template=base_prompt_template,
            **kwargs
        )

    def _normalize_split(self, split: str) -> str:
        s = (split or "").lower()
        if s in ("training", "train"):
            return "training"
        if s in ("evaluation", "test", "eval", "evaluate"):
            return "evaluation"
        return "evaluation"

    def load(self) -> List[Dict[str, Any]]:
        if self.subset is None:
            subset_list = ["default"]
        elif isinstance(self.subset, (list, tuple)):
            subset_list = list(self.subset)
        else:
            subset_list = [self.subset]
        raw = self._download_and_load()
        split_key = self._normalize_split(self.split)
        split_data = raw.get(split_key, {})
        if not isinstance(split_data, dict):
            raise ValueError(
                f"Invalid '{split_key}' split format: expected an object keyed by subsets"
            )

        processed_list: List[Dict[str, Any]] = []
        for sub in subset_list:
            items = split_data.get(sub, [])
            if not isinstance(items, list):
                continue
            processed_list.extend(self._convert_to_list(items, subset_name=sub))
        return processed_list

    def _convert_to_list(self, items, subset_name: str) -> List[Dict[str, Any]]:
        samples = []

        for task in items:
            prompt = _task_to_prompt(
                task,
                task["test"][0]["input"],
                base_prompt_template=self.base_prompt_template or DEFAULT_ARC_PROMPT_TEMPLATE,
            )
            reference = _serialize_grid(task["test"][0]["output"])
            samples.append(
                {
                    "input": prompt,
                    "reference": reference,
                    "_subset_name": subset_name,
                }
            )
            if self.dev and len(samples) >= self.limit:
                break
            if len(samples) >= self.num_samples:
                break

        return samples

    def get_raw_samples(self) -> Any:
        return self._download_and_load()

    def _download_and_load(self) -> Dict[str, Any]:
        from llm_eval.wandb_singleton import WandbConfigSingleton
        artifact_dir = WandbConfigSingleton.download_artifact(self.dataset_name)
        file_path = os.path.join(artifact_dir, f"{self.dataset_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{self.dataset_name}.json not found in artifact: {artifact_dir}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid {self.dataset_name}.json format: expected an object keyed by splits")
        return data

    def info(self) -> Dict[str, Any]:
        base = {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self._normalize_split(self.split),
            "description": f"{self.dataset_name.upper()} dataset loaded from W&B artifact.",
            "evaluation_only": None,
        }
        return base
