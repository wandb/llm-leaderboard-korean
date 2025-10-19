import re
from typing import Any, Dict, List, Optional, Union
import os
import json

from . import register_dataset
from .base import BaseDataset


@register_dataset("kobalt_700")
class KoBALT700Dataset(BaseDataset):
    """
    KoBALT-700 Dataset Class for data sourced from 'HAERAE-HUB/KoSimpleEval'.

    This loader handles the 'KoBALT-700' subset, which features multiple-choice questions
    with lettered options up to 'J'.

    - The 'question' column, containing the full problem text with choices, is used as the 'input'.
    - The 'gold' column, a single letter (e.g., 'H'), is converted to a string like "(H)" for the 'reference'.
    - A fixed list of 'options' from "(A)" to "(J)" is provided for all samples to support log-probability-based evaluations.
    - The 'category' column (e.g., '의미론') is used as '_subset_name' for per-subject analysis.

    Usage example:
        ds = KoBALT700Dataset(split="test")
        data = ds.load()
        # data -> [
        #   {
        #     "input": "지문: 현진, 수빈아, 혹시 지금 시간 돼? 다음주 회의 관련해서 부탁할 게 있어서... ",
        #     "reference": "(H)",
        #     "options": ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)"],
        #     "_subset_name": "의미론"
        #   }, ...
        # ]
    """
    # Define a fixed list of options from (A) to (J)
    DEFAULT_OPTIONS = [chr(ord('A') + i) for i in range(10)]

    def __init__(
        self,
        dataset_name: str = "kobalt_700",
        subset: Optional[Union[str, List[str]]] = None,
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs,
    ):
        if base_prompt_template is None:
            base_prompt_template = (
                "다음은 언어학 관련 객관식 문제입니다. 제시된 지문과 질문, 그리고 선택지를 주의 깊게 읽고, "
                "\"정답은: X\"라고 결론지으십시오. 여기서 X는 A, B, C, D, E, F, G, H, I, J 중 하나입니다.\n\n"
                "질문: {question}\n"
            )
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            subset=subset,
            base_prompt_template=base_prompt_template,
            **kwargs,
        )

    def _normalize_split(self, split: str) -> str:
        s = (split or "").lower()
        if s in ("train", "training"):
            return "train"
        if s in ("dev", "validation", "valid", "val"):
            return "dev"
        return "test"

    def _download_and_load(self) -> Dict[str, Any]:
        from llm_eval.wandb_singleton import WandbConfigSingleton
        artifact_dir = WandbConfigSingleton.download_artifact(self.dataset_name)
        file_path = os.path.join(artifact_dir, "kobalt_700.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"kobalt_700.json not found in artifact: {artifact_dir}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid kobalt_700.json format: expected an object keyed by splits")
        return data

    def load(self) -> List[Dict[str, Any]]:
        """
        아티팩트의 kobalt_700.json을 로드하여 표준 포맷으로 변환합니다.
        {split: {subset: [...]}} 구조를 지원하며, subset None/list/str 모두 허용합니다.
        """
        raw = self._download_and_load()
        split_key = self._normalize_split(self.split)
        split_data = raw.get(split_key, {})
        if not isinstance(split_data, dict):
            raise ValueError(
                f"Invalid '{split_key}' split format: expected an object keyed by subsets"
            )

        if self.subset is None:
            subset_list = list(split_data.keys())
        elif isinstance(self.subset, (list, tuple)):
            subset_list = list(self.subset)
        else:
            subset_list = [self.subset]

        results: List[Dict[str, Any]] = []
        for subset_name in subset_list:
            items = split_data.get(subset_name, [])
            if not isinstance(items, list):
                continue
            results.extend(self._convert_to_list(items, subset_name=subset_name))
        return results

    def _convert_to_list(
        self, items, subset_name: str
    ) -> List[Dict[str, Any]]:
        """
        Converts the HuggingFace Dataset object to a list in the HRET standard format.
        """
        processed_list = []
        for item in items:
            question_text = item.get("question", "").strip()

            final_input = (
                self.base_prompt_template.format(question=question_text)
                if self.base_prompt_template
                else question_text
            )

            # Convert the gold letter (e.g., 'H') to the format "(H)".
            gold_letter = str(item.get("gold", "")).strip()
            reference = gold_letter if gold_letter else ""

            processed_list.append(
                {
                    "input": final_input,
                    "reference": reference,
                    "options": self.DEFAULT_OPTIONS,  # Use the fixed list of options
                    "_subset_name": subset_name,
                }
            )
            if getattr(self, "dev_mode", False) and len(processed_list) >= 10:
                break
            if getattr(self, "limit", None) and len(processed_list) >= self.limit:
                break
        return processed_list

    def get_raw_samples(self) -> Any:
        return self._download_and_load()

    def info(self) -> Dict[str, Any]:
        """Returns metadata about the dataset."""
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self._normalize_split(self.split),
            "description": "KoBALT-700 dataset loaded from W&B artifact with fixed options A-J.",
            "evaluation_only": None,
        }