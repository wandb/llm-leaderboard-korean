from typing import List, Dict, Any, Optional, Union
import os
import json

from .base import BaseDataset
from . import register_dataset


@register_dataset("aime2025")
class AIME2025Dataset(BaseDataset):
    """
    OpenCompass AIME2025 dataset loader.

    Schema per sample:
      - input: formatted prompt text to feed into the model (question with optional template)
      - reference: ground-truth short answer string
      - _subset_name: which subset (AIME2025-I or AIME2025-II) this item belongs to

    Subsets:
      - "AIME2025-I" and "AIME2025-II" (default loads both)
      - Also accepts shorthand: "I" -> "AIME2025-I", "II" -> "AIME2025-II"

    Notes:
      - Questions may include LaTeX; we pass through as-is.
      - Designed mainly for evaluation (exact match). You can use either
        string_match (exact match with simple normalization) or math_match
        (math-verify based) evaluators from this toolkit.
    """

    def __init__(
        self,
        dataset_name: str = "aime2025",
        subset: Optional[Union[str, List[str]]] = None,
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs,
    ):
        # Default prompt template that guides the model to end with "Answer: X"
        if base_prompt_template is None:
            base_prompt_template = (
                "Solve the following AIME-style problem. Briefly show your reasoning, "
                "then end with a single line in the form 'Answer: X'.\n\n{question}"
            )
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            subset=subset,
            base_prompt_template=base_prompt_template,
            **kwargs,
        )

    # --- helpers ---
    def _normalize_split(self, split: str) -> str:
        s = (split or "").lower()
        if s in ("train", "training"):
            return "train"
        if s in ("dev", "validation", "valid", "val"):
            return "dev"
        return "test"
    def _normalize_subset(self, s: str) -> str:
        su = str(s).strip()
        # accept simple forms
        if su.lower() in {"i", "1"}:
            return "AIME2025-I"
        if su.lower() in {"ii", "2"}:
            return "AIME2025-II"
        # accept already full names (case-insensitive)
        if su.upper() in {"AIME2025-I", "AIME2025-II"}:
            return su.upper()
        # fallback: return as-is to let HF raise a helpful error
        return su

    def _download_and_load(self) -> Dict[str, Any]:
        from llm_eval.wandb_singleton import WandbConfigSingleton
        artifact_dir = WandbConfigSingleton.download_artifact(self.dataset_name)
        file_path = os.path.join(artifact_dir, "aime2025.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"aime2025.json not found in artifact: {artifact_dir}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid aime2025.json format: expected an object keyed by splits")
        return data

    # --- core API ---
    def load(self) -> List[Dict[str, Any]]:
        # Normalize subset parameter
        target_subsets: List[str]
        if self.subset is None:
            target_subsets = ["AIME2025-I", "AIME2025-II"]
        elif isinstance(self.subset, list):
            target_subsets = [self._normalize_subset(s) for s in self.subset]
        else:
            target_subsets = [self._normalize_subset(self.subset)]

        raw = self._download_and_load()
        split_key = self._normalize_split(self.split)
        split_data = raw.get(split_key, {})
        if not isinstance(split_data, dict):
            raise ValueError(
                f"Invalid '{split_key}' split format: expected an object keyed by subsets"
            )

        all_items: List[Dict[str, Any]] = []
        for sub in target_subsets:
            items = split_data.get(sub, [])
            if not isinstance(items, list):
                continue
            all_items.extend(self._convert_to_list(items, subset_name=sub))
        return all_items

    def _convert_to_list(self, items, subset_name: str) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        # Default template if none provided (re-derive to keep clarity)
        default_template = (
            "Solve the following AIME-style problem. Briefly show your reasoning, "
            "then end with a single line in the form 'Answer: X'.\n\n{question}"
        )
        template = self.base_prompt_template or default_template

        for item in items:
            question = str(item.get("question", "")).strip()
            reference = str(item.get("answer", "")).strip()
            formatted = template.format(question=question)
            processed.append(
                {
                    "input": formatted,
                    "reference": reference,
                    "_subset_name": subset_name,
                }
            )
            if getattr(self, "dev_mode", False) and len(processed) >= 2:
                break
        return processed

    def get_raw_samples(self) -> Any:
        return self._download_and_load()

    def info(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self._normalize_split(self.split),
            "description": (
                "AIME2025 benchmark loaded from W&B artifact. "
                "Two subsets: AIME2025-I and AIME2025-II. Each sample is a (question, answer) pair for exact-match evaluation."
            ),
            # allow both exact string match or math equivalence
            "evaluation_only": None,
        }
