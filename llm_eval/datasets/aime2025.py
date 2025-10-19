from typing import List, Dict, Any, Optional, Union

from datasets import load_dataset

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
        dataset_name: str = "opencompass/AIME2025",
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

    def _load_hf_split(self, config_name: str):
        """
        Attempt to load the desired split with light fallbacks because some hubs
        publish only a single split (often 'train' or 'test').
        """
        preferred_order = [self.split, "test", "validation", "train"]
        last_err = None
        for sp in preferred_order:
            try:
                return load_dataset(self.dataset_name, config_name, split=sp, **self.kwargs)
            except Exception as e:  # keep trying
                last_err = e
                continue
        # If everything failed, re-raise the last error
        raise last_err if last_err else RuntimeError("Failed to load dataset split")

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

        all_items: List[Dict[str, Any]] = []
        for sub in target_subsets:
            hf_ds = self._load_hf_split(sub)
            all_items.extend(self._convert_to_list(hf_ds, subset_name=sub))
        return all_items

    def _convert_to_list(self, hf_dataset, subset_name: str) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        # Default template if none provided (re-derive to keep clarity)
        default_template = (
            "Solve the following AIME-style problem. Briefly show your reasoning, "
            "then end with a single line in the form 'Answer: X'.\n\n{question}"
        )
        template = self.base_prompt_template or default_template

        for item in hf_dataset:
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
            if getattr(self, "dev_mode", False) and len(processed) >= 10:
                break
        return processed

    def get_raw_samples(self) -> Any:
        # Return HF datasets per subset
        if self.subset is None:
            subsets = ["AIME2025-I", "AIME2025-II"]
        elif isinstance(self.subset, list):
            subsets = [self._normalize_subset(s) for s in self.subset]
        else:
            subsets = [self._normalize_subset(self.subset)]

        raw = []
        for sub in subsets:
            raw.append(self._load_hf_split(sub))
        return raw if len(raw) > 1 else raw[0]

    def info(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": (
                "AIME2025 benchmark (OpenCompass). Two subsets: AIME2025-I and AIME2025-II. "
                "Each sample is a (question, answer) pair for exact-match evaluation."
            ),
            # allow both exact string match or math equivalence
            "evaluation_only": None,
        }
