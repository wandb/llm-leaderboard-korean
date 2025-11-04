from typing import List, Dict, Any, Optional
import os
import json
from .base import BaseDataset
from . import register_dataset

@register_dataset("ifeval_ko")
class IFEvalKoDataset(BaseDataset):
    """
    IFEval-Ko dataset loader.

    - Source: allganize/IFEval-Ko (HF Datasets)
    - Schema (summary):
        - key: int
        - prompt: str  (instruction text presented to the model)
        - instruction_id_list: List[str]  (IDs of applied constraints/rules)
        - kwargs: List[dict]  (parameters for each constraint)

    Return format:
        [
            {
                "input": str,            # original prompt or template-formatted prompt
                "reference": str,        # IFEval provides no gold answer string; keep empty
                "metadata": {            # preserve auxiliary fields
                    "key": int,
                    "instruction_id_list": List[str],
                    "kwargs": List[dict]
                }
            },
            ...
        ]

    참고: IFEval-Ko 소개 및 사용법은 HF 페이지를 참조.
    """

    def __init__(
        self,
        dataset_name: str = "ifeval_ko",
        subset: str = "default",
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs
    ):
        super().__init__(dataset_name, subset=subset, split=split, base_prompt_template=base_prompt_template, **kwargs)

    def _normalize_split(self, split: str) -> str:
        s = (split or "").lower()
        if s in ("train", "training"):
            return "train"
        if s in ("dev", "validation", "valid", "val"):
            return "dev"
        return "test"

    def _download_and_load(self) -> Dict[str, Any]:
        from llm_eval.wandb_singleton import WandbConfigSingleton
        ifeval_ko_artifact_dir = WandbConfigSingleton.download_artifact(self.dataset_name)
        ifeval_ko_path = os.path.join(ifeval_ko_artifact_dir, "ifeval_ko.json")
        if not os.path.exists(ifeval_ko_path):
            raise FileNotFoundError(f"ifeval_ko.json not found in artifact: {ifeval_ko_artifact_dir}")

        with open(ifeval_ko_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, dict):
            raise ValueError("Invalid ifeval_ko.json format: expected an object keyed by splits")
        return raw_data

    def load(self) -> List[Dict[str, Any]]:
        """
        W&B artifact의 ifeval_ko.json을 로드하여 표준 포맷으로 변환합니다.
        반환 형식: [{"input": str, "reference": "", "metadata": {...}}, ...]
        """
        raw = self._download_and_load()
        split_key = self._normalize_split(self.split)

        split_obj = raw.get(split_key, {})
            # subset 별로 구성된 경우 (예: {"test": {"default": [...]}})
        if isinstance(self.subset, (list, tuple)):
            merged: List[Dict[str, Any]] = []
            for subset_name in self.subset:
                items = split_obj.get(subset_name, [])
                if isinstance(items, list):
                    merged.extend(items)
            samples = merged
        else:
            samples = split_obj.get(self.subset, [])

        if not isinstance(samples, list):
            raise ValueError(f"Invalid '{split_key}' split format: expected a list or subset lists")

        results: List[Dict[str, Any]] = []

        for item in samples:
            prompt = str(item.get("prompt", "")).strip()
            formatted_input = (
                self.base_prompt_template.format(prompt=prompt)
                if self.base_prompt_template
                else prompt
            )

            results.append({
                "input": formatted_input,
                "reference": "",
                "metadata": {
                    "key": item.get("key"),
                    "prompt": prompt,
                    "instruction_id_list": item.get("instruction_id_list", []),
                    "kwargs": item.get("kwargs", []),
                },
            })

            if self.dev and len(results) >= self.limit:
                break
            if len(results) >= self.num_samples:
                break
        return results

    def info(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "split": self._normalize_split(self.split),
            "subset": self.subset,
            "description": (
                "IFEval-Ko: Korean instruction-following benchmark loaded from W&B artifact. "
                "Fields: prompt, instruction_id_list, kwargs."
            ),
        }

if __name__ == "__main__":
    dataset = IFEvalKoDataset()
    dt = dataset.load()
    dataset.info()
    # print statistics
    print(f"Number of samples: {len(dataset.load())}")
    print(f"Number of subsets: {len(dataset.subset)}")
    print(f"Number of splits: {len(dataset.split)}")
    
    print(dt[:3])