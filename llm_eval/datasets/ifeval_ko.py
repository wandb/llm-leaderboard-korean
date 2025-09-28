from typing import List, Dict, Any, Optional
from datasets import load_dataset
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
        dataset_name: str = "allganize/IFEval-Ko",
        split: str = "train",
        base_prompt_template: Optional[str] = None,
        **kwargs
    ):
        super().__init__(dataset_name, split=split, base_prompt_template=base_prompt_template, **kwargs)

    def load(self) -> List[Dict[str, Any]]:
        """
        Load HF dataset and convert to standardized format:
        [{"input": ..., "reference": "", "metadata": {...}}, ...]
        """
        raw_data = load_dataset(self.dataset_name, split=self.split, **self.kwargs)
        result: List[Dict[str, Any]] = []

        for item in raw_data:
            prompt = str(item.get("prompt", "")).strip()
            formatted_input = (
                self.base_prompt_template.format(prompt=prompt)
                if self.base_prompt_template
                else prompt
            )

            sample = {
                "input": formatted_input,
                "reference": "",
                "metadata": {
                    "key": item.get("key"),
                    "prompt": prompt,
                    "instruction_id_list": item.get("instruction_id_list", []),
                    "kwargs": item.get("kwargs", []),
                },
            }
            result.append(sample)

        return result

    def get_raw_samples(self) -> Any:
        """
        Return the raw HF Dataset object.
        """
        return load_dataset(self.dataset_name, split=self.split, **self.kwargs)

    def info(self) -> Dict[str, Any]:
        """
        Return dataset metadata.
        """
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "description": (
                "IFEval-Ko: Korean instruction-following benchmark. "
                "Fields: prompt, instruction_id_list, kwargs."
            )
        }
