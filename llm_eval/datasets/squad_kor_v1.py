from typing import List, Dict, Any, Optional, Union
import os
import json
from .base import BaseDataset
from . import register_dataset

@register_dataset("squad_kor_v1")
class SQuADKorV1(BaseDataset):
    """
    SQuAD Kor V1 dataset class (KorQuAD/squad_kor_v1 on Hugging Face).

    - subset: None -> default config
              list[str] -> load and merge multiple configs
              str -> load only the specified config
    - Each sample contains:
        {
          "input": formatted prompt built from context and question,
          "reference": gold short answer text,
          "_subset_name": subset identifier
        }

    Example usage:
        ds = SQuADKorV1(
            dataset_name="KorQuAD/squad_kor_v1",
            split="validation"
        )
        data = ds.load()
        # data -> [{"input": "...", "reference": "...", "_subset_name": ...}, ...]
    """
    def __init__(
        self, 
        dataset_name: str = "squad_kor_v1",
        subset: Optional[Union[str, list]] = None,
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs
    ):
        if base_prompt_template is None:
            base_prompt_template = (
                "다음 지문과 질문을 읽고, 질문에 대한 정답을 한 단어 또는 짧은 구로 추출해 주세요."
                "\n지문: {context}\n\n질문: {question}\n정답:"
            )
        super().__init__(dataset_name, split=split, subset=subset, base_prompt_template=base_prompt_template, **kwargs)

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
        file_path = os.path.join(artifact_dir, "squad_kor_v1.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"squad_kor_v1.json not found in artifact: {artifact_dir}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid squad_kor_v1.json format: expected an object keyed by splits")
        return data

    def load(self) -> List[Dict[str, Any]]:
        """
        W&B artifact의 squad_kor_v1.json을 로드하여 표준 포맷으로 변환합니다.
        반환 형식: [{"input": ..., "reference": ..., "_subset_name": ...}, ...]
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
            for item in items:
                context = item.get("context", "")
                question = item.get("question", "")

                if self.base_prompt_template:
                    prompt = self.base_prompt_template.format(context=context, question=question)
                else:
                    prompt = f"{context}\n\n{question}"

                answers = item.get("answers", {}) or {}
                answer_texts = answers.get("text", []) or []
                reference = ""
                for a in answer_texts:
                    if isinstance(a, str) and a.strip():
                        reference = a.strip()
                        break

                results.append({
                    "input": prompt,
                    "reference": reference,
                    "_subset_name": subset_name,
                })

                if getattr(self, "dev_mode", False) and len(results) >= 10:
                    break
                if getattr(self, "limit", None) and len(results) >= self.limit:
                    break

        return results

    # HF Hub 변환 유틸은 제거되었습니다.

    def get_raw_samples(self) -> Any:
        return self._download_and_load()

    def info(self) -> Dict[str, Any]:
        """
        Returns metadata information about the dataset.
        """
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self._normalize_split(self.split),
            "description": (
                "SQuAD Kor v1.0 extractive QA loaded from W&B artifact."
            ),
            "evaluation_only": None
        }
