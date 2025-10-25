from typing import List, Dict, Any, Optional, Union
import os
import json
from .base import BaseDataset
from . import register_dataset

@register_dataset("haerae_bench_v1")
class HaeraeDatasetV1(BaseDataset):
    """
    Haerae Bench dataset class.

    - subset: None -> load all data
              list[str] -> load and merge only the specified multiple subsets
              str -> load only the specified subset
    - Each sample includes an 'options' field (if not present in the original data, defaults to ["(A)", "(B)", "(C)", "(D)", "(E)"])
    - Adds a '_subset_name' field to indicate which subtask the data was loaded from,
      so that evaluation can compute scores separately for each subset.

    Example usage:
        ds = HaeraeDataset(
            dataset_name="haerae_bench_v1",
            subset=["general_knowledge", "history"],  # multiple subsets
            split="test"
        )
        data = ds.load()
        # data -> [
        #   {
        #     "input": "...",
        #     "reference": "...",
        #     "options": ["(A)", "(B)", "(C)", "(D)", "(E)"], # required for log-prob calculation
        #     "_subset_name": "general_knowledge"
        #   }, ...
        # ]
    """
    def __init__(
        self, 
        dataset_name: str = "haerae_bench_v1",
        subset: Optional[Union[str, list]] = ["standard_nomenclature", "loan_words", "rare_words", "general_knowledge", "history", "reading_comprehension"],
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs
    ):
        if base_prompt_template is None:
            base_prompt_template = (
                "주어진 질문과 선택지 중에서 정답이 될 수 있는 선택지의 알파벳을 선택하여 답변하십시오. 답변에는 오직 (A), (B), (C), (D), (E) 중 하나만 포함해야 합니다. 마침표(.), 쉼표(,), 공백, 줄바꿈 등 어떤 추가 문자나 텍스트도 절대 포함하지 마십시오. 예시: B (틀림), (B) (올바름), (B). (틀림), (B)) (틀림)\n\n{query}"
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
        file_path = os.path.join(artifact_dir, "haerae_bench_v1.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"haerae_bench_v1.json not found in artifact: {artifact_dir}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid haerae_bench_v1.json format: expected an object keyed by splits")
        return data

    def load(self) -> List[Dict[str, Any]]:
        """
        W&B artifact의 haerae_bench_v1.json을 로드하여 표준 포맷으로 변환합니다.
        반환 형식: [{"input": ..., "reference": ..., "options": ["(A)",...], "_subset_name": ...}, ...]
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
        print(self.limit)
        print(subset_list)
        for subset_name in subset_list:
            items = split_data.get(subset_name, [])
            if not isinstance(items, list):
                continue
            added = 0
            for item in items:
                query = item.get("query", "")
                query = query.replace("### 정답", "").strip()
                final_input = (
                    self.base_prompt_template.format(query=query)
                    if self.base_prompt_template
                    else query
                )
                answer = (item.get("answer", "") or "").strip()
                results.append({
                    "input": final_input,
                    "reference": answer,
                    "options": ["(A)", "(B)", "(C)", "(D)", "(E)"],
                    "_subset_name": subset_name,
                })
                added += 1
                # dev/limit 처리
                if getattr(self, "dev_mode", False) and added >= 2:
                    break
                if subset_name == "reading_comprehension":
                    if getattr(self, "limit", None) and added >= 100:
                        break
                else:
                    if getattr(self, "limit", None) and added >= self.limit:
                        break
                

        return results

    # HF Hub 의존 변환 로직은 제거되었습니다.

    def get_raw_samples(self) -> Any:
        return self._download_and_load()

    def info(self) -> Dict[str, Any]:
        """
        Returns metadata information about the dataset.
        """
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": (
                "Haerae Bench V1 dataset. "
                "subset=list -> load partial subsets, "
                "subset=str -> load single subset."
            ),
            "evaluation_only": None
        }
