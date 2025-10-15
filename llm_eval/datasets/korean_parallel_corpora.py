from typing import List, Dict, Any, Optional, Union
import os
import json
import wandb

from .base import BaseDataset
from . import register_dataset


@register_dataset("korean_parallel_corpora")
class KoreanParallelCorporaDataset(BaseDataset):
    """
    Korean Hate Speech(내부) 데이터셋 로더.

    - 데이터는 W&B Artifact에서 내려받은 `korean_parallel_corpora.json`을 사용합니다.
    - 파일 형식은 {"training": [...], "dev": [...]} 구조이며, 각 아이템은
      {"input", "output"} 키를 가집니다.
    - 평가 파이프라인 표준 포맷으로 변환 시:
      - input: 그대로 사용(필요 시 base_prompt_template 적용)
      - reference: output를 "1"/"0" 문자열로 변환
      - metadata: {"label"}
    """

    def __init__(
        self,
        dataset_name: str = "korean_parallel_corpora",
        subset: Optional[Union[str, list]] = ["e2k", "k2e"],
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        artifact_ref: str = "horangi/horangi4-dataset/korean-parallel-corpora:latest",
        **kwargs,
    ):
        if base_prompt_template is None:
            base_prompt_template = (
                "제시되는 문장을 한국어 문장인지 영어 문장인지 판단하여, 한국어 문장이면 영어로, 영어 문장이면 한국어로 번역하십시오. 답변에는 오직 한국어 번역문 또는 영어 번역문만을 포함하고, 그 외의 설명, 주석, 추가 텍스트를 절대 포함하지 마십시오. 반드시 한국어 번역문 또는 영어 번역문만 출력해야 합니다.\n\n{query}"
            )
        super().__init__(dataset_name, split=split, subset=subset, base_prompt_template=base_prompt_template, **kwargs)
        self.artifact_ref = artifact_ref
        self._raw_json: Optional[Dict[str, Any]] = None

    def _normalize_split(self, split: str) -> str:
        s = (split or "").lower()
        if s in ("train", "training"):  # 허용 별칭
            return "train"
        if s in ("dev"):
            return "dev"
        # 기본값은 test 로 둔다
        return "test"

    def _download_and_load(self) -> Dict[str, Any]:
        if self._raw_json is not None:
            return self._raw_json

        from llm_eval.wandb_singleton import WandbConfigSingleton
        run = WandbConfigSingleton.get_instance().run
        artifact = run.use_artifact(self.artifact_ref, type="dataset")
        artifact_dir = artifact.download()
        
        file_path = os.path.join(artifact_dir, "korean-parallel-corpora.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"korean-parallel-corpora.json not found in artifact: {artifact_dir}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid korean-parallel-corpora.json format: expected an object with 'train'/'test'/'dev' keys")
        self._raw_json = data
        return data

    def load(self) -> List[Dict[str, Any]]:
        raw = self._download_and_load()
        split_key = self._normalize_split(self.split)
        split_data = raw.get(split_key, {})
        if not isinstance(split_data, dict):
            raise ValueError(
                f"Invalid '{split_key}' split format: expected an object keyed by subsets"
            )

        subset_list = (
            list(self.subset)
            if isinstance(self.subset, (list, tuple))
            else [self.subset]
        )

        results: List[Dict[str, Any]] = []
        for subset_name in subset_list:
            items = split_data.get(subset_name, [])
            if not isinstance(items, list):
                continue
            # Determine per-subset limit
            subset_limit = getattr(self, "limit", None)
            if getattr(self, "dev_mode", False):
                subset_limit = 10 if subset_limit is None else min(subset_limit, 10)

            added_count = 0
            for item in items:
                text = (item.get("input") or "")
                formatted_input = (
                    self.base_prompt_template.format(query=text)
                    if self.base_prompt_template
                    else text
                )
                reference = item.get("output", "")
                results.append({
                    "input": formatted_input,
                    "reference": reference,
                    "_subset_name": subset_name,
                })
                added_count += 1
                if subset_limit is not None and added_count >= subset_limit:
                    break

        return results

    def get_raw_samples(self) -> Any:
        return self._download_and_load()

    def info(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "split": self._normalize_split(self.split),
            "artifact_ref": self.artifact_ref,
            "description": "Korean Parallel Corpora dataset loaded from Weights & Biases artifact",
        }
