from typing import List, Dict, Any, Optional
import os
import json
import wandb

from .base import BaseDataset
from . import register_dataset


@register_dataset("korean_hate_speech")
class KoreanHateSpeechDataset(BaseDataset):
    """
    Korean Hate Speech(내부) 데이터셋 로더.

    - 데이터는 W&B Artifact에서 내려받은 `korean_hate_speech.json`을 사용합니다.
    - 파일 형식은 {"training": [...], "dev": [...]} 구조이며, 각 아이템은
      {"input", "output"} 키를 가집니다.
    - 평가 파이프라인 표준 포맷으로 변환 시:
      - input: 그대로 사용(필요 시 base_prompt_template 적용)
      - reference: output를 "1"/"0" 문자열로 변환
      - metadata: {"label"}
    """

    def __init__(
        self,
        dataset_name: str = "korean_hate_speech",
        subset: str = "default",
        split: str = "train",
        base_prompt_template: Optional[str] = None,
        **kwargs,
    ):
        if base_prompt_template is None:
            base_prompt_template = (
                "제시되는 한국어 문장이 혐오 발언인지 판단하여, '1'(혐오 발언) 또는 '0'(아님) 중 하나의 숫자로만 답변하십시오. 답변에는 오직 숫자 하나만 포함해야 하며, 마침표(.), 쉼표(,), 공백, 줄바꿈 등 어떤 추가 문자나 텍스트도 절대 포함하지 마십시오. 정확히 '1' 또는 '0' 중 하나만 출력하십시오. 예시: 0 (올바름), 0. (틀림), 0, (틀림), A0 (틀림)\n\n{query}"
            )
        super().__init__(dataset_name, split=split, subset=subset, base_prompt_template=base_prompt_template, **kwargs)

    def _normalize_split(self, split: str) -> str:
        s = (split or "").lower()
        if s in ("train", "training"):
            return "train"
        if s in ("dev", "validation", "valid", "val"):
            return "dev"
        if s in ("test",):
            return "test"
        # 기본값: dev 우선, 없으면 train
        return "train"

    def _download_and_load(self) -> Dict[str, Any]:

        from llm_eval.wandb_singleton import WandbConfigSingleton
        artifact_dir = WandbConfigSingleton.download_artifact(self.dataset_name)

        file_path = os.path.join(artifact_dir, "korean_hate_speech.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"korean_hate_speech.json not found in artifact: {artifact_dir}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid korean_hate_speech.json format: expected an object keyed by splits")
        return data

    def load(self) -> List[Dict[str, Any]]:
        raw = self._download_and_load()
        split_key = self._normalize_split(self.split)

        split_obj = raw.get(split_key, {})
        # 지원: {split: {subset: [...]}} 또는 {split: [...]}
        if isinstance(split_obj, dict):
            if isinstance(self.subset, (list, tuple)):
                merged: List[Dict[str, Any]] = []
                for subset_name in self.subset:
                    items = split_obj.get(subset_name, [])
                    if isinstance(items, list):
                        merged.extend(items)
                samples = merged
            else:
                samples = split_obj.get(self.subset, [])
        else:
            samples = split_obj if isinstance(split_obj, list) else []

        if not isinstance(samples, list):
            raise ValueError(f"Invalid '{split_key}' split format: expected a list or subset lists")

        results: List[Dict[str, Any]] = []
        for item in samples:
            text = (item.get("comments") or "")
            formatted_input = (
                self.base_prompt_template.format(query=text)
                if self.base_prompt_template
                else text
            )
            reference = "1" if item.get("label", "") == "hate" else "0"
            results.append({
                "input": formatted_input,
                "reference": reference,
                "metadata": {
                    "label": item.get("label", ""),
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
            "description": "Korean Hate Speech dataset loaded from Weights & Biases artifact",
        }


