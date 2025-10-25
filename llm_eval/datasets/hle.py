from typing import Any, Dict, List, Optional, Union
import os
import json
from .base import BaseDataset
from . import register_dataset
from llm_eval.utils.logging import get_logger

logger = get_logger(name="hle_dataset", level="INFO")

@register_dataset("hle")
class HLEDataset(BaseDataset):
    """
    HLE (HAERAE Language Evaluation) Dataset Loader.
    This loader handles the 'HLE' subset from 'HAERAE-HUB/KoSimpleEval'.

    By default, it formats data for multimodal (vision-language) models.
    If 'exclude_images' is set to True, it filters out all samples that contain an image,
    making it compatible with text-only LLMs by providing only text-based problems.
    """

    def __init__(
        self,
        dataset_name: str = "hle_text",
        subset: Optional[Union[str, List[str]]] = [
            "Other",
            "Humanities/Social Science",
            "Math",
            "Physics",
            "Computer Science/AI",
            "Biology/Medicine",
            "Chemistry",
            "Engineering",
        ],
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        exclude_images: bool = True,
        **kwargs,
    ):
        if base_prompt_template is None:
            base_prompt_template = (
                "다음은 전문가 수준의 문제입니다. 문제 해결 과정을 단계별로 서술하고, "
                "마지막에 \"최종 정답:\" 형식으로 결론을 제시하십시오.\n\n"
                "문제: {question}"
            )
            
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            subset=subset,
            base_prompt_template=base_prompt_template,
            **kwargs,
        )
        # Text-only dataset; image fields are not expected or used.
        self.exclude_images = True

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
        # 텍스트 전용 파일만 사용
        file_path = os.path.join(artifact_dir, "hle_text.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"hle_text.json not found in artifact: {artifact_dir}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid HLE json format: expected an object keyed by splits")
        return data

    def load(self) -> List[Dict[str, Any]]:
        """
        HLE 아티팩트를 로드하여 표준 포맷 리스트로 변환합니다.
        텍스트 전용 데이터만 사용합니다(이미지 없음).
        """
        logger.info(f"Loading dataset '{self.dataset_name}' with subset '{self.subset}' for split '{self.split}'.")
        raw = self._download_and_load()
        split_key = self._normalize_split(self.split)
        split_data = raw.get(split_key, {})
        if not isinstance(split_data, dict):
            raise ValueError(f"Invalid '{split_key}' split format: expected an object keyed by subsets")

        if self.subset is None:
            subset_list = list(split_data.keys())
        elif isinstance(self.subset, (list, tuple)):
            subset_list = list(self.subset)
        else:
            subset_list = [self.subset]

        processed_list: List[Dict[str, Any]] = []
        for subset_name in subset_list:
            items = split_data.get(subset_name, [])
            if not isinstance(items, list):
                continue
            added = 0
            for item in items:
                try:
                    question_text = (item.get("question") or "").strip()
                    gold_answer = (item.get("gold") or "").strip()

                    formatted_question = self.base_prompt_template.format(question=question_text)

                    # Text-only input
                    final_input = formatted_question

                    possible_answers = [ans.strip() for ans in gold_answer.split(',') if str(ans).strip()]
                    reference = possible_answers[0] if possible_answers else ""

                    processed_list.append(
                        {
                            "input": final_input,
                            "reference": reference,
                            "_subset_name": subset_name,
                            "metadata": {
                                "all_references": possible_answers,
                            },
                        }
                    )
                    added += 1  
                    if getattr(self, "dev_mode", False) and added >= 2:
                        break
                    if getattr(self, "limit", None) and added >= self.limit:
                        break
                except Exception as e:
                    logger.error(f"Failed to process item: {e}")
                    continue

        logger.info(f"Loaded {len(processed_list)} text-only samples.")
        return processed_list

    # HF Hub 변환 유틸은 제거되었습니다.

    def info(self) -> Dict[str, Any]:
        """Returns metadata about the dataset."""
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self._normalize_split(self.split),
            "description": "HLE Text: A text-only QA dataset loaded from W&B artifact.",
            "evaluation_only": None,
        }