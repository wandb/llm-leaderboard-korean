from typing import Any, Dict, List, Optional, Union
import os
import json

from . import register_dataset
from .base import BaseDataset


@register_dataset("kmmlu_pro")
class KMMLUProDataset(BaseDataset):
    """
    KMMLU-Pro Dataset Class for data sourced from 'HAERAE-HUB/KoSimpleEval'.

    This loader is specifically designed for the 'KMMLU-Pro' subset, which has a
    different structure from the original KMMLU dataset.

    - The 'question' column, containing the full problem text with choices, is used as the 'input'.
    - The integer 'gold' column (1-5) is converted to a string like "(1)" for the 'reference'.
    - The 'category' column (e.g., '민법') is used as '_subset_name' for detailed per-subject analysis.
    - Default 'options' are provided as `["(1)", "(2)", "(3)", "(4)", "(5)"]` to support log-probability-based evaluations.

    Usage example:
        ds = KMMLUProDataset(split="test")
        data = ds.load()
        # data -> [
        #   {
        #     "input": "민법의 법원(法源)에 관한 설명으로 옳지 않은 것은?...",
        #     "reference": "(4)",
        #     "options": ["(1)", "(2)", "(3)", "(4)", "(5)"],
        #     "_subset_name": "민법"
        #   }, ...
        # ]
    """

    def __init__(
        self,
        dataset_name: str = "kmmlu_pro",
        subset: Optional[Union[str, List[str]]] = None,  # allow all or specific subject names
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs,
    ):
        # Set a default prompt template suitable for professional MCQA.
        if base_prompt_template is None:
            base_prompt_template = (
                "다음은 전문 분야의 객관식 문제입니다. 당신의 추론 과정을 간결하게 요약한 후, "
                "\"따라서, 정답은: X\"라고 결론지으십시오. 여기서 X는 1, 2, 3, 4, 5 중 하나입니다.\n\n"
                "질문: {question}"
                # "질문: {question}\n"
                # "(1). {choice_1}\n"
                # "(2). {choice_2}\n"
                # "(3). {choice_3}\n"
                # "(4). {choice_4}\n"
                # "(5). {choice_5}"
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
        file_path = os.path.join(artifact_dir, "kmmlu_pro.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"kmmlu_pro.json not found in artifact: {artifact_dir}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid kmmlu_pro.json format: expected an object keyed by splits")
        return data

    def load(self) -> List[Dict[str, Any]]:
        """
        아티팩트의 kmmlu_pro.json을 로드하여 표준 포맷으로 변환합니다.
        서브셋(None/list/str) 모두 지원하며, 각 서브셋 키는 과목명(예: "민법").
        """
        default_subset = [
            '민법', '경제학원론', '부동산학원론', '감정평가관계법규', '회계학', '노동법1', '노동법2', 
            '사회보험법', '경영학개론', '경영학', '경제원론', '상법', '세법개론', '관세법개론', '무역영어', '내국소비세법',
            '영상치의학/구강내과학/구강병리학', '치과보철학', '소아치과학/치과교정학', '치과재료학/구강생물학', '치주과학/구강보건학',
            '구강악안면외과학', '치과보존학', '보건의약관계법규', '의학총론', '의학각론', '가족관계의등록등에관한법률', '공탁법',
            '민사집행법', '부동산등기법', '상업등기법및비송사건절차법', '헌법', '민사법', '형사법', '공법', '보험계약법', '보험업법',
            '손해사정이론', '내과학1', '내과학2', '침구학', '외과학', '신경정신과학', '안이비인후과학', '부인과학', '소아과학',
            '예방의학', '한방생리학', '본초학', '한약학기초', '한약학 응용', '산업재산권법', '민법개론', '자연과학개론', '생명약학',
            '산업약학', '임상실무약학', '재정학', '세법학개론', '회계학개론', '행정소송법']

        raw = self._download_and_load()
        split_key = self._normalize_split(self.split)
        split_data = raw.get(split_key, {})
        if not isinstance(split_data, dict):
            raise ValueError(
                f"Invalid '{split_key}' split format: expected an object keyed by subsets"
            )

        if self.subset is None:
            subset_list = default_subset
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
        # As the choices are 1-5, we provide them as default options.
        options = ["1", "2", "3", "4", "5"]

        for item in items:
            question_text = item.get("question", "").strip()
            choice_1 = item.get("1", "").strip()
            choice_2 = item.get("2", "").strip()
            choice_3 = item.get("3", "").strip()
            choice_4 = item.get("4", "").strip()
            choice_5 = item.get("5", "").strip()

            # Apply the base prompt template if provided.
            final_input = (
                self.base_prompt_template.format(question=question_text, choice_1=choice_1, choice_2=choice_2, choice_3=choice_3, choice_4=choice_4, choice_5=choice_5)
                if self.base_prompt_template
                else question_text
            )

            # Convert the integer 'gold' answer to a string format like "(4)".
            gold_answer_index = item.get("gold")
            # reference = f"({gold_answer_index})" if gold_answer_index else ""
            reference = gold_answer_index if gold_answer_index else ""

            processed_list.append(
                {
                    "input": final_input,
                    "reference": reference,
                    "options": options,
                    "_subset_name": subset_name,
                    "metadata": {"original_gold_index": gold_answer_index},
                }
            )
            if self.dev and len(processed_list) >= self.limit:
                break
            if len(processed_list) >= self.num_samples:
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
            "description": "KMMLU-Pro dataset loaded from W&B artifact, formatted for HRET.",
            "evaluation_only": None,
        }