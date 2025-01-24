from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset

@register_dataset("k2_eval")
class K2_EvalDataset(BaseDataset):
    """
    K2-Eval 데이터셋 클래스

    - subset: None -> ["knowledge"] 로 설정
              list[str] -> 여러 subset만 로드하여 합침
              str -> 해당 subset만 로드

    * 로직:
      - input: question + "\n선택지 : a, b, c, d"
      - reference: gold_answer 값(0~3)에 따라 a/b/c/d 중 실제 텍스트
      - options: [a, b, c, d]

    예시:
        ds = K2_EvalDataset(
            dataset_name="HAERAE-HUB/K2-Eval",
            subset="knowledge",
            split="test"
        )
        data = ds.load()
        # data -> [
        #   {
        #     "input": "질문\n선택지 : ...",
        #     "reference": "...",  # gold_answer 인덱스에 해당하는 선택지
        #     "options": ["...", "...", "...", "..."],
        #     "_subset_name": "knowledge"
        #   }, ...
        # ]
    """

    def __init__(
        self,
        dataset_name: str = "HAERAE-HUB/K2-Eval",
        subset: Optional[Union[str, list]] = None,
        split: str = "test",
        **kwargs
    ):
        super().__init__(dataset_name, split=split, subset=subset, **kwargs)

    def load(self) -> List[Dict[str, Any]]:
        """
        데이터를 로드하고 
        [{"input":"...", "reference":"...", "options":[a,b,c,d], "_subset_name":...}, ...] 형태를 반환.
        """
        # 1) subset 기본값 처리
        if self.subset is None:
            # K2-Eval은 knowledge subset만 현재 활용 가능 (generate는 아직.. )
            self.subset = ["knowledge"]

        # 여러 subset 경우
        if isinstance(self.subset, list):
            all_items = []
            for sub in self.subset:
                partial_data = load_dataset(
                    self.dataset_name,
                    sub,
                    split=self.split,
                    **self.kwargs
                )
                all_items.extend(self._convert_to_list(partial_data, subset_name=sub))
            return all_items

        else:
            # subset이 문자열 1개
            raw_data = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                **self.kwargs
            )
            return self._convert_to_list(raw_data, subset_name=self.subset)

    def _convert_to_list(self, hf_dataset, subset_name: str) -> List[Dict[str, Any]]:
        """
        HuggingFace Dataset 객체(hf_dataset)를 순회하며,
        {"input":..., "reference":..., "options":[...], "_subset_name": subset_name} 형태로 변환.
        """
        processed_list = []

        for item in hf_dataset:
            question = item.get("question", "")
            gold_idx = item.get("gold_answer", -1)

            # 선택지 텍스트들을 리스트화
            choices = [
                item.get("a", ""),  # 0
                item.get("b", ""),  # 1
                item.get("c", ""),  # 2
                item.get("d", ""),  # 3
            ]

            # gold_idx가 0~3 범위 안이라면 그 인덱스의 문자열을 정답 참조
            if 0 <= gold_idx < len(choices):
                reference_text = choices[gold_idx]
            else:
                reference_text = ""

            # input에 선택지 표시
            # 예) "질문\n선택지 : optionA, optionB, optionC, optionD"
            input_text = question.strip() + "\n선택지 : " + ", ".join(choices)+"\n답:"

            processed_list.append({
                "input": input_text,
                "reference": reference_text,
                "options": choices,
                "_subset_name": subset_name,
            })

        return processed_list

    def get_raw_samples(self) -> Any:
        """
        원본 데이터를 반환.
        여러 subset이면 리스트로, 하나거나 None이면 단일 Dataset 반환 (간단 예시).
        """
        if self.subset is None:
            return load_dataset(self.dataset_name, split=self.split, **self.kwargs)
        elif isinstance(self.subset, list):
            result = []
            for s in self.subset:
                partial = load_dataset(
                    self.dataset_name, s, split=self.split, **self.kwargs
                )
                result.append(partial)
            return result
        else:
            return load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                **self.kwargs
            )

    def info(self) -> Dict[str, Any]:
        """데이터셋 관련 메타 정보."""
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": (
                "K2_Eval benchmark featuring multiple-choice knowledge tasks. "
                "Columns: question, gold_answer(0~3), a, b, c, d. "
                "subset=list -> load partial subsets, subset=str -> load single subset."
            ),
            "evaluation_only": None,
        }
