from typing import List, Dict, Any, Optional, Union
import ast
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset


@register_dataset("kobbq")
class KobbqDataset(BaseDataset):
    """
    KoBBQ (naver-ai/kobbq) 데이터셋 로더.

    - subset 파라미터는 사용하지 않으며, 항상 전체 데이터를 로드합니다.
    - 각 샘플은 'options'를 포함하며, 선택지 길이에 맞춰 "(A)".. 형식으로 동적 생성합니다.
    - 정답(`answer`)은 선택지 리스트(`choices`)에서 동일 문자열을 찾아 인덱스→문자 선택지로 매핑합니다.
    - 입력 프롬프트는 `context`(있으면)와 `question`, 그리고 보기(A/B/...)를 나열하여 구성합니다.
    - `_subset_name`은 고정 문자열 "kobbq"로 설정합니다.

    사용 예:
        ds = KobbqDataset(split="test")
        data = ds.load()
        # data -> [
        #   {
        #     "input": "...",
        #     "reference": "(A)",
        #     "options": ["(A)", "(B)", "(C)"],
        #     "_subset_name": "kobbq"
        #   }, ...
        # ]
    """

    def __init__(
        self,
        dataset_name: str = "naver-ai/kobbq",
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs
    ):
        self.dev_mode = kwargs.pop("dev", False)
        if base_prompt_template is None:
            base_prompt_template = (
                "주어진 질문과 선택지 중에서 정답이 될 수 있는 선택지의 알파벳을 선택하여 답변하십시오. 답변에는 오직 (A), (B), (C), (D), (E) 중 하나만 포함해야 합니다. 마침표(.), 쉼표(,), 공백, 줄바꿈 등 어떤 추가 문자나 텍스트도 절대 포함하지 마십시오. 예시: B (틀림), (B) (올바름), (B). (틀림), (B)) (틀림)\n\n{query}"
            )
        super().__init__(dataset_name, split=split, base_prompt_template=base_prompt_template, **kwargs)

    def load(self) -> List[Dict[str, Any]]:
        """
        KoBBQ는 HuggingFace 상에서 단일 서브셋(default)과 test split을 제공합니다.
        항상 전체 데이터를 로드합니다.
        """
        raw_data = load_dataset(self.dataset_name, split=self.split, **self.kwargs)
        return self._convert_to_list(raw_data)

    def _convert_to_list(self, hf_dataset) -> List[Dict[str, Any]]:
        """
        HF Dataset을 순회하며 HRET 표준 포맷의 리스트로 변환:
        {"input": ..., "reference": ..., "options": ..., "_subset_name": "kobbq"}
        """
        processed_list = []
        # 카테고리별 최대 샘플 수 제한
        per_category_limit = 100
        category_counts: Dict[str, int] = {}

        for item in hf_dataset:
            category = (item.get("bbq_category") or item.get("category") or "unknown")

            # 카테고리별 샘플 수 제한 체크
            current_count = category_counts.get(category, 0)
            if current_count >= per_category_limit:
                continue

            context = (item.get("context") or "").strip()
            question = (item.get("question") or "").strip()

            # choices는 일반적으로 list이나, 문자열로 저장된 경우에도 대비
            raw_choices = item.get("choices", [])
            if isinstance(raw_choices, str):
                try:
                    raw_choices = ast.literal_eval(raw_choices)
                except Exception:
                    raw_choices = [raw_choices]
            choices_list = [str(c).strip() for c in (raw_choices or [])]
            if not choices_list:
                choices_list = ["선택지 A", "선택지 B", "선택지 C"]

            # 보기 레터 및 options 생성 (choices 길이에 맞춤)
            options = [f"({chr(ord('A') + i)})" for i in range(len(choices_list))]
            letter_labels = [chr(ord('A') + i) for i in range(len(choices_list))]

            # 입력 프롬프트 구성
            choices_block = "\n".join(
                [f"{letter_labels[i]}. {choices_list[i]}" for i in range(len(choices_list))]
            )
            query_text = (
                f"배경: {context}\n질문: {question}\n{choices_block}" if context else f"질문: {question}\n{choices_block}"
            )
            final_input = (
                self.base_prompt_template.format(query=query_text)
                if self.base_prompt_template
                else query_text
            )

            # 정답 매핑: answer 텍스트를 choices에서 찾아 인덱스→레터 변환
            answer_text = (item.get("answer") or "").strip()
            try:
                answer_index = next(
                    (idx for idx, c in enumerate(choices_list) if c == answer_text),
                    -1,
                )
            except Exception:
                answer_index = -1
            reference = options[answer_index] if 0 <= answer_index < len(options) else ""

            processed_list.append({
                "input": final_input,
                "reference": reference,
                "options": options,
                "_subset_name": "kobbq",
                "metadata": {
                    "bbq_category": category,
                    "sample_id": item.get("sample_id"),
                    "biased_answer": item.get("biased_answer"),
                    "label_annotation": item.get("label_annotation"),
                    "choices": choices_list,
                },
            })
            # 카테고리별 카운트 갱신
            category_counts[category] = current_count + 1
            if self.dev_mode and len(processed_list) >= 100:
                break
        return processed_list

    def get_raw_samples(self) -> Any:
        """
        KoBBQ는 단일 서브셋(default)만 존재하므로 필터 없이 원본 반환.
        """
        return load_dataset(self.dataset_name, split=self.split, **self.kwargs)

    def info(self) -> Dict[str, Any]:
        """
        데이터셋 메타정보 반환.
        """
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "description": (
                "KoBBQ (naver-ai/kobbq) 다지선다 편향 벤치마크."
            ),
            "evaluation_only": None,
        }
