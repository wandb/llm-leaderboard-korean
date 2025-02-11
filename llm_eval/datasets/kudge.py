from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset

@register_dataset("KUDGE")
class KUDGEDataset(BaseDataset):
    """
    KUDGE 데이터셋 클래스.

    - subset: None -> 전체 로드
              list[str] -> 해당 여러 subset만 로드하여 합침
              str -> 해당 subset만 로드
    - '_subset_name' 필드를 추가해, 어떤 서브태스크에서 불러온 데이터인지 표시
      (Evaluation 시점에 subset별 점수를 구분하기 위해)

    * 서브셋 구조:
      1. Pairwise:
         - input: judge_query
         - prediction: chosen_response (더 나은 응답)
         - reference: winner (A: chosen_model, B: rejected_model)
         
      2. Pairwise-False:
         - input: judge_query
         - prediction: response_with_false_info
         - reference: winner
         
      3. Pointwise:
         - input: judge_query
         - prediction: response
         - reference: final_score

      4. Pointwise-False:
         - input: judge_query
         - prediction: response
         - reference: original_human_score

    **자동으로 evaluation only 적용**
    - 오직 `split="test"`일 때만 데이터셋이 정상적으로 로드됨
    - `split="train"` 또는 `split="validation"` 사용 시 ValueError 발생

    사용 예시:
        ds = KUDGEDataset(
            dataset_name="KUDGE",
            subset=["Pairwise"],  
            split="test"  # evaluation only 적용됨
        )
        data = ds.load()
    """

    def __init__(
        self, 
        dataset_name: str = "HAERAE-HUB/KUDGE",
        subset: str = None,  
        split: str = "test",
        **kwargs
    ):
        """
        KUDGE 데이터셋 초기화
        Args:
            dataset_name: HuggingFace 데이터셋 이름
            subset: 로드할 서브셋 이름(들)
            split: 데이터 분할 (예: "train", "test", "validation")
            **kwargs: 추가 데이터 로드 옵션
        """
        # 오직 "test" split만 허용함.
        if split != "test":
            raise ValueError("This dataset is for evaluation only. Use 'test' split.")
            
        super().__init__(dataset_name, split=split, subset=subset, **kwargs)

    def load(self) -> List[Dict[str, Any]]:
        """
        데이터를 로드하고 표준 형식으로 변환하여 반환

        Returns:
            List[Dict[str, Any]]: 변환된 데이터 리스트
                - input: 평가 지시문과 판단 기준
                - prediction: 평가 대상 응답
                - reference: 정답 또는 점수
                - _subset_name: 데이터 출처
                - 기타 서브셋별 추가 필드들
        """
        if self.subset is None:
            raise ValueError("subset must be specified for KUDGE dataset")
            
        raw_data = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            **self.kwargs
        )
        return self._convert_to_list(raw_data, subset_name=self.subset)

    def _convert_to_list(self, hf_dataset, subset_name: str) -> List[Dict[str, Any]]:
        """
        HuggingFace Dataset 객체를 평가용 표준 형식으로 변환
        """
        processed_list = []
        
        # 서브셋별 변환 규칙 정의
        conversion_rules = {
            'Pairwise': {
                'input_fields': ['judge_query'],  
                'prediction_field': 'chosen_response',
                'reference_field': 'winner',
                'additional_fields': {
                    'model_response_b': 'rejected_response',
                    'judge_type': 'response_comparison',
                    'model_a': 'chosen_model',
                    'model_b': 'rejected_model'
                }
            },
            'Pairwise-False': {
                'input_fields': ['judge_query'],  
                'prediction_field': 'response_with_false_info',
                'reference_field': 'winner',
                'additional_fields': {
                    'model_response_b': 'original_response',
                    'judge_type': 'response_comparison'
                }
            },
            'Pointwise': {
                'input_fields': ['judge_query'],  
                'prediction_field': 'response',
                'reference_field': 'final_score',
                'additional_fields': {
                    'judge_type': 'rubric_and_response',
                }
            },
            'Pointwise-False': {
                'input_fields': ['judge_query'],  
                'prediction_field': 'response',
                'reference_field': 'original_human_score',
                'additional_fields': {
                    'judge_type': 'rubric_and_response',
                }
            },
        }
        
        rule = conversion_rules.get(subset_name)
        if not rule:
            raise ValueError(f"Unknown subset: {subset_name}")
        
        for item in hf_dataset:
            result = {
                "input": "\n".join(str(item.get(field, '')) for field in rule['input_fields']),
                "prediction": str(item.get(rule['prediction_field'], '')),
                "reference": str(item.get(rule['reference_field'], '')),
                "_subset_name": subset_name
            }
                
            if 'additional_fields' in rule:
                for key, field in rule['additional_fields'].items():
                    if field in item:
                        result[key] = str(item[field])
                    else:
                        result[key] = field  
            
            processed_list.append(result)

        return processed_list

    def get_raw_samples(self) -> Any:
        """
        원본 데이터셋을 반환

        Returns:
            Any: 단일 서브셋이면 Dataset 객체, 
                 여러 서브셋이면 Dataset 객체들의 리스트
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
        """
        데이터셋 메타 정보를 반환

        Returns:
            Dict[str, Any]: 데이터셋 이름, 서브셋, 분할, 설명 등의 메타 정보
        """
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": (
                "KUDGE dataset. "
                "subset=list -> load partial subsets, "
                "subset=str -> load single subset."
            ),
            "evaluation_only": self.split == "test"  # split이 "test"일 때만 evaluation only
        }

