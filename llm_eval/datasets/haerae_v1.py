from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
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
        dataset_name: str = "HAERAE-HUB/HAE_RAE_BENCH_1.1",
        subset: Optional[Union[str, list]] = None,
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs
    ):
        self.dev_mode = kwargs.pop("dev", False)
        if base_prompt_template is None:
            base_prompt_template = (
                "주어진 질문과 선택지 중에서 정답이 될 수 있는 선택지의 알파벳을 선택하여 답변하십시오. 답변에는 오직 (A), (B), (C), (D), (E) 중 하나만 포함해야 합니다. 마침표(.), 쉼표(,), 공백, 줄바꿈 등 어떤 추가 문자나 텍스트도 절대 포함하지 마십시오. 예시: B (틀림), (B) (올바름), (B). (틀림), (B)) (틀림)\n\n{query}"
            )
        super().__init__(dataset_name, split=split, subset=subset, base_prompt_template=base_prompt_template, **kwargs)

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the data and returns it in the format:
        [{"input": ..., "reference": ..., "options": ..., "_subset_name": ...}, ...]
        """
        # 1) Handle different types of subset parameter
        if self.subset is None:
            self.subset = [
                'standard_nomenclature',
                'loan_words', 
                'rare_words',  
                'general_knowledge', 
                'history', 
                'reading_comprehension'
            ]

        if isinstance(self.subset, list):
            # In the case of multiple subsets
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

        else:  # If subset is a single string
            raw_data = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                **self.kwargs
            )
            return self._convert_to_list(raw_data, subset_name=self.subset)

    def _convert_to_list(self, hf_dataset, subset_name: str) -> List[Dict[str, Any]]:
        """
        Iterates over the HuggingFace Dataset object (hf_dataset)
        and converts each item to the form:
        {"input": ..., "reference": ..., "options": ..., "_subset_name": subset_name}
        """
        processed_list = []
        # Fixed options A~E
        options = ["(A)", "(B)", "(C)", "(D)", "(E)"]

        for item in hf_dataset:
            query = item.get("query", "")
            query = query.replace("### 정답", "").strip() # haerae containes '### 정답', but have to be removed for parsing/cot/etc
            if self.base_prompt_template:
                query = self.base_prompt_template.format(query=query)
            
            answer = item.get("answer", "")
            processed_list.append({
                "input": query,
                "reference": answer.strip(),
                "options": options,
                "_subset_name": subset_name,
            })
            if self.dev_mode and len(processed_list) >= 10:
                break
        return processed_list

    def get_raw_samples(self) -> Any:
        """
        Returns the raw data.
        If multiple subsets are specified, returns a list; 
        if a single subset or None is specified, returns a single Dataset (simple example).
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
