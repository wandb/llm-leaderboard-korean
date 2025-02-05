from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset

@register_dataset("haerae_bench")
class HaeraeDataset(BaseDataset):
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
            dataset_name="haerae_bench",
            subset=["csat_geo", "csat_law"],  # multiple subsets
            split="test"
        )
        data = ds.load()
        # data -> [
        #   {
        #     "input": "...",
        #     "reference": "...",
        #     "options": ["(A)", "(B)", "(C)", "(D)", "(E)"], # required for log-prob calculation
        #     "_subset_name": "csat_geo"
        #   }, ...
        # ]
    """
    def __init__(
        self, 
        dataset_name: str = "HAERAE-HUB/HAE_RAE_BENCH_1.1",
        subset: Optional[Union[str, list]] = None,
        split: str = "test",
        **kwargs
    ):
        super().__init__(dataset_name, split=split, subset=subset, **kwargs)
        
    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the data and returns it in the format:
        [{"input": ..., "reference": ..., "options": ..., "_subset_name": ...}, ...]
        """
        # 1) Handle different types of subset parameter
        if self.subset is None:
            self.subset = [
                'correct_definition_matching', 
                'csat_geo', 
                'csat_law', 
                'csat_socio', 
                'date_understanding', 
                'general_knowledge', 
                'history', 
                'loan_words', 
                'lyrics_denoising', 
                'proverbs_denoising', 
                'rare_words', 
                'standard_nomenclature', 
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
            # Extract 'query' and 'answer' fields from the original data (default to empty string if missing)
            query = item.get("query", "")
            answer = item.get("answer", "")
            processed_list.append({
                "input": query.strip(),
                "reference": answer.strip(),
                "options": options,
                "_subset_name": subset_name,
            })
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
                "Haerae Bench dataset. "
                "subset=list -> load partial subsets, "
                "subset=str -> load single subset."
            ),
            "evaluation_only": None
        }
