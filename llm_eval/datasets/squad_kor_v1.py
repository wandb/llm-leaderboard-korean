from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset

@register_dataset("squad_kor_v1")
class SQuADKorV1(BaseDataset):
    """
    SQuAD Kor V1 dataset class (KorQuAD/squad_kor_v1 on Hugging Face).

    - subset: None -> default config
              list[str] -> load and merge multiple configs
              str -> load only the specified config
    - Each sample contains:
        {
          "input": formatted prompt built from context and question,
          "reference": gold short answer text,
          "_subset_name": subset identifier
        }

    Example usage:
        ds = SQuADKorV1(
            dataset_name="KorQuAD/squad_kor_v1",
            split="validation"
        )
        data = ds.load()
        # data -> [{"input": "...", "reference": "...", "_subset_name": ...}, ...]
    """
    def __init__(
        self, 
        dataset_name: str = "KorQuAD/squad_kor_v1",
        subset: Optional[Union[str, list]] = None,
        split: str = "validation",
        base_prompt_template: Optional[str] = None,
        **kwargs
    ):
        self.dev_mode = kwargs.pop("dev", False)
        if base_prompt_template is None:
            base_prompt_template = (
                "다음 지문과 질문을 읽고, 질문에 대한 정답을 한 단어 또는 짧은 구로 추출해 주세요."
                "\n지문: {context}\n\n질문: {question}\n정답:"
            )
        super().__init__(dataset_name, split=split, subset=subset, base_prompt_template=base_prompt_template, **kwargs)

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the data and returns it in the format:
        [{"input": ..., "reference": ..., "options": ..., "_subset_name": ...}, ...]
        """
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

        for item in hf_dataset:
            context = item.get("context", "")
            question = item.get("question", "")

            if self.base_prompt_template:
                prompt = self.base_prompt_template.format(context=context, question=question)
            else:
                prompt = f"{context}\n\n{question}"

            answers = item.get("answers", {}) or {}
            answer_texts = answers.get("text", []) or []
            # Choose the first non-empty gold answer; fall back to empty string
            reference = ""
            for a in answer_texts:
                if isinstance(a, str) and a.strip():
                    reference = a.strip()
                    break

            processed_list.append({
                "input": prompt,
                "reference": reference,
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
                "KorQuAD SQuAD Kor v1.0 extractive QA (KorQuAD/squad_kor_v1). "
                "subset=list -> load multiple configs, subset=str -> single config."
            ),
            "evaluation_only": None
        }
