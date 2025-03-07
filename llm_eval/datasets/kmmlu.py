from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset

@register_dataset("kmmlu")
class KMMLUDataset(BaseDataset):
    """
    KMMLU Dataset Class.

    - subset: None -> Load the entire dataset
              list[str] -> Load and merge multiple specified subsets
              str -> Load a single specified subset
    - Each sample includes an 'options' field (if the original data does not have it, use ["(A)", "(B)", "(C)", "(D)"])
    - A '_subset_name' field is added to indicate which subtask the data came from
      (used to calculate separate scores per subset during evaluation)

    Usage example:
        ds = KMMLUDataset(
            dataset_name="kmmlu",
            subset=["Accounting", "Biology"],  # multiple subsets
            split="test"
        )
        data = ds.load()
        # data -> [
        #   {
        #     "input": "...",
        #     "reference": "...",
        #     "options": ["(A)","(B)","(C)","(D)"],  # necessary for log-prob calculation
        #     "_subset_name": "Accounting"
        #   }, ...
        # ]
    """
    def __init__(
        self, 
        dataset_name: str = "HAERAE-HUB/KMMLU",
        subset: Optional[Union[str, list]] = None,
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs
    ):
        # Set default base prompt template if not provided
        if base_prompt_template is None:
            base_prompt_template = (
                "다음은 객관식 문제입니다. 당신의 추론 과정을 간결하게 요약한 후, "
                "\"따라서, 정답은: X\"라고 결론지으십시오. 여기서 X는 (A), (B), (C), (D) 중 하나입니다.\n\n"
                "질문: {question}\n"
                "A. {choice_A}\n"
                "B. {choice_B}\n"
                "C. {choice_C}\n"
                "D. {choice_D}"
            )
        super().__init__(dataset_name, split=split, subset=subset, base_prompt_template=base_prompt_template, **kwargs)
        
    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the data and returns it in the form of 
        [{"input":..., "reference":..., "options":..., "_subset_name":...}, ...].
        """
        # 1) Split based on the type of subset
        if self.subset is None:
            self.subset = [
                'Accounting', 'Agricultural-Sciences', 'Aviation-Engineering-and-Maintenance', 'Biology',
                'Chemical-Engineering', 'Chemistry', 'Civil-Engineering', 'Computer-Science', 'Construction',
                'Criminal-Law', 'Ecology', 'Economics', 'Education', 'Electrical-Engineering', 'Electronics-Engineering',
                'Energy-Management', 'Environmental-Science', 'Fashion', 'Food-Processing',
                'Gas-Technology-and-Engineering', 'Geomatics', 'Health', 'Industrial-Engineer',
                'Information-Technology', 'Interior-Architecture-and-Design', 'Law', 'Machine-Design-and-Manufacturing',
                'Management', 'Maritime-Engineering', 'Marketing', 'Materials-Engineering', 'Mechanical-Engineering',
                'Nondestructive-Testing', 'Patent', 'Political-Science-and-Sociology', 'Psychology', 'Public-Safety',
                'Railway-and-Automotive-Engineering', 'Real-Estate', 'Refrigerating-Machinery', 'Social-Welfare',
                'Taxation', 'Telecommunications-and-Wireless-Technology', 'Korean-History', 'Math'
            ]

        if isinstance(self.subset, list):
            # When multiple subsets are provided
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

        else:  # when subset is a single string
            raw_data = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                **self.kwargs
            )
            return self._convert_to_list(raw_data, subset_name=self.subset)

    def _convert_to_list(self, hf_dataset, subset_name: str) -> List[Dict[str, Any]]:
        """
        Iterates over the HuggingFace Dataset object (hf_dataset) and converts each sample into the format:
        {"input":..., "reference":..., "options":..., "_subset_name": subset_name}.

        English comment:
        - This method constructs a formatted prompt using a base prompt template.
        - If `self.base_prompt_template` is provided, it is used to format the prompt by injecting the question and choices.
        - If not provided, a default template is used.
        """
        processed_list = []
        # Fixed options A~D
        options = ["(A)", "(B)", "(C)", "(D)"]

        # Default base prompt template
        default_template = (
            "다음은 객관식 문제입니다. 당신의 추론 과정을 간결하게 요약한 후, "
            "\"따라서, 정답은: X\"라고 결론지으십시오. 여기서 X는 (A), (B), (C), (D) 중 하나입니다.\n\n"
            "질문: {question}\n"
            "A. {choice_A}\n"
            "B. {choice_B}\n"
            "C. {choice_C}\n"
            "D. {choice_D}"
        )
        
        for item in hf_dataset:
            # Extract and clean the original question and choices
            raw_question = item.get('question', '').replace("### 정답", "").strip()
            choice_A = item.get('A', '').strip()
            choice_B = item.get('B', '').strip()
            choice_C = item.get('C', '').strip()
            choice_D = item.get('D', '').strip()
            
            # Use base_prompt_template if provided, else use default_template
            template = self.base_prompt_template if self.base_prompt_template else default_template
            formatted_query = template.format(
                question=raw_question,
                choice_A=choice_A,
                choice_B=choice_B,
                choice_C=choice_C,
                choice_D=choice_D
            )
            
            # Determine the answer based on the numeric value in "answer" field
            answer_index = int(item.get("answer", 1)) - 1 if item.get("answer") else 0
            answer = options[answer_index] if 0 <= answer_index < len(options) else ""
            
            processed_list.append({
                "input": formatted_query,
                "reference": answer.strip(),
                "options": options,
                "_subset_name": subset_name,
            })
        return processed_list

    def get_raw_samples(self) -> Any:
        """
        Returns the raw data.
        If multiple subsets are specified, returns a list; 
        if a single subset or None is specified, returns a single Dataset (for simplicity).
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
        Returns metadata about the dataset.
        """
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": (
                "KMMLU Benchmark. https://arxiv.org/abs/2402.11548 "
                "subset=list -> load partial subsets, "
                "subset=str -> load single subset."
            ),
            "evaluation_only": None
        }
