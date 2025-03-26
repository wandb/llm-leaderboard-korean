from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset

@register_dataset("KUDGE")
class KUDGEDataset(BaseDataset):
    """
    KUDGE Dataset Class.

    - subset: 
        None -> Load the entire dataset
        list[str] -> Load and merge only the specified subsets
        str -> Load only the specified subset
    - A '_subset_name' field is added to indicate which subtask the data was loaded from
      (used to compute separate scores per subset during evaluation)

    * Subset Structure:
      1. Pairwise:
         - input: judge_query
         - prediction: chosen_response (the better response)
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

    ** Automatically applies evaluation only **
    - The dataset loads correctly only when split="test"
    - Using split="train" or split="validation" will raise a ValueError

    Usage example:
        ds = KUDGEDataset(
            dataset_name="KUDGE",
            subset=["Pairwise"],  
            split="test"  # evaluation only is enforced
        )
        data = ds.load()
    """

    def __init__(
        self, 
        dataset_name: str = "HAERAE-HUB/KUDGE",
        subset: str = None,  
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs
    ):
        """
        Initializes the KUDGE dataset.

        Args:
            dataset_name: HuggingFace dataset identifier.
            subset: The subset(s) to load.
            split: Dataset split (e.g., "train", "test", "validation").
            **kwargs: Additional options for dataset loading.
        """
        # Enforce that only the "test" split is allowed.
        if split != "test":
            raise ValueError("This dataset is for evaluation only. Use 'test' split.")
            
        super().__init__(dataset_name, split=split, subset=subset, **kwargs)

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the data and converts it into a standardized format.

        Returns:
            List[Dict[str, Any]]: A list of processed samples in the format:
                - input: Evaluation prompt and criteria.
                - prediction: The response to be evaluated.
                - reference: The correct answer or score.
                - _subset_name: The source subset.
                - Additional fields specific to each subset.
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
        Converts a HuggingFace Dataset object into a standardized format for evaluation.

        English comments:
        - This method iterates over the raw dataset and extracts the relevant fields.
        - It constructs the "input" field using conversion rules based on the subset.
        - The "prediction" field is set to the chosen response, and "reference" is determined from the winner field.
        - Additional fields are added as specified in the conversion rules.
        
        Returns:
            List[Dict[str, Any]]: A list of processed samples.
        """
        processed_list = []
        
        # Define conversion rules for different subsets
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
        Returns the raw dataset.

        Returns:
            Any: If a single subset is specified, returns the Dataset object;
                 if multiple subsets are specified, returns a list of Dataset objects.
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

        Returns:
            Dict[str, Any]: Metadata including dataset name, subset, split, and description.
        """
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": (
                "KUDGE dataset. "
                "subset=list -> load multiple subsets, "
                "subset=str -> load a single subset."
            ),
            "evaluation_only": ["llm_judge"]  # llm_judge 평가 방법만 허용
        }
