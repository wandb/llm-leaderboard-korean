from typing import List, Dict, Any, Optional, Union
import os
import pandas as pd

from .base import BaseDataset
from . import register_dataset

@register_dataset("generic_file")
class GenericFileDataset(BaseDataset):
    """
    A generic dataset class for loading local files (CSV, Parquet, XLSX) and mapping specific columns
    to 'input' and 'reference' fields. Other columns are optionally stored in a 'metadata' dictionary.

    Logic:
      - 'input': The content from the specified input column, optionally formatted using a prompt template.
      - 'reference': The content from the specified reference column.
      - 'metadata': All other columns are stored here.

    Example usage:
        ds = GenericFileDataset(
            dataset_name="generic_file",  # the registered dataset name
            file_path="data.csv",
            input_col="question",
            reference_col="answer",
            split="test"
        )
        data = ds.load()
        # -> [
        #      {
        #        "input": "Formatted prompt text if template is provided or original question text",
        #        "reference": "answer",
        #        "metadata": {...}  # any other columns
        #      },
        #      ...
        #    ]
    """

    def __init__(
        self,
        dataset_name: str,           # "generic_file"
        file_path: str,
        input_col: str = "input",
        reference_col: str = "reference",
        subset: Optional[str] = None,
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            dataset_name (str):
                The registry key for this dataset (usually "generic_file").
            file_path (str):
                Path to the local file (CSV, Parquet, or XLSX).
            input_col (str):
                The column name to be mapped to 'input'.
            reference_col (str):
                The column name to be mapped to 'reference'.
            subset (str | None):
                For compatibility with BaseDataset; unused in this context unless needed.
            split (str):
                Dataset split (e.g., "train", "validation", "test").
            base_prompt_template (str, optional):
                A prompt template for formatting the input. If provided, the template should include a
                placeholder "{input}" which will be replaced by the original input from the file.
                For example: "Please answer the following question:\n{input}"
            **kwargs:
                Additional arguments used for pandas reading functions (e.g., delimiter, sheet_name)
                or for BaseDataset constructor.
        """
        super().__init__(dataset_name=dataset_name, split=split, subset=subset, base_prompt_template=base_prompt_template, **kwargs)
        self.file_path = file_path
        self.input_col = input_col
        self.reference_col = reference_col
        self.loader_kwargs = kwargs  # Arguments for pandas read_* functions

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the local file based on its extension (CSV/TSV, XLS/XLSX, or Parquet) using pandas,
        and then converts it to a standardized list of dictionaries:
          [{"input": ..., "reference": ..., "metadata": ...}, ...].

        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary contains:
                - "input": The formatted input text.
                - "reference": The expected output.
                - "metadata": A dictionary containing any additional columns.
        """
        _, ext = os.path.splitext(self.file_path)
        ext = ext.lower()

        if ext in [".csv", ".tsv"]:
            df = pd.read_csv(self.file_path, **self.loader_kwargs)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(self.file_path, **self.loader_kwargs)
        elif ext == ".parquet":
            df = pd.read_parquet(self.file_path, **self.loader_kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        result = []
        for _, row in df.iterrows():
            input_val = row.get(self.input_col, "")
            ref_val = row.get(self.reference_col, "")

            # Format the input using the base_prompt_template if provided.
            # The template should include a placeholder "{input}".
            raw_input = str(input_val)
            if self.base_prompt_template:
                formatted_input = self.base_prompt_template.format(input=raw_input)
            else:
                formatted_input = raw_input

            # Optionally store all other columns as metadata.
            meta = {}
            for col in df.columns:
                if col not in [self.input_col, self.reference_col]:
                    meta[col] = row[col]

            item = {
                "input": formatted_input,
                "reference": str(ref_val),
                "metadata": meta
            }
            result.append(item)

        return result

    def get_raw_samples(self) -> Any:
        """
        Returns the raw pandas DataFrame before converting it to a standardized format.

        Returns:
            Any: The raw DataFrame loaded from the file.
        """
        _, ext = os.path.splitext(self.file_path)
        ext = ext.lower()

        if ext in [".csv", ".tsv"]:
            return pd.read_csv(self.file_path, **self.loader_kwargs)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(self.file_path, **self.loader_kwargs)
        elif ext == ".parquet":
            return pd.read_parquet(self.file_path, **self.loader_kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def info(self) -> Dict[str, Any]:
        """
        Provides metadata about this dataset, such as the file path and the column mappings.

        Returns:
            Dict[str, Any]: A dictionary containing dataset metadata.
        """
        return {
            "dataset_name": self.dataset_name,
            "file_path": self.file_path,
            "input_col": self.input_col,
            "reference_col": self.reference_col,
            "split": self.split,
            "subset": self.subset,
            "description": "Generic dataset loader for local CSV/XLSX/Parquet files."
        }
