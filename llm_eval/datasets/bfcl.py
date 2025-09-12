import json
import logging
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llm_eval.datasets.base import BaseDataset
from llm_eval.datasets import register_dataset
from llm_eval.utils.logging import get_logger


logger = get_logger(name="bfcl", level=logging.INFO)


def _extract_question_text(question_field: Any) -> str:
    """
    Extract question text from BFCL question field.

    BFCL question field (in Horangi JSONL) is typically structured as:
      [[{"role": "user", "content": "..."}]]

    This function extracts the first message's content as the prompt.
    Falls back to str(question_field) if the structure differs from expected format.

    Args:
        question_field: The question field from BFCL data, typically a nested list structure

    Returns:
        str: The extracted question text content
    """
    try:
        if isinstance(question_field, list) and len(question_field) > 0:
            first_turn = question_field[0]
            if isinstance(first_turn, list) and len(first_turn) > 0:
                first_msg = first_turn[0]
                if isinstance(first_msg, dict) and "content" in first_msg:
                    return str(first_msg["content"])
        return str(question_field)
    except Exception:
        return str(question_field)


def download_bfcl_data(data_dir: Path, use_full_dataset: bool = False, ids_file: Optional[Path] = None) -> None:
    """
    Download BFCL data folder from the official repository.

    Downloads the entire data directory from the Berkeley Function Call Leaderboard repository:
    https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/data/

    Uses git sparse-checkout to efficiently download only the required data folder without
    cloning the entire repository.

    Args:
        data_dir: Target directory where BFCL data should be stored
        use_full_dataset: If True, raises an error as full dataset is not supported
        ids_file: Optional path to test_case_ids_to_generate.json file for sampling

    Raises:
        NotImplementedError: If use_full_dataset is True
        FileNotFoundError: If the expected data directory is not found after download
        subprocess.CalledProcessError: If git operations fail
    """
    import shutil

    if use_full_dataset:
        raise NotImplementedError("Full dataset is not supported. Only sampled dataset is available.")

    # Create parent directory if it doesn't exist
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading BFCL data folder to {data_dir}")

    # Repository and path configuration
    repo_url = "https://github.com/ShishirPatil/gorilla.git"
    data_path = "berkeley-function-call-leaderboard/bfcl_eval/data"

    # Create a temporary directory for the sparse checkout
    with tempfile.TemporaryDirectory(prefix=".bfcl_download_", suffix="_cache") as temp_dir:
        temp_dir = Path(temp_dir)

        try:
            # Initialize git repo and set up sparse checkout
            subprocess.run(["git", "init"], cwd=temp_dir, check=True, capture_output=True)
            subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=temp_dir, check=True, capture_output=True)
            subprocess.run(["git", "config", "core.sparseCheckout", "true"], cwd=temp_dir, check=True, capture_output=True)

            # Set up sparse checkout pattern to only download the data folder
            sparse_checkout_file = temp_dir / ".git" / "info" / "sparse-checkout"
            sparse_checkout_file.parent.mkdir(parents=True, exist_ok=True)
            with open(sparse_checkout_file, "w") as f:
                f.write(f"{data_path}/*\n")

            # Pull only the data folder
            subprocess.run(["git", "pull", "origin", "main"], cwd=temp_dir, check=True, capture_output=True)

            # Verify the full data directory exists
            source_data_dir = temp_dir / data_path
            if not source_data_dir.exists():
                raise FileNotFoundError(f"Expected data directory not found at {source_data_dir}")

            logger.info("Successfully downloaded BFCL data folder")

            # Build the sampled version directly to the target data_dir
            logger.info("Building sampled BFCL data...")
            build_bfcl_data_sampled(source_data_dir, data_dir, ids_file)

        except subprocess.CalledProcessError as e:
            logger.error(f"Git sparse checkout failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise
        # Temporary directory is automatically cleaned up by context manager


def build_bfcl_data_sampled(source_data_dir: Path, target_data_dir: Path, ids_file: Optional[Path] = None) -> None:
    """
    Build a sampled BFCL data directory from the full BFCL data.

    This function creates a subset of the full BFCL data based on test case IDs specified
    in a test_case_ids_to_generate.json file. It processes all BFCL_v4_*.json files,
    filters entries based on the specified IDs, and copies necessary supporting files.

    Args:
        source_data_dir: Path to the full BFCL data directory
        target_data_dir: Path where the sampled data should be created
        ids_file: Optional path to test_case_ids_to_generate.json file. If None, looks for it
                 in the target directory.

    Raises:
        FileNotFoundError: If source directory or ids_file doesn't exist
        json.JSONDecodeError: If ids_file contains invalid JSON
    """
    import shutil

    logger.info(f"Building sampled BFCL data from {source_data_dir} to {target_data_dir}")

    # Ensure source exists
    if not source_data_dir.exists():
        raise FileNotFoundError(f"Source data directory not found: {source_data_dir}")

    # Create target directory
    target_data_dir.mkdir(parents=True, exist_ok=True)

    # Determine the IDs file to use
    if ids_file is None:
        # Look for existing ids file in target directory
        ids_file = target_data_dir / "test_case_ids_to_generate.json"
        if not ids_file.exists():
            raise FileNotFoundError(f"test_case_ids_to_generate.json not found at {ids_file}. " "Please provide a valid ids_file parameter or ensure the file exists in the target directory.")

    # Load the IDs mapping from the configuration file
    with open(ids_file, "r") as f:
        ids_mapping = json.load(f)

    # Collect all IDs that should be included across all categories
    all_target_ids = set()
    for category, ids in ids_mapping.items():
        all_target_ids.update(ids)

    logger.info(f"Filtering for {len(all_target_ids)} specific test case IDs")

    # Process each BFCL data file
    processed_files = 0
    total_entries = 0
    filtered_entries = 0

    for source_file in source_data_dir.glob("BFCL_v4_*.json"):
        # Extract category from filename (e.g., "BFCL_v4_simple_python.json" -> "simple_python")
        category = source_file.stem.replace("BFCL_v4_", "")
        target_file = target_data_dir / source_file.name

        # Get target IDs for this specific category
        category_target_ids = set(ids_mapping.get(category, []))

        # Process the source file and filter entries
        filtered_data = []
        with open(source_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    total_entries += 1

                    entry_id = entry.get("id", "")

                    # Include entry if its ID is in the target set for this category
                    if entry_id in category_target_ids:
                        filtered_data.append(entry)
                        filtered_entries += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON line in {source_file}: {e}")
                    continue

        # Write filtered data (creates empty file if no entries match)
        with open(target_file, "w") as f:
            for entry in filtered_data:
                f.write(json.dumps(entry) + "\n")

        if filtered_data:
            logger.debug(f"Created {target_file.name} with {len(filtered_data)} entries")
        else:
            logger.debug(f"Created empty {target_file.name} (no matching entries for category '{category}')")
        processed_files += 1

    # Copy other necessary files from source to target
    other_files_to_copy = ["multi_turn_func_doc", "memory_prereq_conversation"]

    for item_name in other_files_to_copy:
        source_item = source_data_dir / item_name
        target_item = target_data_dir / item_name

        if source_item.exists():
            if source_item.is_dir():
                # Remove existing directory and copy fresh
                if target_item.exists():
                    shutil.rmtree(target_item)
                shutil.copytree(source_item, target_item)
                logger.debug(f"Copied directory {item_name}")
            else:
                shutil.copy2(source_item, target_item)
                logger.debug(f"Copied file {item_name}")

    # Filter possible_answer directory by IDs
    source_possible_answer = source_data_dir / "possible_answer"
    target_possible_answer = target_data_dir / "possible_answer"

    if source_possible_answer.exists():
        target_possible_answer.mkdir(parents=True, exist_ok=True)

        for source_answer_file in source_possible_answer.glob("*.json"):
            # Extract category from filename (e.g., "BFCL_v4_simple_python.json" -> "simple_python")
            category = source_answer_file.stem.replace("BFCL_v4_", "")
            target_answer_file = target_possible_answer / source_answer_file.name

            # Get target IDs for this category
            category_target_ids = set(ids_mapping.get(category, []))

            # Process and filter answer entries
            filtered_answers = []
            with open(source_answer_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                        entry_id = entry.get("id", "")

                        # Handle potential ID format mismatch where some files use double underscore
                        # instead of single underscore (e.g., "simple_javascript__0" vs "simple_javascript_0")
                        normalized_entry_id = entry_id.replace("__", "_")

                        if entry_id in category_target_ids or normalized_entry_id in category_target_ids:
                            # Update the entry ID to use the normalized format for consistency
                            entry["id"] = normalized_entry_id
                            filtered_answers.append(entry)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON line in {source_answer_file}: {e}")
                        continue

            # Write filtered answers (creates empty file if no entries match)
            with open(target_answer_file, "w") as f:
                for entry in filtered_answers:
                    f.write(json.dumps(entry) + "\n")

            if filtered_answers:
                logger.debug(f"Created {target_answer_file.name} with {len(filtered_answers)} entries")
            else:
                logger.debug(f"Created empty {target_answer_file.name} (no matching entries for category '{category}')")

    # Copy format sensitivity file if it exists
    for format_file in source_data_dir.glob("*format_sensitivity.json"):
        target_format_file = target_data_dir / format_file.name
        shutil.copy2(format_file, target_format_file)
        logger.debug(f"Copied {format_file.name}")

    # Log summary of the sampling operation
    logger.info(f"Successfully built sampled BFCL data:")
    logger.info(f"  - Processed {processed_files} data files")
    logger.info(f"  - Filtered {filtered_entries} entries from {total_entries} total")
    logger.info(f"  - Target directory: {target_data_dir}")


@register_dataset("bfcl")
class BFCLDataset(BaseDataset):
    """
    BFCL (Berkeley Function Call Leaderboard) dataset loader for function-calling evaluation data.

    This loader reads local JSONL files under the bfcl folder structure with the pattern:
      llm_eval/datasets/bfcl/BFCL_v4_*.json

    Each line in those files is a JSON object containing fields such as 'id', 'question', and 'function'.
    The dataset is designed for evaluating large language models' ability to make function calls correctly.

    Output format per item:
      {
        "input": str,           # extracted from the first user message in 'question'
        "reference": str,       # empty (evaluation handled by bfcl_eval pipeline)
        "metadata": {
            "id": str,          # unique identifier for the test case
            "category": str,    # derived from filename after the 'BFCL_v4_' prefix
            "functions": [...], # raw function/tool schema array from the JSONL
        }
      }

    Parameters:
      - data_dir: Base directory that contains BFCL_v4_*.json files. If None, defaults to
                  'bfcl' relative to this package root.
      - categories: list[str] or str; when provided, only load files whose derived category matches.
                   If None, defaults to categories that have instances in test_case_ids_to_generate.json.
      - auto_download: if True, automatically download BFCL data files if they don't exist.
      - use_full_dataset: if True, raises an error as full dataset is not supported.
                         Only sampled dataset is available for efficiency.
    """

    def __init__(
        self,
        dataset_name: str = "bfcl",
        subset: Optional[Union[str, List[str]]] = None,
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        data_dir: Optional[str] = None,
        categories: Optional[Union[str, List[str]]] = None,
        auto_download: bool = False,
        use_full_dataset: bool = False,
        **kwargs,
    ):
        super().__init__(dataset_name=dataset_name, split=split, subset=subset, base_prompt_template=base_prompt_template, **kwargs)

        if use_full_dataset:
            raise NotImplementedError("Full dataset is not supported. Only sampled dataset is available.")

        # Resolve default data_dir if not provided.
        # Default to bfcl folder in the datasets directory
        if data_dir is None:
            root_dir = Path(__file__).resolve().parent
            data_dir = root_dir / "bfcl"
        self.data_dir = Path(str(data_dir))

        # Set default categories from test_case_ids_to_generate.json if not provided
        if categories is None:
            categories = self._get_default_categories()

        # Normalize categories to list[str] if provided as string
        if isinstance(categories, str):
            categories = [categories]
        self.categories = categories

        self.auto_download = auto_download
        self.use_full_dataset = use_full_dataset

    def _get_default_categories(self) -> List[str]:
        """
        Get default categories from test_case_ids_to_generate.json.

        Returns categories that have at least one instance in the target IDs mapping.

        Returns:
            List[str]: List of category names that have instances
        """
        try:
            ids_file = self.data_dir / "test_case_ids_to_generate.json"
            if ids_file.exists():
                with open(ids_file, "r") as f:
                    ids_mapping = json.load(f)

                # Return categories that have at least one instance
                return [category for category, ids in ids_mapping.items() if ids]
            else:
                logger.warning(f"test_case_ids_to_generate.json not found at {ids_file}, using all categories")
                return []
        except Exception as e:
            logger.warning(f"Failed to load default categories: {e}, using all categories")
            return []

    def load(self) -> List[Dict[str, Any]]:
        """
        Load the dataset, downloading if necessary, and return the processed samples.

        This method checks if the BFCL data files exist locally. If not and auto_download
        is enabled, it will download the data from the official repository. Then it
        processes the JSONL files and returns a list of formatted samples.

        Returns:
            List[Dict[str, Any]]: A list of processed samples, where each sample is a
                                  dictionary containing 'input', 'reference', and 'metadata'.

        Raises:
            FileNotFoundError: If data files don't exist and auto_download is False
        """
        # Check if data exists, download if necessary
        if not self.data_dir.exists() or not any(self.data_dir.glob("BFCL_v4_*.json")):
            if self.auto_download:
                logger.info("BFCL data files not found, downloading...")
                download_bfcl_data(self.data_dir, use_full_dataset=self.use_full_dataset)
            else:
                raise FileNotFoundError(f"BFCL data_dir not found: {self.data_dir}")

        return self._convert_to_list()

    def _convert_to_list(self) -> List[Dict[str, Any]]:
        """
        Convert BFCL data files to a list of examples with category information.

        Processes all BFCL_v4_*.json files in the data directory, extracting the question
        text and function schemas for each entry. Applies category filtering if specified.

        Returns:
            List[Dict[str, Any]]: List of processed examples with input, reference, and metadata
        """
        examples: List[Dict[str, Any]] = []

        # Get all BFCL_v4_*.json files and sort for consistent ordering
        files = sorted(self.data_dir.glob("BFCL_v4_*.json"))

        for file_path in files:
            # Extract category from filename
            category = self._category_from_filename(file_path.name)

            # Skip if categories filter is specified and this category is not included
            if self.categories is not None and category not in self.categories:
                continue

            # Read JSONL file line by line
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)

                        # Extract question text from the nested structure
                        question_text = _extract_question_text(obj.get("question"))
                        formatted_input = self.base_prompt_template.format(input=question_text) if self.base_prompt_template else question_text

                        # Get functions field (handle both 'function' and 'functions' keys)
                        functions_field = obj.get("function")
                        if functions_field is None:
                            functions_field = obj.get("functions")

                        # Create example with category information
                        example = {
                            "input": formatted_input,
                            "reference": "",  # BFCL evaluation is handled by external pipeline
                            "metadata": {
                                "id": obj.get("id"),
                                "category": category,
                                "functions": functions_field,
                            },
                        }
                        examples.append(example)

                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSON line in {file_path}")
                        continue

        # Log summary of loaded data
        filtered_files = [f for f in files if self.categories is None or self._category_from_filename(f.name) in self.categories]
        logger.info(f"Loaded {len(examples)} BFCL examples from {len(filtered_files)} file(s).")
        return examples

    @staticmethod
    def _category_from_filename(filename: str) -> str:
        """
        Extract category from BFCL filename.

        Converts filenames like "BFCL_v4_simple_python.json" to "simple_python".

        Args:
            filename: The filename to extract category from

        Returns:
            str: The extracted category name
        """
        name = Path(filename).name
        if name.startswith("BFCL_v4_") and name.endswith(".json"):
            return name[len("BFCL_v4_") : -len(".json")]
        return name

    def get_raw_samples(self) -> Any:
        """
        Get raw sample information for debugging or inspection.

        Returns:
            Dict containing file paths that would be processed
        """
        files = sorted(self.data_dir.glob("BFCL_v4_*.json"))
        if self.categories is not None:
            files = [f for f in files if self._category_from_filename(f.name) in self.categories]
        return {"files": [str(p) for p in files]}

    def info(self) -> Dict[str, Any]:
        """
        Get dataset information and configuration.

        Returns:
            Dict containing dataset metadata and configuration parameters
        """
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "subset": self.subset,
            "data_dir": str(self.data_dir),
            "categories": self.categories,
            "auto_download": self.auto_download,
            "use_full_dataset": self.use_full_dataset,
            "description": "BFCL dataset loader that reads local BFCL_v4_*.json JSONL files and extracts prompts with category information for function calling evaluation.",
        }
