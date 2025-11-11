"""
SWE-bench Dataset Integration

This module integrates SWE-bench Verified dataset into the HRET framework,
providing a standardized interface for software engineering benchmarking tasks.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import wandb
from datasets import load_from_disk

from llm_eval.datasets.base import BaseDataset
from llm_eval.datasets import register_dataset
from llm_eval.utils.logging import get_logger

logger = get_logger(name="swebench", level=logging.INFO)


@register_dataset("swebench")
class SWEBenchDataset(BaseDataset):
    """
    SWE-bench Verified Dataset for evaluating LLM's ability to solve real-world
    GitHub issues by generating code patches.

    This dataset loads problem instances from wandb artifacts using Hugging Face
    datasets' Arrow format via load_from_disk.

    Args:
        dataset_name (str): Name of the dataset (default: "swebench")
        split (str): Dataset split to load (default: "test")
        subset (str): Not used for SWE-bench, kept for compatibility
        artifacts_path (str): W&B artifact path (e.g., "entity/project/artifact:version")
        dataset_dir (str): Directory path within the artifact (default: ".")
        max_samples (int): Maximum number of samples to load (default: None, loads all)
        **kwargs: Additional parameters
    """

    def __init__(
        self,
        dataset_name: str = "swebench",
        split: str = "test",
        subset: str = None,
        artifacts_path: Optional[str] = None,
        dataset_dir: str = ".",
        **kwargs
    ):
        super().__init__(dataset_name, split, subset, **kwargs)
        self.artifacts_path = artifacts_path
        self.dataset_dir = dataset_dir
        self._raw_data = None

    def load(self) -> List[Dict[str, Any]]:
        """
        Load SWE-bench dataset from wandb artifacts using load_from_disk.

        Returns:
            List[Dict[str, Any]]: List of samples with 'input' and 'reference' keys.
                - 'input': Problem description formatted as a prompt
                - 'reference': Expected patch/solution
                - Additional metadata: instance_id, repo, version, etc.
        """
        if self._raw_data is None:
            self._raw_data = self._load_from_wandb()

        # Determine effective limit (from multiple sources)
        if self.dev:
            effective_limit = self.limit
        else:
            effective_limit = self.num_samples
        print(f"effective_limit: {effective_limit}, limit: {self.limit}, num_samples: {self.num_samples}")
        samples = []
        for idx, instance in enumerate(self._raw_data):
            # Apply limit
            if effective_limit and idx >= effective_limit:
                break

            # Build prompt: prioritize 'text' field if available (official dataset format)
            if "text" in instance and instance.get("text", "").strip():
                input_text = instance["text"]
            else:
                # Fallback: format from problem_statement
                input_text = self._format_problem(instance)

            # The reference is the gold patch
            reference = instance.get("patch", "")

            sample = {
                "input": input_text,
                "reference": reference,
                # Include all metadata from the original instance for evaluation
                **{k: v for k, v in instance.items() if k not in ["patch"]},
            }
            samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples from SWE-bench dataset")
        return samples

    def _load_from_wandb(self) -> List[Dict[str, Any]]:
        """
        Load dataset from wandb artifacts using Hugging Face datasets load_from_disk.

        Returns:
            List[Dict[str, Any]]: Raw dataset instances
        """
        if not self.artifacts_path:
            raise ValueError(
                "artifacts_path must be specified for SWE-bench dataset. "
                "Example: 'entity/project/swebench_verified:v0'"
            )

        logger.info(f"Loading SWE-bench data from artifact: {self.artifacts_path}")

        try:
            # Download artifact from wandb
            api = wandb.Api()
            artifact = api.artifact(self.artifacts_path)
            artifact_dir = Path(artifact.download())

            # Construct dataset directory path
            dataset_path = artifact_dir / self.dataset_dir

            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Dataset directory not found: {dataset_path}"
                )

            # Load Arrow format dataset using Hugging Face datasets
            logger.info(f"Loading dataset from {dataset_path}")
            hf_dataset = load_from_disk(str(dataset_path))

            # Get the specified split (default: 'test')
            if hasattr(hf_dataset, 'keys') and self.split in hf_dataset:
                hf_dataset = hf_dataset[self.split]
                logger.info(f"Using '{self.split}' split")
            elif hasattr(hf_dataset, 'keys'):
                # If requested split doesn't exist, use the first available
                available_splits = list(hf_dataset.keys())
                logger.warning(
                    f"Split '{self.split}' not found. "
                    f"Available splits: {available_splits}. "
                    f"Using '{available_splits[0]}'"
                )
                hf_dataset = hf_dataset[available_splits[0]]

            # Convert to list of dictionaries
            instances = []
            for sample in hf_dataset:
                instances.append(dict(sample))

            logger.info(f"Loaded {len(instances)} instances from dataset")
            return instances

        except Exception as e:
            logger.error(f"Failed to load SWE-bench data: {e}", exc_info=True)
            raise

    def _format_problem(self, instance: Dict[str, Any]) -> str:
        """
        Format the problem instance into a prompt for the LLM.

        Uses the same format as the official SWE-bench evaluation to ensure consistency.

        Args:
            instance (Dict[str, Any]): Raw instance data

        Returns:
            str: Formatted prompt
        """
        problem_statement = instance.get("problem_statement", "")
        hints_text = instance.get("hints_text", "")

        prompt = f"""You are a software engineer tasked with fixing a bug in a Python codebase.

<issue>
{problem_statement}
</issue>"""

        if hints_text and hints_text.strip():
            prompt += f"""

<hints>
{hints_text}
</hints>"""

        prompt += """

Please provide the **unified diff** that fixes the bug. The diff MUST follow the standard unified diff format with proper line numbers.

Example of correct unified diff format:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,7 +10,7 @@
 def function_name():
     context_line1
     context_line2
-    old_line_to_remove
+    new_line_to_add
     context_line3
     context_line4
```

Important requirements:
1. Each hunk MUST start with `@@ -start,count +start,count @@` where:
   - First pair (-start,count) refers to the original file
   - Second pair (+start,count) refers to the modified file
   - start = starting line number
   - count = number of lines in the hunk
2. Include at least 3 lines of context before and after changes
3. Use `-` prefix for removed lines and `+` prefix for added lines
4. Context lines have no prefix (just a space)

Enclose your diff in one of the following:

<patch>
...your unified diff here...
</patch>

OR

```diff
...your unified diff here...
```

Include only the minimal changes necessary to fix the bug."""

        return prompt

    def get_raw_samples(self) -> List[Dict[str, Any]]:
        """
        Get raw dataset instances.

        Returns:
            List[Dict[str, Any]]: Raw instances
        """
        if self._raw_data is None:
            self._raw_data = self._load_from_wandb()
        return self._raw_data

    def info(self) -> Dict[str, Any]:
        """
        Get dataset metadata.

        Returns:
            Dict[str, Any]: Dataset information
        """
        return {
            "name": self.dataset_name,
            "split": self.split,
            "artifacts_path": self.artifacts_path,
            "num_samples": self.num_samples,
        }
