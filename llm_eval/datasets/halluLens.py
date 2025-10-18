import os
from typing import List, Optional
import wandb
import logging
from llm_eval.datasets.base import BaseDataset
from llm_eval.datasets import register_dataset
from llm_eval.utils.logging import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(name="HalluLens", level=logging.INFO)


@register_dataset("halluLens")
class HalluLensDataset(BaseDataset):
    def __init__(self, subset: Optional[List[str]] = None, split: str = "test", **kwargs):
        """
        This dataset class downloads and loads the HalluLens dataset from Weights & Biases (wandb) artifacts.
        load instance method returns a dictionary containing file paths for each subset of the HalluLens dataset.

        parameters:
            subset : List[str]: 'precise_wikiqa', 'longwiki', 'mixed_entities', 'generated_entities'
            split : str: dataset split (default: "test")
        """
        super().__init__(dataset_name='halluLens', split=split, **kwargs)  # Fixed typo
        self.subset = subset

    def load(self):
        """
        Load HalluLens dataset from wandb artifacts.

        Returns:
            dict: Dictionary mapping subset names to file paths
        """
        logger.info("Downloading halluLens artifact...")
        from llm_eval.wandb_singleton import WandbConfigSingleton
        halluLens_artifact_dir = WandbConfigSingleton.download_artifact(self.dataset_name)

        hallulens_path_set = {
            'precise_wikiqa': os.path.join(halluLens_artifact_dir, "precise_wikiqa/precise_wikiqa.jsonl"),
            'longwiki': os.path.join(halluLens_artifact_dir, "longwiki/longwiki.jsonl"),
            'mixed_entities': os.path.join(halluLens_artifact_dir, "non_entity_refusal/mixed_entity_2000.csv"),
            'generated_entities': os.path.join(halluLens_artifact_dir, "non_entity_refusal/generated_entity_1950.csv")
        }

        # Filter by subset if specified
        if self.subset is not None:
            logger.info(f"Filtering datasets to subset: {self.subset}")
            hallulens_path_set = {
                task: hallulens_path_set[task]
                for task in self.subset
                if task in hallulens_path_set
            }

            # Warn about invalid subset names
            invalid_tasks = set(self.subset) - set(hallulens_path_set.keys())
            if invalid_tasks:
                logger.warning(f"Invalid subset names will be ignored: {invalid_tasks}")

        # Assert dataset file path existence check
        for key, path in hallulens_path_set.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expected file not found for '{key}': {path}")
            logger.info(f"✓ Found {key}: {path}")

        logger.info(f"Successfully loaded {len(hallulens_path_set)} dataset(s)")
        return hallulens_path_set

    def info(self):
        return {
            "description": (
                "This dataset is HalluLens translated in Korean. "
                "The dataset consists of four dynamic generated datasets: "
                "PreciseWiki, LongWiki, MixedEntities, and GeneratedEntities. "
                "load instance method returns a dictionary containing file paths for each subset of the HalluLens dataset."
            ),
            "evaluation_only": True,
            "citation": "HalluLens",
            "available_subsets": ['precise_wikiqa', 'longwiki', 'mixed_entities', 'generated_entities']
        }
