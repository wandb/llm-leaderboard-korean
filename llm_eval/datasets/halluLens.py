import os
from typing import List, Optional
import wandb
import logging
from llm_eval.datasets.base import BaseDataset
from llm_eval.datasets import register_dataset
from llm_eval.utils.logging import get_logger
from dotenv import load_dotenv

load_dotenv()

WANDB_PROJECT_NAME = os.getenv("WANDB_HALLULENS_PROJECT")
logger = get_logger(name="HalluLens", level=logging.INFO)


@register_dataset("halluLens")
class KoHalluLensDataset(BaseDataset):
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
        if not WANDB_PROJECT_NAME:
            raise ValueError("WANDB_HALLULENS_PROJECT environment variable is not set.")

        logger.info("Initializing wandb run...")
        run = wandb.init(project=WANDB_PROJECT_NAME, job_type="dataset-loading")

        try:
            logger.info("Downloading precise_wikiqa artifact...")
            precise_wikiqa_artifact_dir = run.use_artifact(
                f'{WANDB_PROJECT_NAME}/precise_wikiqa:v0',
                type='dataset'
            ).download()

            logger.info("Downloading longwiki artifact...")
            long_wiki_artifact_dir = run.use_artifact(
                f'{WANDB_PROJECT_NAME}/longwiki:v0',
                type='dataset'
            ).download()

            logger.info("Downloading non_entity_refusal artifact...")
            non_entity_refusal_artifact_dir = run.use_artifact(
                f'{WANDB_PROJECT_NAME}/non_entity_refusal:v0',
                type='dataset'
            ).download()

            hallulens_path_set = {
                'precise_wikiqa': os.path.join(precise_wikiqa_artifact_dir, "precise_wikiqa.jsonl"),
                'longwiki': os.path.join(long_wiki_artifact_dir, "longwiki.jsonl"),
                'mixed_entities': os.path.join(non_entity_refusal_artifact_dir, "mixed_entity_2000.csv"),
                'generated_entities': os.path.join(non_entity_refusal_artifact_dir, "generated_entity_1950.csv")
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

        finally:
            wandb.finish()

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
