import os
from typing import List
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
    def __init__(self, subset: List[str] = None, split="test", **kwargs):
        """
        This dataset class downloads and loads the HalluLens dataset from Weights & Biases (wandb) artifacts.
        load instance method returns a dictionary containing file paths for each subset of the HalluLens dataset.

        parameters:
            subset : List[str]: 'precise_wikiqa', 'longwiki', 'mixed_entities', 'generated_entities'
        """
        super().__init__(dataset_name='halluLens', plit=split, **kwargs)
        self.subset = subset

    def load(self):
        if not WANDB_PROJECT_NAME:
            raise ValueError("WANDB_HALLULENS_PROJECT environment variable is not set.")
        run = wandb.init()
        precise_wikiqa_artifact_dir = run.use_artifact(os.path.join(WANDB_PROJECT_NAME, 'precise_wikiqa:v0'),
                                                       type='dataset').download()
        long_wiki_artifact_dir = run.use_artifact(os.path.join(WANDB_PROJECT_NAME, 'longwiki:v0'),
                                                  type='dataset').download()
        non_entity_refusal_artifact_dir = run.use_artifact(os.path.join(WANDB_PROJECT_NAME, 'non_entity_refusal:v0'),
                                                           type='dataset').download()

        hallulens_path_set = {
            'precise_wikiqa': os.path.join(precise_wikiqa_artifact_dir, "precise_wikiqa.jsonl"),
            'longwiki': os.path.join(long_wiki_artifact_dir, "longwiki.jsonl"),
            'mixed_entities': os.path.join(non_entity_refusal_artifact_dir, "mixed_entity_2000.csv"),
            'generated_entities': os.path.join(non_entity_refusal_artifact_dir, "generated_entity_1950.csv")
        }
        if self.subset is not None:
            hallulens_path_set = {task: hallulens_path_set[task] for task in self.subset if task in hallulens_path_set}

        # Asset dataset file path existence check
        for key, path in hallulens_path_set.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expected file not found: {path}")

        return hallulens_path_set

    def info(self):
        return {
            "description": (
                "This dataset is HalluLens translated in Korean."
                "The dataset consists of four dynamic generated datasets, which are PreciseWiki, LongWiki, MixedEntities, and GeneratedEntities"
                "load instance method returns a dictionary containing file paths for each subset of the HalluLens dataset."
            ),
            "evaluation_only": None,
            "citation": "HalluLens",
        }
