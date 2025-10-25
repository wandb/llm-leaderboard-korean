# llm_eval/datasets/arc_agi2.py

from .arc_agi import ARCAGIDataset
from . import register_dataset

@register_dataset("arc_agi2")
class ARCAGI2Dataset(ARCAGIDataset):
    def __init__(self, *args, **kwargs):
        kwargs.update({
            "dataset_name": "arc_agi2",
            "split": "evaluation",
            "subset": "default",
        })
        super().__init__(*args, **kwargs)