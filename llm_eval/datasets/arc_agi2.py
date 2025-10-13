# llm_eval/datasets/arc_agi2.py

from .arc_agi import ARCAGIDataset
from . import register_dataset

@register_dataset("arc_agi2")
class ARCAGI2Dataset(ARCAGIDataset):
    def __init__(self, *args, **kwargs):
        kwargs.update({
            "mode": "github",
            "gh_owner": "arcprize",
            "gh_repo": "ARC-AGI-2",
            "gh_branch": "main",
            "gh_root_dir": "data",
        })
        super().__init__(*args, **kwargs)