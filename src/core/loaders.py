"""
Data loading functions
"""

from pathlib import Path
from typing import Any

import weave

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"


def _to_native(obj: Any) -> Any:
    """Recursively convert Weave types to Python native types"""
    if hasattr(obj, 'keys'):
        # WeaveDict or dict-like
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, str):
        # Strings stay as-is
        return obj
    elif hasattr(obj, '__iter__'):
        # WeaveList or list-like
        return [_to_native(item) for item in obj]
    else:
        return obj


def load_weave_data(
    ref: str,
    split: str | None = None,
) -> list[dict]:
    """Load data from Weave (converted to Python native types)"""
    parts = ref.replace("weave:///", "").split("/")
    project = f"{parts[0]}/{parts[1]}"
    
    weave.init(project)
    data = weave.ref(ref).get()
    rows = data.rows if hasattr(data, "rows") else list(data)
    
    if split:
        rows = [r for r in rows if r.get("split") == split]
    
    # Convert Weave types to Python native
    return [_to_native(r) for r in rows]


def load_jsonl_data(path: str) -> list[dict]:
    """Load data from local JSONL file"""
    import json
    
    file_path = DATA_DIR / path if not Path(path).is_absolute() else Path(path)
    
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
