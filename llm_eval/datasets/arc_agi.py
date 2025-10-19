from __future__ import annotations

from typing import Any, Dict, List, Optional, Iterable
import requests
from urllib.parse import quote

from .base import BaseDataset
from . import register_dataset
from llm_eval.utils.logging import get_logger

logger = get_logger(name="arc_agi", level="INFO")

try:
    from datasets import load_dataset, Dataset, DatasetDict
except Exception:
    load_dataset = None  # type: ignore[assignment]

DEFAULT_ARC_PROMPT_TEMPLATE = (
    "You are given ARC tasks. Grids use digits 0-9 only.\n"
    "Return only the final grid as comma-separated rows, with no extra text.\n\n"
    "{examples}\n"
    "{query_block}\n"
)

Grid = List[List[int]]


def _serialize_grid(g: Grid) -> str:
    return "\n".join(",".join(str(c) for c in row) for row in g)


def _build_examples(train_pairs: List[Dict[str, Any]]) -> str:
    if not train_pairs:
        return ""
    blocks = []
    for i, pair in enumerate(train_pairs, start=1):
        tin = _serialize_grid(pair["input"])
        tout = _serialize_grid(pair["output"])
        blocks.append(f"[Example {i}]\n[Input]\n{tin}\n[Output]\n{tout}\n")
    return "\n".join(blocks).strip()


def _build_query_block(test_input_grid: Grid) -> str:
    return "[Query Input]\n" + _serialize_grid(test_input_grid)


def _task_to_prompt(
    task: Dict[str, Any],
    test_input_grid: Grid,
    base_prompt_template: str,
) -> str:
    examples = _build_examples(task.get("train", []))
    query_block = _build_query_block(test_input_grid)
    template = base_prompt_template or DEFAULT_ARC_PROMPT_TEMPLATE
    if not examples:
        template = template.replace("{examples}\n", "")
    return template.format(examples=examples, query_block=query_block)


def _list_github_json_urls(
    owner: str,
    repo: str,
    branch: str,
    subdir: str,
    suffix: str = ".json",
    token: Optional[str] = None,
    timeout: int = 30,
) -> List[str]:
    api = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/contents/{quote(subdir)}?ref={quote(branch)}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    r = requests.get(api, headers=headers, timeout=timeout)
    try:
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"GitHub Contents API failed: {api} ({r.status_code}) {r.text[:200]}") from e

    items = r.json()
    if not isinstance(items, list):
        raise RuntimeError(f"Unexpected GitHub API response at {api}: {items}")

    return [
        it["download_url"]
        for it in items
        if isinstance(it, dict) and it.get("type") == "file" and str(it.get("name", "")).endswith(suffix)
    ]


def _fetch_json_from_url(url: str, token: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
    headers = {}
    if token and "raw.githubusercontent.com" not in url:
        headers["Authorization"] = f"token {token}"
    r = requests.get(url, headers=headers, timeout=timeout)
    try:
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Fetch failed: {url} ({r.status_code}) {r.text[:200]}") from e
    return r.json()


def _iter_hf_samples(
    hf_obj: Dataset | DatasetDict,  # type: ignore[name-defined]
    arc_split: str,
    limit: Optional[int],
) -> Iterable[Dict[str, Any]]:
    """
    Assumes each HF row is a task dict like:
      { "train": [ {input, output}, ... ], "test": [ {input, output}, ... ] }
    If the HF builder exposes multiple splits, choose by arc_split.
    """
    taken = 0

    def take() -> bool:
        nonlocal taken
        if limit is not None and taken >= limit:
            return False
        taken += 1
        return True

    if isinstance(hf_obj, DatasetDict):  # type: ignore[name-defined]
        want_keys = []
        if arc_split in ("training", "all"):
            want_keys += ["training", "train"]
        if arc_split in ("evaluation", "all"):
            want_keys += ["evaluation", "test"]
        seen = set()
        for k in want_keys:
            if k in hf_obj and k not in seen:
                seen.add(k)
                for row in hf_obj[k]:
                    if not take():
                        return
                    yield row
    else:
        for row in hf_obj:
            if not take():
                return
            yield row


@register_dataset("arc_agi")
class ARCAGIDataset(BaseDataset):
    """
    ARC-AGI loader with two modes:
      - mode="huggingface": load from Hugging Face Hub (dataset_id default: "dataartist/arc-agi")
      - mode="github": list files via GitHub Contents API and fetch JSONs directly

    Expected per-task JSON format:
    {
      "train": [ {"input": [[...]], "output": [[...]]}, ... ],
      "test":  [ {"input": [[...]], "output": [[...]]}, ... ]
    }
    """

    def __init__(
        self,
        dataset_name: str = "arc_agi",
        split: str = "test",
        base_prompt_template: Optional[str] = DEFAULT_ARC_PROMPT_TEMPLATE,
        mode: str = "github",
        arc_split: str = "evaluation",
        multi_test: str = "first",
        include_reference: bool = True,
        limit: Optional[int] = None,
        hf_dataset_id: str = "dataartist/arc-agi",
        hf_name: Optional[str] = None,
        hf_revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        gh_owner: str = "fchollet",
        gh_repo: str = "ARC-AGI",
        gh_branch: str = "master",
        gh_root_dir: str = "data",
        gh_token: Optional[str] = None,
        gh_timeout: int = 30,
        **kwargs: Any
    ):
        subset_from_kwargs = kwargs.pop("subset", None)
        super().__init__(
            dataset_name,
            split=split,
            subset=subset_from_kwargs,
            base_prompt_template=base_prompt_template,
            **kwargs
        )

        self.mode = (mode or "huggingface").lower()
        if self.mode not in {"huggingface", "github"}:
            raise ValueError("mode must be 'huggingface' or 'github'")

        self.arc_split = (arc_split or "evaluation").lower()
        if self.arc_split not in {"training", "evaluation", "all"}:
            raise ValueError("arc_split must be 'training', 'evaluation', or 'all'")

        self.multi_test = "all" if str(multi_test).lower() == "all" else "first"
        self.include_reference = bool(include_reference)
        self.limit = int(limit) if (isinstance(limit, int) or (isinstance(limit, str) and str(limit).isdigit())) else None

        # HF fields
        self.hf_dataset_id = hf_dataset_id
        self.hf_name = hf_name
        self.hf_revision = hf_revision
        self.hf_token = hf_token

        # GitHub fields
        self.gh_owner = gh_owner
        self.gh_repo = gh_repo
        self.gh_branch = gh_branch
        self.gh_root_dir = gh_root_dir.strip("/")
        self.gh_token = gh_token
        self.gh_timeout = int(gh_timeout)

        self._github_urls: List[str] = []
        self._raw_tasks: Optional[List[Dict[str, Any]]] = None
        self._hf_dataset: Optional[Any] = None  # Dataset or DatasetDict

        if self.mode == "github":
            subdirs: List[str] = []
            if self.arc_split in ("training", "all"):
                subdirs.append(f"{self.gh_root_dir}/training")
            if self.arc_split in ("evaluation", "all"):
                subdirs.append(f"{self.gh_root_dir}/evaluation")

            urls: List[str] = []
            for sd in subdirs:
                try:
                    urls.extend(
                        _list_github_json_urls(
                            owner=self.gh_owner,
                            repo=self.gh_repo,
                            branch=self.gh_branch,
                            subdir=sd,
                            suffix=".json",
                            token=self.gh_token,
                            timeout=self.gh_timeout,
                        )
                    )
                except RuntimeError as e:
                    logger.warning(f"GitHub listing failed for {sd}: {e}")

            if not urls:
                logger.warning(
                    f"No ARC-AGI JSON files found via GitHub API under "
                    f"{self.gh_owner}/{self.gh_repo}@{self.gh_branch}/{self.gh_root_dir} "
                    f"(arc_split='{self.arc_split}')"
                )

            if self.limit:
                urls = urls[: self.limit]

            self._github_urls = urls

        else:
            if load_dataset is None:
                raise RuntimeError("datasets library not available. Install with `pip install datasets`.")
            load_kwargs: Dict[str, Any] = {}
            if self.hf_name:
                load_kwargs["name"] = self.hf_name
            if self.hf_revision:
                load_kwargs["revision"] = self.hf_revision
            if self.hf_token:
                load_kwargs["token"] = self.hf_token
            logger.info(
                f"Loading Hugging Face dataset: id={self.hf_dataset_id}, "
                f"name={self.hf_name}, revision={self.hf_revision}"
            )
            self._hf_dataset = load_dataset(self.hf_dataset_id, **load_kwargs)

        logger.info(
            f"ARC-AGI dataset initialized. mode={self.mode}, arc_split={self.arc_split}, "
            f"multi_test={self.multi_test}, limit={self.limit}"
        )

    def _iter_tasks(self) -> Iterable[Dict[str, Any]]:
        if self.mode == "github":
            for url in self._github_urls:
                try:
                    yield _fetch_json_from_url(url, token=self.gh_token, timeout=self.gh_timeout)
                except Exception as e:
                    logger.error(f"Failed to fetch/parse: {url} ({e})")
        else:
            assert self._hf_dataset is not None
            for task in _iter_hf_samples(self._hf_dataset, self.arc_split, self.limit):
                yield task

    def load(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        for task_idx, task in enumerate(self._iter_tasks()):
            test_pairs = task.get("test", [])
            if self.arc_split == "training":
                # Still use 'test' if present; we don't synthesize tests from 'train' here.
                pass

            all_references = [
                _serialize_grid(tp["output"]) for tp in test_pairs if "output" in tp
            ] or None

            iter_pairs = test_pairs if self.multi_test == "all" else test_pairs[:1]
            for local_idx, test_pair in enumerate(iter_pairs):
                prompt = _task_to_prompt(
                    task,
                    test_pair["input"],
                    base_prompt_template=self.base_prompt_template or DEFAULT_ARC_PROMPT_TEMPLATE,
                )
                reference = (
                    _serialize_grid(test_pair["output"])
                    if self.include_reference and "output" in test_pair
                    else None
                )

                samples.append(
                    {
                        "input": prompt,
                        "reference": reference,
                        "metadata": {
                            "id": f"task{task_idx}_test_{local_idx}",
                            "source_mode": self.mode,
                            "arc_split": self.arc_split,
                            "references_all": all_references,
                        },
                    }
                )

        return samples

    def get_raw_samples(self) -> Any:
        if self._raw_tasks is None:
            self._raw_tasks = list(self._iter_tasks())
        return self._raw_tasks

    def info(self) -> Dict[str, Any]:
        desc = "ARC-AGI loader supporting Hugging Face Hub and GitHub sources."
        base = {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "mode": self.mode,
            "arc_split": self.arc_split,
            "multi_test": self.multi_test,
            "include_reference": self.include_reference,
            "description": desc,
        }
        if self.mode == "github":
            base.update(
                {
                    "gh_owner": self.gh_owner,
                    "gh_repo": self.gh_repo,
                    "gh_branch": self.gh_branch,
                    "gh_root_dir": self.gh_root_dir,
                    "num_urls": len(self._github_urls),
                }
            )
        else:
            base.update(
                {
                    "hf_dataset_id": self.hf_dataset_id,
                    "hf_name": self.hf_name,
                    "hf_revision": self.hf_revision,
                }
            )
        return base

