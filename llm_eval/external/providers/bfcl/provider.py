from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import os
import sys
from pathlib import Path
import shutil

from llm_eval.external.provider_base import ExternalProvider
from llm_eval.external.registry import register_provider


def _require_bfcl_eval():
    try:
        import bfcl_eval  # noqa: F401

        return True
    except Exception:
        # Try to import from a vendored in-repo copy if available
        try:
            repo_root = Path(__file__).resolve().parents[3]
            # Preferred order: explicit env → external/bfcl_eval
            env_vendor_root = os.getenv("BFCL_EVAL_VENDOR_ROOT")
            candidates = []
            if env_vendor_root:
                candidates.append(Path(env_vendor_root))
            candidates.append(repo_root / "external" / "bfcl_eval")

            added_any = False
            for root in candidates:
                if not root.exists():
                    continue
                for p in [root, root / "bfcl_eval"]:
                    if p.exists():
                        p_str = str(p)
                        if p_str not in sys.path:
                            sys.path.insert(0, p_str)
                            added_any = True

            if not added_any:
                raise ImportError("No vendored bfcl_eval found in expected locations.")

            import bfcl_eval  # noqa: F401  # re-attempt after path injection

            return True
        except Exception as e:
            raise ImportError(
                (
                    "bfcl_eval is not importable. Install it (pip install -e) or place a vendored copy at:\n"
                    "  <repo>/external/bfcl_eval\n"
                    "or set BFCL_EVAL_VENDOR_ROOT to the bfcl_eval folder cloned from the Gorilla BFCL directory: \n"
                    "  https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval"
                )
            ) from e


@register_provider("bfcl")
class BFCLProvider(ExternalProvider):
    """
    BFCL (Berkeley Function Call Leaderboard) Provider

    Environment Setup:
    - Set BFCL_PROJECT_ROOT environment variable to change the project root directory
      where results and scores are stored. If not set, defaults to a provider-specific workspace.
    - Example: export BFCL_PROJECT_ROOT=/path/to/your/bfcl/workspace

    Data Setup:
    - PROMPT_PATH is where BFCL reads prompts/answers/configs.
    - Defaults to <repo>/llm_eval/datasets/bfcl and is auto-created.

    The PROJECT_ROOT determines where the following directories are created:
    - result/: Generated model responses
    - score/: Evaluation scores
    - test_case_ids_to_generate.json: Test case configuration
    - .env: Environment configuration
    """

    def __init__(self):
        """Initialize the BFCL provider and set up a default project root."""
        super().__init__()
        self._project_root_configured = False
        self._default_project_root = None
        self._data_root_configured = False
        self._default_data_root = None

    def _get_default_data_root(self) -> str:
        """Get the default local data root for BFCL prompts and resources."""
        if self._default_data_root is None:
            # Put sampled/original data copy under repo's datasets directory
            llm_eval_dir = Path(__file__).resolve().parents[3]
            self._default_data_root = str(llm_eval_dir / "datasets" / "bfcl")
        return self._default_data_root

    def _get_default_project_root(self) -> str:
        """Get the default project root path for this provider."""
        if self._default_project_root is None:
            # Create a provider-specific workspace in the current working directory
            # This keeps BFCL results organized and separate from other evaluation results
            self._default_project_root = str(Path.cwd() / "hret_results" / "_bfcl_workspace")
        return self._default_project_root

    def _ensure_project_root(self, project_root: Optional[str] = None) -> None:
        """Ensure project root is configured."""
        if self._project_root_configured and project_root is None:
            return

        if project_root is not None:
            os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
        elif "BFCL_PROJECT_ROOT" not in os.environ:
            # Use provider-specific default workspace
            default_root = self._get_default_project_root()
            os.environ["BFCL_PROJECT_ROOT"] = default_root
            # Ensure the directory exists
            Path(default_root).mkdir(parents=True, exist_ok=True)

        # Verify the project root directory exists and is accessible
        current_project_root = os.environ.get("BFCL_PROJECT_ROOT")
        if current_project_root:
            project_path = Path(current_project_root)
            if not project_path.exists():
                try:
                    project_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise RuntimeError(f"Cannot create project root directory '{current_project_root}': {e}")
            elif not project_path.is_dir():
                raise RuntimeError(f"Project root path '{current_project_root}' exists but is not a directory")
            elif not os.access(project_path, os.R_OK | os.W_OK):
                raise RuntimeError(f"Project root directory '{current_project_root}' is not readable/writable")

        self._project_root_configured = True

    def _ensure_data_root(self, data_root: Optional[str] = None) -> None:
        """Ensure data root is configured."""
        if self._data_root_configured and data_root is None:
            return

        # Resolve target root
        target_root = Path(data_root if data_root is not None else self._get_default_data_root())
        target_root.mkdir(parents=True, exist_ok=True)

        # Make sure bfcl_eval is importable first
        _require_bfcl_eval()

        # Override constants used by bfcl_eval to locate data
        from bfcl_eval.constants import eval_config as ec
        from bfcl_eval.constants.category_mapping import VERSION_PREFIX

        # Ensure we store as Path
        ec.PROMPT_PATH = Path(target_root)
        ec.MULTI_TURN_FUNC_DOC_PATH = ec.PROMPT_PATH / "multi_turn_func_doc"
        ec.POSSIBLE_ANSWER_PATH = ec.PROMPT_PATH / "possible_answer"
        ec.MEMORY_PREREQ_CONVERSATION_PATH = ec.PROMPT_PATH / "memory_prereq_conversation"
        ec.FORMAT_SENSITIVITY_IDS_PATH = ec.PROMPT_PATH / f"{VERSION_PREFIX}_format_sensitivity.json"

        self._data_root_configured = True

    def setup_data_root(self, data_root: Optional[str] = None) -> None:
        """
        Configure a local data root for BFCL to load prompts/answers from.

        Priority:
        1) Explicit argument
        2) Provider default (llm_eval/datasets/bfcl)

        This function overrides bfcl_eval's data path constants at runtime,
        without modifying the upstream package.
        """
        self._ensure_data_root(data_root)

    def setup_project_root(self, project_root: Optional[str] = None) -> None:
        """
        Set up the BFCL_PROJECT_ROOT environment variable.

        Args:
            project_root: Path to the desired project root directory.
                         If None, uses the current BFCL_PROJECT_ROOT env var or provider default.
        """
        self._ensure_project_root(project_root)

    def list_categories(self) -> Dict[str, List[str]]:
        self._ensure_project_root()
        _require_bfcl_eval()
        from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING

        return dict(TEST_COLLECTION_MAPPING)

    def list_models(self) -> List[str]:
        self._ensure_project_root()
        _require_bfcl_eval()
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

        return list(MODEL_CONFIG_MAPPING.keys())

    def paths(self, project_root: Optional[str] = None) -> Dict[str, Any]:
        """
        Get BFCL paths configuration.

        Args:
            project_root: Optional project root to set before retrieving paths.
                         If provided, will set BFCL_PROJECT_ROOT environment variable.

        Returns:
            Dictionary containing all relevant BFCL paths.
        """
        self._ensure_project_root(project_root)
        self._ensure_data_root()

        _require_bfcl_eval()
        from bfcl_eval.constants.eval_config import (
            PROJECT_ROOT,
            RESULT_PATH,
            SCORE_PATH,
            DOTENV_PATH,
            TEST_IDS_TO_GENERATE_PATH,
            PROMPT_PATH,
            MULTI_TURN_FUNC_DOC_PATH,
            POSSIBLE_ANSWER_PATH,
            MEMORY_PREREQ_CONVERSATION_PATH,
            FORMAT_SENSITIVITY_IDS_PATH,
        )

        return {
            "project_root": str(PROJECT_ROOT),
            "result_path": str(RESULT_PATH),
            "score_path": str(SCORE_PATH),
            "dotenv_path": str(DOTENV_PATH),
            "test_ids_to_generate_path": str(TEST_IDS_TO_GENERATE_PATH),
            "prompt_path": str(PROMPT_PATH),
            "multi_turn_func_doc_path": str(MULTI_TURN_FUNC_DOC_PATH),
            "possible_answer_path": str(POSSIBLE_ANSWER_PATH),
            "memory_prereq_conversation_path": str(MEMORY_PREREQ_CONVERSATION_PATH),
            "format_sensitivity_ids_path": str(FORMAT_SENSITIVITY_IDS_PATH),
        }

    def generate(
        self,
        models: List[str],
        test_categories: Optional[List[str]] = None,
        temperature: float = 0.001,
        include_input_log: bool = False,
        exclude_state_log: bool = False,
        num_gpus: int = 1,
        num_threads: int = 1,
        gpu_memory_utilization: float = 0.9,
        backend: str = "sglang",
        skip_server_setup: bool = False,
        local_model_path: Optional[str] = None,
        result_dir: Optional[str] = None,
        allow_overwrite: bool = False,
        run_ids: bool = False,
        **kwargs,
    ) -> None:
        self._ensure_project_root()
        self._ensure_data_root()

        _require_bfcl_eval()
        from bfcl_eval._llm_response_generation import main as generation_main
        from bfcl_eval.constants.eval_config import RESULT_PATH

        if test_categories is None:
            test_categories = ["all"]

        args = SimpleNamespace(
            model=models,
            test_category=test_categories,
            temperature=temperature,
            include_input_log=include_input_log,
            exclude_state_log=exclude_state_log,
            num_gpus=num_gpus,
            num_threads=num_threads,
            gpu_memory_utilization=gpu_memory_utilization,
            backend=backend,
            skip_server_setup=skip_server_setup,
            local_model_path=local_model_path,
            result_dir=(result_dir or RESULT_PATH),
            allow_overwrite=allow_overwrite,
            run_ids=run_ids,
        )
        generation_main(args)

    def evaluate(
        self,
        models: Optional[List[str]] = None,
        test_categories: Optional[List[str]] = None,
        result_dir: Optional[str] = None,
        score_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._ensure_project_root()
        self._ensure_data_root()

        # Use local eval_runner instead of external bfcl_eval
        from .eval_runner import main as evaluation_main

        evaluation_main(
            models,
            test_categories,
            result_dir,
            score_dir,
        )

    def load_scores(self, score_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        self._ensure_project_root()

        from pathlib import Path
        import csv

        _require_bfcl_eval()
        from bfcl_eval.constants.eval_config import SCORE_PATH

        path = Path(score_dir) if score_dir else Path(SCORE_PATH)
        file = path / "data_overall.csv"
        if not file.exists():
            return {"exists": False, "path": str(file), "rows": []}

        with open(file, newline="") as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader, [])
            rows = [dict(zip(headers, row)) for row in reader]

        return {"exists": True, "path": str(file), "rows": rows}
