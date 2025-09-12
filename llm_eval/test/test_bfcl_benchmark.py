import os
import sys
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

# Add the repo root directory to sys.path to ensure llm_eval can be imported
# The test file is at llm_eval/test/test_bfcl_benchmark.py
# So we need to go up 2 levels to reach the repo root where llm_eval package is located
current_dir = Path(__file__).resolve().parent  # llm_eval/test/
repo_root = current_dir.parent.parent  # repo root (contains llm_eval/)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from llm_eval.external import get_provider
from llm_eval.datasets.bfcl import BFCLDataset


def _bfcl_available() -> bool:
    """Check if BFCL provider is available."""
    try:
        provider = get_provider("bfcl")
        return True
    except Exception:
        print("[SKIP] bfcl provider not available.")
        return False


def _pick_models(provider) -> list:
    """Select test models for benchmarking."""
    return ["gpt-4.1-nano-2025-04-14", "gpt-4.1-nano-2025-04-14-FC"]


def test_provider_info():
    """Test provider information retrieval and display."""
    if not _bfcl_available():
        return

    provider = get_provider("bfcl")
    # Ensure data root is set (defaults to repo-local path if not provided)
    provider.paths()
    models = provider.list_models()
    cats = provider.list_categories()
    paths = provider.paths()

    print("=" * 60)
    print("BFCL Provider Information")
    print("=" * 60)

    print(f"\nModels ({len(models)} total):")
    for i, model in enumerate(models[:10], 1):  # Show first 10 models
        print(f"  {i:2d}. {model}")
    if len(models) > 10:
        print(f"  ... and {len(models) - 10} more models")

    print(f"\nTest Categories ({len(cats)} total):")
    for category, subcats in cats.items():
        print(f"  {category}: {len(subcats)} subcategories")
        if subcats and len(subcats) <= 5:  # Show subcategories if few enough
            for subcat in subcats:
                print(f"    - {subcat}")
        elif subcats:
            print(f"    - {subcats[0]} ... and {len(subcats) - 1} more")

    print("\nPaths Configuration:")
    for key, path in paths.items():
        print(f"  {key:30s}: {path}")

    print("=" * 60)


def test_dataset_loading():
    """Test basic dataset loading functionality."""
    try:
        ds = BFCLDataset(auto_download=True, use_full_dataset=False)  # Auto-download for test

        # Test loading samples
        data = ds.load()
        print(f"Loaded {len(data)} samples")
        assert len(data) > 0, "No samples loaded"

        # Test sample structure
        sample = data[0]
        assert "input" in sample, "Sample missing 'input' field"
        assert "reference" in sample, "Sample missing 'reference' field"
        assert "metadata" in sample, "Sample missing 'metadata' field"

        metadata = sample["metadata"]
        assert "id" in metadata, "Metadata missing 'id' field"
        assert "category" in metadata, "Metadata missing 'category' field"
        assert "functions" in metadata, "Metadata missing 'functions' field"

        print("✅ Dataset loading test passed")

        # Additional validation
        print(f"Sample input preview: {sample['input'][:100]}...")
        print(f"Sample category: {metadata['category']}")
        print(f"Functions available: {len(metadata.get('functions', []))}")

    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        raise


def test_simple_generation():
    """Test model generation functionality with simple_python category."""
    if not _bfcl_available():
        return

    provider = get_provider("bfcl")
    # Ensure data root and project root are initialized
    provider.paths()
    models = _pick_models(provider)
    provider.generate(
        models=models,
        test_categories=["simple_python"],  # use a light category for smoke test
        # Optional kwargs: temperature=0.001, backend="sglang", run_ids=True, allow_overwrite=False
    )


def test_generation_all_categories():
    """Test model generation functionality with all categories."""
    if not _bfcl_available():
        return

    provider = get_provider("bfcl")
    # Ensure data root and project root are initialized
    provider.paths()
    models = _pick_models(provider)
    provider.generate(
        models=models,
        test_categories=["all"],  # test all categories
        # Optional kwargs: temperature=0.001, backend="sglang", run_ids=True, allow_overwrite=False
    )


def test_simple_evaluation():
    """Test model evaluation functionality with simple_python category."""
    if not _bfcl_available():
        return

    provider = get_provider("bfcl")
    provider.paths()
    models = _pick_models(provider)
    provider.evaluate(
        models=models,
        test_categories=["simple_python"],
    )


def test_evaluation_all_categories():
    """Test model evaluation functionality with all categories."""
    if not _bfcl_available():
        return

    provider = get_provider("bfcl")
    provider.paths()
    models = _pick_models(provider)
    provider.evaluate(
        models=models,
        test_categories=["all"],
    )


def test_load_scores():
    """Test score loading functionality."""
    if not _bfcl_available():
        return

    provider = get_provider("bfcl")
    provider.paths()
    scores = provider.load_scores()
    print("Scores loaded:", bool(scores.get("exists")), "rows:", len(scores.get("rows", [])))


if __name__ == "__main__":
    print("Running BFCL benchmark tests...")

    # Basic tests
    test_provider_info()
    test_dataset_loading()

    # # Simple tests
    # test_simple_generation()
    # test_simple_evaluation()
    # test_load_scores()

    # # Full category tests
    # test_generation_all_categories()
    # test_evaluation_all_categories()
    # test_load_scores()

    print("Tests completed.")
