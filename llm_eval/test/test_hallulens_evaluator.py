from llm_eval.datasets import load_datasets

def test_halluLens_dataset_loading():
    # Load HalluLens dataset
    loader = load_datasets(name="halluLens", split="test")
    all_samples_path = loader.load()

    assert len(all_samples_path) == 4

    # Test subset keys
    expected_keys = ['precise_wikiqa', 'longwiki']
    subset_loader = load_datasets(name="halluLens", subset=expected_keys, split="test")
    subset_samples_path = subset_loader.load()

    assert set(subset_samples_path.keys()) == set(expected_keys)
