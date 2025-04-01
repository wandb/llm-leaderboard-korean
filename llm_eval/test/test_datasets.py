import pytest
from llm_eval.datasets import DATASET_REGISTRY, load_datasets

# Get dataset keys from the registry
dataset_keys = list(DATASET_REGISTRY.keys())

@pytest.mark.parametrize("dataset_key", dataset_keys)
def test_dataset_loading(dataset_key):
    """
    Test dataset loading and instance creation.

    This test checks the following:
    1) Whether the dataset key is correctly registered in the DATASET_REGISTRY.
    2) Whether load_datasets() successfully creates an instance without errors.
    3) Whether the loaded instance has the 'load' method.
    """
    try:
        ds = load_datasets(dataset_key)
        assert ds is not None, f"Failed to load dataset: {dataset_key}"
        assert hasattr(ds, 'load'), f"Dataset {dataset_key} does not have a 'load' method."
    except Exception as e:
        pytest.fail(f"Dataset loading failed for {dataset_key}: {e}")

@pytest.mark.parametrize("dataset_key", dataset_keys)
def test_dataset_load_output(dataset_key):
    """
    Test the output format of the load() method.

    This test verifies that the load() method of each dataset returns a list
    and that each item in the list is a dictionary containing 'input' and 'reference' keys.
    """
    ds = load_datasets(dataset_key)
    data = ds.load()
    assert isinstance(data, list), "load() should return a list."
    if data:
        assert isinstance(data[0], dict), "Each item should be a dictionary."
        assert "input" in data[0], "Each item should have 'input' key."
        assert "reference" in data[0], "Each item should have 'reference' key."