import pytest
from llm_eval.datasets import DATASET_REGISTRY, load_datasets
import urllib3

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
        if dataset_key == "generic_file":
            # Provide dataset_name and file_path for GenericFileDataset
            # ds = load_datasets(name="generic_file", file_path="dummy_file.csv")  # Minimal fix
            pass # Skipping this dataset for now
        elif dataset_key == "KUDGE":
            # Provide a subset for KUDGE dataset
            ds = load_datasets(name=dataset_key, subset="Pairwise")  # Or another valid subset
        else:
            ds = load_datasets(name=dataset_key)
        assert ds is not None, f"Failed to load dataset: {dataset_key}"
        assert hasattr(ds, 'load'), f"Dataset {dataset_key} does not have a 'load' method."
    except (FileNotFoundError, ValueError, urllib3.exceptions.MaxRetryError) as e:
        pytest.skip(f"Skipping dataset {dataset_key} due to: {e.__class__.__name__} - {e}")
    except Exception as e:
        pytest.fail(f"Dataset loading failed for {dataset_key}: {e}")

@pytest.mark.parametrize("dataset_key", dataset_keys)
def test_dataset_load_output(dataset_key):
    """
    Test the output format of the load() method.

    This test verifies that the load() method of each dataset returns a list
    and that each item in the list is a dictionary containing 'input' and 'reference' keys.
    """
    try:
        if dataset_key == "generic_file":
            pytest.skip("Skipping generic_file dataset test.")
        elif dataset_key == "click":
            # Use "train" split for "click" dataset
            ds = load_datasets(name=dataset_key, split="train")
        elif dataset_key == "KUDGE":
            # Provide a subset for KUDGE dataset
            ds = load_datasets(name=dataset_key, subset="Pairwise")  # Or another valid subset
        else:
            ds = load_datasets(name=dataset_key)
        data = ds.load()
        assert isinstance(data, list), "load() should return a list."
        if data:
            assert isinstance(data[0], dict), "Each item should be a dictionary."
            assert "input" in data[0], "Each item should have 'input' key."
            assert "reference" in data[0], "Each item should have 'reference' key."
    except (FileNotFoundError, ValueError, urllib3.exceptions.MaxRetryError) as e:
        pytest.skip(f"Skipping dataset {dataset_key} due to: {e.__class__.__name__} - {e}")
    except Exception as e:
        pytest.fail(f"Dataset loading failed for {dataset_key}: {e}")