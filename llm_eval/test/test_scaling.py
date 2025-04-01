import pytest
from llm_eval.scaling_methods import SCALING_REGISTRY, load_scaling_method
from llm_eval.models.base import BaseModel  # Import BaseModel

# Get scaling method keys from the registry
scaling_keys = list(SCALING_REGISTRY.keys())

@pytest.mark.parametrize("scaler_key", scaling_keys)
def test_scaler_registration(scaler_key):
    """
    Test scaling method registration.

    This test checks if the scaling method is correctly registered and can be
    instantiated using load_scaling_method().
    """
    scaler = load_scaling_method(scaler_key)
    assert scaler is not None, f"Scaler not found: {scaler_key}"

@pytest.mark.parametrize("scaler_key", scaling_keys)
def test_scaler_apply(scaler_key):
    """
    Test the apply() method of each scaling method.

    This test verifies that the apply() method returns a list and that each item
    in the list contains the 'prediction' key.
    """
    scaler = load_scaling_method(scaler_key)

    # Mock model for testing (using BaseModel as a base)
    class MockModel(BaseModel):
        def generate_batch(self, inputs, return_logits=False, **kwargs):
            return [{"prediction": "Hello world"} for _ in inputs]

    mock_model = MockModel()
    scaler.model = mock_model  # Assign the mock model to the scaler

    test_data = [{"input": "Test 1", "reference": "Ref 1"}]
    try:
        output_data = scaler.apply(test_data)
        assert isinstance(output_data, list), "apply() should return a list."
        assert len(output_data) == len(test_data), "Output should have the same length as input."
        for item in output_data:
            assert "prediction" in item, "Each item should have 'prediction' key."
    except Exception as e:
        pytest.fail(f"Scaler {scaler_key} failed during apply() call: {e}")