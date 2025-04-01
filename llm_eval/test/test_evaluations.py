import pytest
from llm_eval.evaluation import EVALUATION_REGISTRY, get_evaluator
from llm_eval.models.base import BaseModel  # Import BaseModel

# Get evaluator keys from the registry
eval_keys = list(EVALUATION_REGISTRY.keys())

@pytest.mark.parametrize("eval_key", eval_keys)
def test_evaluator_registration(eval_key):
    """
    Test evaluator registration and instance creation.

    This test checks if the evaluator is correctly registered and can be
    instantiated using get_evaluator().
    """
    evaluator = get_evaluator(eval_key)
    assert evaluator is not None, f"Evaluator not found: {eval_key}"

@pytest.mark.parametrize("eval_key", eval_keys)
def test_evaluator_evaluate(eval_key):
    """
    Test the evaluate() method of each evaluator.

    This test verifies that the evaluate() method returns a dictionary
    containing 'metrics' and 'samples' keys.
    """
    evaluator = get_evaluator(eval_key)

    # Mock model for testing (using BaseModel as a base)
    class MockModel(BaseModel):
        def generate_batch(self, inputs, return_logits=False, **kwargs):
            return [{"prediction": "test prediction"} for _ in inputs]

    mock_model = MockModel()

    # Test data
    test_data = [
        {"input": "test input", "reference": "test reference", "prediction": "test prediction"}
    ]
    try:
        results = evaluator.evaluate(test_data, model=mock_model)
        assert isinstance(results, dict), "evaluate() should return a dictionary."
        assert "metrics" in results, "Results should contain 'metrics' key."
        assert "samples" in results, "Results should contain 'samples' key."
    except Exception as e:
        pytest.fail(f"Evaluator {eval_key} failed during evaluate() call: {e}")