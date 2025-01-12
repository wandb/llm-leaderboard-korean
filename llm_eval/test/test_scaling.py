import pytest
import yaml
from pathlib import Path

from llm_eval.scaling_methods import load_scaling_method
from llm_eval.models.base import BaseModel

CONFIG_FILE = Path(__file__).parent / "test_config.yaml"
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    test_config = yaml.safe_load(f)

SCALING_KEYS = test_config.get("scalers", [])

@pytest.mark.parametrize("scaler_key", SCALING_KEYS)
def test_scaler_registration(scaler_key):
    scaler = load_scaling_method(scaler_key)
    assert scaler is not None, f"Scaler not found: {scaler_key}"

@pytest.mark.parametrize("scaler_key", SCALING_KEYS)
def test_scaler_apply(scaler_key):
    """
    각 scaler에 대해 간단히 apply() 로직이 동작하는지 확인.
    """
    scaler = load_scaling_method(scaler_key)

    # 모의 모델(언제나 "Hello world"만만 반환)
    class FakeModel(BaseModel):
        def generate_batch(self, inputs, return_logits=False):
            return [
                {
                    "input": inp["input"],
                    "reference": inp["reference"],
                    "prediction": "Hello world",
                }
                for inp in inputs
            ]
    
    scaler.model = FakeModel()
    test_data = [{"input": "Test 1", "reference": "Ref 1"}]
    output_data = scaler.apply(test_data)
    assert len(output_data) == len(test_data)
    for item in output_data:
        assert "prediction" in item
        assert item["prediction"] == "Hello world"
