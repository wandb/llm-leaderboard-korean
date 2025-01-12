import pytest
import yaml
from pathlib import Path
from llm_eval.models import load_model

CONFIG_FILE = Path(__file__).parent / "test_config.yaml"
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    test_config = yaml.safe_load(f)

MODEL_KEYS = test_config.get("models", [])

@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_model_registration(model_key):
    """
    모델이 레지스트리에 등록되어 있고, load_model()로 생성 가능한지 확인.
    """
    model = load_model(model_key, endpoint="http://fake-endpoint:8000") # TODO: fake-endpoint면 model class에서 fail이 뜰 수 있을 것 같은데. 고민.
    assert model is not None, f"Failed to load model: {model_key}"

@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_model_generate_batch(model_key):
    """
    단순 샘플 입력에 대해 model.generate_batch()가
    'prediction' 필드를 반환하는지 확인.
    """
    model = load_model(model_key, endpoint="http://fake-endpoint:8000") # TODO: 위와 동일
    sample_input = [{"input": "Hello?", "reference": "Hi."}]
    outputs = model.generate_batch(sample_input)
    assert isinstance(outputs, list)
    assert len(outputs) == len(sample_input)
    assert "prediction" in outputs[0], "No 'prediction' in model output."
