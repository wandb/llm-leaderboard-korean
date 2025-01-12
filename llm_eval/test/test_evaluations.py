import pytest
import yaml
from pathlib import Path
from llm_eval.evaluation import get_evaluator

CONFIG_FILE = Path(__file__).parent / "test_config.yaml"
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    test_config = yaml.safe_load(f)

EVAL_KEYS = test_config.get("evaluations", [])

@pytest.mark.parametrize("eval_key", EVAL_KEYS)
def test_evaluator_registration(eval_key):
    evaluator = get_evaluator(eval_key)
    assert evaluator is not None, f"Evaluator not found: {eval_key}"

@pytest.mark.parametrize("eval_key", EVAL_KEYS)
def test_evaluator_simple_run(eval_key):
    evaluator = get_evaluator(eval_key)
    data = [{"input":"Q", "reference":"A", "prediction":"A"}]

    # 모의(Fake) 모델 (evaluate()가 model을 필요로 할 수 있으니,,)
    class FakeModel: pass
    fake_model = FakeModel()

    results = evaluator.evaluate(data, fake_model)
    assert "metrics" in results
    assert "samples" in results
