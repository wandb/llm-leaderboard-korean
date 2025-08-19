import pytest
import yaml

# Try to import the required modules, skip tests if dependencies are missing
try:
    from llm_eval.datasets import BaseDataset, register_dataset
    from llm_eval.evaluator import run_from_config
    from llm_eval.models import BaseModel, register_model
    EVALUATOR_CONFIG_AVAILABLE = True
except ImportError as e:
    EVALUATOR_CONFIG_AVAILABLE = False
    pytest.skip(f"Skipping evaluator config tests due to missing dependencies: {e}", allow_module_level=True)


@register_dataset("dummy_dataset")
class DummyDataset(BaseDataset):
    def __init__(self, split: str = "test", **kwargs):
        super().__init__(dataset_name="dummy_dataset", split=split)

    def load(self):
        return [{"input": "hello", "reference": "hello"}]


@register_model("dummy_model")
class DummyModel(BaseModel):
    def generate_batch(self, inputs, return_logits=False, **kwargs):
        for sample in inputs:
            sample["prediction"] = sample["input"]
        return inputs


def test_run_from_config(tmp_path):
    cfg = {
        "dataset": {"name": "dummy_dataset", "split": "test", "params": {}},
        "model": {"name": "dummy_model", "params": {}},
        "evaluation": {"method": "string_match", "params": {}},
        "language_penalize": False,
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(cfg))

    result = run_from_config(str(config_file))
    assert result.metrics["accuracy"] == 1.0
    assert len(result.samples) == 1
