import pytest
import yaml
from pathlib import Path
from llm_eval.datasets import load_datasets

# 1) 테스트 대상이 되는 객체의 정보가 담긴 YAML 로드
CONFIG_FILE = Path(__file__).parent / "test_config.yaml"
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    test_config = yaml.safe_load(f)

# 2) datasets 목록 가져오기
DATASET_KEYS = test_config.get("datasets", [])

@pytest.mark.parametrize("dataset_key", DATASET_KEYS)
def test_dataset_registration(dataset_key):
    """
    1) dataset_key가 레지스트리에 정상 등록되어 있는지
    2) load_dataset() 호출 시 에러 없이 인스턴스가 생성되는지
    """
    ds = load_datasets(dataset_key, split="test")
    assert ds is not None, f"Failed to load dataset: {dataset_key}"

@pytest.mark.parametrize("dataset_key", DATASET_KEYS)
def test_dataset_load_output(dataset_key):
    """
    등록된 모든 데이터셋이 실제로 load()를 통해
    [{'input':..., 'reference':...}, ...] 형태를 반환하는지 확인.
    """
    ds = load_datasets(dataset_key, split="test")
    data = ds.load()
    assert isinstance(data, list), "load() should return a list."
    if data:
        assert "input" in data[0]
        assert "reference" in data[0]
