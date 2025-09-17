from dotenv import load_dotenv

from llm_eval.datasets import load_datasets
from llm_eval.evaluation.llm_judge import LLMJudgeEvaluator
from llm_eval.evaluator import Evaluator
from llm_eval.models.multi import MultiModel
import pytest
import openai
import os


@pytest.mark.integration
def test_korean_sat_llm_judge():
    # Load Korean SAT dataset (test split)
    loader = load_datasets(name="korean_sat", split="test")
    samples = loader.load()
    assert len(samples) > 0, "No samples loaded from Korean SAT dataset."

    # Prepare dummy model that returns the reference answer as prediction
    class DummyModel(MultiModel):
        def judge_batch(self, batch_inputs):
            # Always return the reference answer as the judge's prediction
            return [{"prediction": sample["reference"]} for sample in samples]

    # Instantiate LLMJudgeEvaluator with dummy model and Korean SAT judge type
    evaluator = LLMJudgeEvaluator(model=DummyModel(), default_judge_type="korean_sat_eval")
    metrics = evaluator.evaluate_predictions(samples)

    # Check that the score is as expected (all correct)
    total_score = sum(int(s["metadata"]["score"]) for s in samples)
    assert metrics["average_score_per_item"] == total_score / len(samples)
    assert len(samples) == len(metrics['korean_sat_result'])
    print("Korean SAT LLM Judge test passed. Metrics:", metrics)
