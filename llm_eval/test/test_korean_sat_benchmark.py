from llm_eval.datasets import load_datasets
from llm_eval.evaluation.llm_judge import LLMJudgeEvaluator
from llm_eval.models.multi import MultiModel
import pytest

def test_evaluation():
    from llm_eval.evaluator import Evaluator
    # 1) Initialize an Evaluator.
    evaluator = Evaluator()
    import os
    # 2) Run the evaluation pipeline
    results = evaluator.run(
        model="openai",  # or "litellm", "openai", etc.
        model_params={"api_base": "https://api.openai.com/v1",
                      "model_name": "gpt-3.5-turbo",
                      "batch_size" : 10},

        dataset="korean_sat",
        evaluation_method="llm_judge",
        judge_model="openai_judge",
        judge_params={
            "api_base": "https://api.openai.com/v1/",
            "model_name": "gpt-4o-mini",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "batch_size": 5
        },
    )

    print(results)

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
