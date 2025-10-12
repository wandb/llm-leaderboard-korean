import pytest

from llm_eval.evaluation import get_evaluator


@pytest.mark.parametrize(
    "samples,expected_score,expected_miss",
    [
        (
            [
                {
                    "input": [
                        {
                            "role": "user",
                            "content": "Here are the MRCR instructions.",
                        }
                    ],
                    "reference": "prefixCorrect poem",
                    "prediction": "prefixCorrect poem",
                    "metadata": {
                        "random_string_to_prepend": "prefix",
                        "model_name": "openai/gpt-4o",
                    },
                },
                {
                    "input": [
                        {
                            "role": "user",
                            "content": "Return the second poem.",
                        }
                    ],
                    "reference": "prefixTarget stanza",
                    "prediction": "Target stanza",
                    "metadata": {
                        "random_string_to_prepend": "prefix",
                        "model_name": "openai/gpt-4o",
                    },
                },
            ],
            0.5,
            0.5,
        ),
    ],
)
def test_sequence_match_for_mrcr_with_openai(samples, expected_score, expected_miss):
    evaluator = get_evaluator("sequence_match")

    metrics = evaluator.evaluate_predictions(samples)

    assert pytest.approx(metrics["sequence_match_score"], rel=1e-6) == expected_score
    assert pytest.approx(
        metrics["sequence_match_prefix_miss_rate"], rel=1e-6
    ) == expected_miss

    # 첫 번째 샘플은 prefix를 지킨 gpt-4o 응답으로 간주하므로 ratio가 1.0이어야 한다.
    assert samples[0]["evaluation"]["sequence_match_prefix_present"] is True
    assert samples[0]["evaluation"]["sequence_match_ratio"] == pytest.approx(1.0)

    # 두 번째 샘플은 prefix가 누락된 경우 → ratio 0, miss rate 반영
    assert samples[1]["evaluation"]["sequence_match_prefix_present"] is False
    assert samples[1]["evaluation"]["sequence_match_ratio"] == pytest.approx(0.0)

