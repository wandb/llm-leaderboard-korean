import os
import logging
import pytest
import torch
import dotenv

dotenv.load_dotenv()

from llm_eval.evaluation.llm_judge import MultiLLMJudge, JudgeInput, JudgeType

@pytest.fixture(scope="module")
def judge():
    """
    pytest fixture: 모듈 범위에서 한 번만 실행.
    MultiLLMJudge 인스턴스를 생성 후 반환.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found. Skipping Judge tests.")

    logger = logging.getLogger("test_judge")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # 예시: 두 모델을 동시에 등록 (openai, huggingface)
    # 실제로는 본인 환경에서 유효한 모델/파라미터 설정 필요.
    models_config = [
        {
            "name": "openai",   # 레지스트리 키
            "type": "model",
            "api_key": api_key,
            "model_name": "gpt-4",  # 실제 사용 가능한 모델명
        },
        {
            "name": "huggingface", 
            "type": "model",
            "model_name_or_path": "google/gemma-2-2b-it",  # 예시(실제 사용 가능 여부 확인 필요)
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.95
        }
    ]

    # aggregation_strategy="all" -> "judge(...)"가 {"results": [...]} 형태로 반환
    multi_judge = MultiLLMJudge(
        models_config=models_config,
        aggregation_strategy="all",  # or "majority"/"first" 등
        logger=logger
    )
    return multi_judge

def test_rubric_judge(judge):
    """
    1) Rubric 방식 평가 테스트
    """
    rubric_input = JudgeInput(
        judge_type=JudgeType.RUBRIC_AND_RESPONSE,
        model_response="Python is a high-level programming language with clear syntax.",
        rubric="Assess the technical correctness and detail level."
    )

    result = judge.judge(rubric_input)
    print("\n=== 1. Rubric Evaluation Results ===")
    print(result)

    # 예: aggregator="all" => result: {"results": [ { "score": ..., "model_name":"..." }, {...} ]}
    assert "results" in result, "Expected key 'results' in the output"
    parsed_results = result["results"]
    assert isinstance(parsed_results, list), "'results' field should be a list"

    # 각 파싱 결과에 대해 최소한 1개 키(score/winner/correct 등)가 있어야 함
    for pr in parsed_results:
        # 예: "score"가 있을 수도 있고, "correct"가 있을 수도 있음
        assert any(key in pr for key in ["score", "winner", "correct"]), (
            "Parsed result should contain at least 'score', 'winner', or 'correct'"
        )

def test_gold_judge(judge):
    """
    2) Gold answer와의 비교 테스트
    """
    gold_input = JudgeInput(
        judge_type=JudgeType.RUBRIC_RESPONSE_AND_GOLD,
        model_response="Python uses indentation for code blocks.",
        gold_response="Python uses indentation to define code blocks and structure.",
        rubric="Check the correctness with respect to the gold answer."
    )

    result = judge.judge(gold_input)
    print("\n=== 2. Gold Answer Comparison Results ===")
    print(result)

    assert "results" in result, "Expected key 'results' in the output"
    parsed_results = result["results"]
    assert isinstance(parsed_results, list), "'results' field should be a list"
    
    # 예: GoldComparisonParser => {"correct": bool, "step": int, "model_name":...}
    for pr in parsed_results:
        assert any(key in pr for key in ["score", "winner", "correct"]), (
            "Parsed result should contain at least 'score', 'winner', or 'correct'"
        )

def test_comparison_judge(judge):
    """
    3) Model response 간 비교 테스트
    """
    comparison_input = JudgeInput(
        judge_type=JudgeType.RESPONSE_COMPARISON,
        model_response="Python is a versatile programming language.",
        model_response_b="Python is a popular language used in many fields."
    )

    result = judge.judge(comparison_input)
    print("\n=== 3. Response Comparison Results ===")
    print(result)

    assert "results" in result, "Expected key 'results' in the output"
    parsed_results = result["results"]
    assert isinstance(parsed_results, list), "'results' field should be a list"

    # PairwiseComparisonParser => {"winner": "A"/"B"/"tie", "model_name":...}
    for pr in parsed_results:
        assert any(key in pr for key in ["winner", "score", "correct"]), (
            "Parsed result should contain at least 'winner', 'score', or 'correct'"
        )
