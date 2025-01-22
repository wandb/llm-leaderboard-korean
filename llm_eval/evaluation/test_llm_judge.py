import torch
from .llm_judge import MultiLLMJudge, JudgeInput, JudgeType  # 절대 경로 대신 상대 경로 사용
import dotenv
import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Load environment variables from .env file
dotenv.load_dotenv()

def test_multi_judge():
    # OpenAI API 키가 제대로 설정되어 있는지 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
        
    # Configure multiple models with type specification
    models_config = [
        {
            "name": "openai",
            "type": "model",  # type 필드 추가
            "api_key": api_key,
            "model_name": "gpt-4o"
        },
        {
            "name": "openai",
            "type": "model",  # type 필드 추가
            "api_key": api_key,
            "model_name": "gpt-3.5-turbo"
        }
    ]

    # Initialize MultiLLMJudge with logging
    logger = logging.getLogger("test_judge")
    logger.setLevel(logging.DEBUG)  # 디버그 레벨로 설정
    
    # 콘솔에 로그 출력을 위한 핸들러 추가
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    judge = MultiLLMJudge(
        models_config=models_config,
        aggregation_strategy="first",  # "majority", "first", "all" 중 선택
        logger=logger
    )

    def test_rubric_evaluation():
        """Test rubric-based single response evaluation"""
        test_input = JudgeInput(
            judge_type=JudgeType.RUBRIC_AND_RESPONSE,
            rubric="""Evaluation criteria:
1. Clarity (5 points): Is the explanation easy to understand and clear?
2. Accuracy (5 points): Is the provided information technically accurate?""",
            model_response="""Python is a high-level programming language with clear syntax."""
        )
        try:
            result = judge.judge(test_input)
            print("\n=== Rubric Evaluation Results (Multiple Models) ===")
            print(result)
        except Exception as e:
            print(f"Error during rubric evaluation: {str(e)}")
            raise

    def test_response_comparison():
        """Test comparison between two responses"""
        test_input = JudgeInput(
            judge_type=JudgeType.RESPONSE_COMPARISON,
            model_response="""Python is a programming language.""",
            model_response_b="""Python is a high-level programming language with clear syntax,
            dynamic typing support, and object-oriented programming capabilities."""
        )
        result = judge.judge(test_input)
        print("\n=== Response Comparison Results (Multiple Models) ===")
        print(result)

    def test_gold_comparison():
        """Test comparison with gold standard response"""
        test_input = JudgeInput(
            judge_type=JudgeType.RUBRIC_RESPONSE_AND_GOLD,
            rubric="Explain the main features of Python.",
            gold_response="""Python's main features:
1. Easy to read and clear syntax
2. Dynamic typing support
3. Object-oriented programming support
4. Rich standard library""",
            model_response="""Python has a clear syntax and supports dynamic typing.
            It also enables object-oriented programming."""
        )
        result = judge.judge(test_input)
        print("\n=== Gold Standard Comparison Results (Multiple Models) ===")
        print(result)

    # Run all tests
    test_rubric_evaluation()
    test_response_comparison()
    test_gold_comparison()

if __name__ == "__main__":
    test_multi_judge()