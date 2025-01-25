from .llm_judge import MultiLLMJudge, JudgeInput, JudgeType
from ..models.huggingface import HuggingFaceModel
from ..models.openai_backend import OpenAIModel
import dotenv
import os
import logging
import torch

dotenv.load_dotenv()

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")

    models_config = [
        {
            "name": "openai",
            "type": "model",
            "api_key": api_key,  # Add API key here
            "model_name": "gpt-4o"
        },
        {
            "name": "huggingface", 
            "type": "model",
            "model_name_or_path": "google/gemma-2-2b-it",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.95
        }
    ]

    logger = logging.getLogger("test_judge")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    judge = MultiLLMJudge(
        models_config=models_config,
        aggregation_strategy="all",
        logger=logger
    )

    rubric_input = JudgeInput(
        judge_type=JudgeType.RUBRIC_AND_RESPONSE,
        model_response="Python is a high-level programming language with clear syntax."
    )
    
    gold_input = JudgeInput(
        judge_type=JudgeType.RUBRIC_RESPONSE_AND_GOLD,
        model_response="Python uses indentation for code blocks.",
        gold_response="Python uses indentation to define code blocks and structure."
    )
    
    comparison_input = JudgeInput(
        judge_type=JudgeType.RESPONSE_COMPARISON,
        model_response="Python is a versatile programming language.",
        model_response_b="Python is a popular programming language used in many fields."
    )

    print("\n=== 1. Rubric Evaluation Results ===")
    result1 = judge.judge(rubric_input)
    print(result1)

    print("\n=== 2. Gold Answer Comparison Results ===")
    result2 = judge.judge(gold_input)
    print(result2)

    print("\n=== 3. Response Comparison Results ===")
    result3 = judge.judge(comparison_input)
    print(result3)



if __name__ == "__main__":
    main()