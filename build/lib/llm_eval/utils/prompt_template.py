import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from llm_eval.utils.logging import get_logger
import logging

logger = get_logger(name="prompt_template", level=logging.INFO)

DEFAULT_FEW_SHOT_INSTRUCTION = "다음은 문제와 정답의 몇 가지 예시입니다.\n\n"
DEFAULT_FEW_SHOT_EXAMPLE_TEMPLATE = "{input}\n정답: {reference}\n\n"

# 대신 직접 정의
class JudgeType(Enum):
    RUBRIC_AND_RESPONSE = "rubric_and_response"
    RUBRIC_RESPONSE_AND_GOLD = "rubric_response_and_gold"
    RESPONSE_COMPARISON = "response_comparison"
    K2_EVAL = "k2_eval"

def format_few_shot_prompt_prefix(
    few_shot_samples: List[Dict[str, Any]],
    instruction: Optional[str] = DEFAULT_FEW_SHOT_INSTRUCTION,
    example_template: str = DEFAULT_FEW_SHOT_EXAMPLE_TEMPLATE,
) -> str:
    """
    Few-shot 예시들로부터 프롬프트 접두사(prefix)를 생성합니다.

    Args:
        few_shot_samples (List[Dict[str, Any]]):
            Few-shot 예시로 사용될 샘플 리스트.
            각 샘플은 example_template에 필요한 키(예: 'input', 'reference')를 가져야 합니다.
        instruction (Optional[str]):
            Few-shot 예시 앞에 추가될 전체 설명/지시문.
        example_template (str):
            각 few-shot 예시를 포맷팅할 템플릿.

    Returns:
        str: Few-shot 예시들로 구성된 프롬프트 접두사 문자열.
    """
    if not few_shot_samples:
        logger.debug("[FewShotFormatter] No few-shot samples provided, returning empty prefix.")
        return ""

    prefix_parts = []
    if instruction:
        prefix_parts.append(instruction)

    valid_examples_count = 0
    for i, sample in enumerate(few_shot_samples):
        try:
            # 'input'과 'reference' 키가 샘플에 있는지 확인하고 문자열로 변환
            sample_input = str(sample.get('input', ''))
            sample_reference = str(sample.get('reference', ''))

            if not sample_input:
                logger.warning(f"[FewShotFormatter] Skipping few-shot example {i+1} due to empty 'input'. Sample: {sample.get('id', sample)}")
                continue

            current_example_str = example_template.format(
                input=sample_input,
                reference=sample_reference
            )
            prefix_parts.append(current_example_str)
            valid_examples_count += 1
        except KeyError as e:
            logger.warning(f"[FewShotFormatter] Skipping few-shot example {i+1} due to missing key '{e}' for template. Sample keys: {list(sample.keys())}")
            continue
        except Exception as e:
            logger.error(f"[FewShotFormatter] Unexpected error formatting few-shot example {i+1}: {e}. Sample: {sample}", exc_info=True)
            continue

    if valid_examples_count == 0:
        logger.warning("[FewShotFormatter] No valid few-shot examples were formatted. Returning empty prefix even if instruction was provided.")
        return ""

    return "".join(prefix_parts)

def extract_final_answer(raw_output: str) -> str:
    """
    Extracts the final answer from a raw output that may contain unnecessary chain-of-thought (CoT) details,
    using the MULTILINGUAL_ANSWER_PATTERN.

    Steps:
      1) Pattern matching: Capture the text following markers such as "정답:", "답변:", "Answer:", etc.
      2) If no match is found, return the original raw_output.
      3) If a match is found, process the captured group ("content")—for example, split by newline and take the first part,
         or simply apply strip()—as appropriate.

    Returns:
        str: The extracted final answer (or the original raw_output if no match is found).
    """
    match = re.search(MULTILINGUAL_ANSWER_PATTERN, raw_output, flags=re.DOTALL)
    if match:
        # The "content" group captures the actual final answer part.
        content = match.group("content")
        # Final processing: here we simply strip the whitespace.
        return content.strip()
    else:
        # If the pattern is not found, return the original raw output.
        return raw_output

def default_cot_parser(raw_output: str) -> Tuple[str, str]:
    """
    Default chain-of-thought (CoT) parser.
    Uses the extract_final_answer function to extract the final answer from the raw output,
    and considers everything before the final answer as the chain-of-thought.

    Returns:
        Tuple[str, str]: A tuple (chain_of_thought, final_answer).
    """
    final_answer = extract_final_answer(raw_output)
    if final_answer:
        # Find the last occurrence of the final_answer within raw_output
        idx = raw_output.rfind(final_answer)
        if idx != -1:
            chain_of_thought = raw_output[:idx].strip()
        else:
            chain_of_thought = ""
    else:
        chain_of_thought = ""
    return chain_of_thought, final_answer


MULTILINGUAL_ANSWER_REGEXES = [
    r"Answer\s*:",
    r"Final\s*Answer\s*:",
    r"답변\s*:",
    r"정답\s*:",
    r"정답은\s*:",
    r"답\s*:",
    r"답은\s*:",
    r"답안\s*:",
    r"答案\s*[:：]",
    r"解答\s*[:：]",
    r"回答\s*[:：]",
    r"答\s*[:：]",
    r"Jawaban\s*:",
    r"Réponse\s*:",
    r"Resposta\s*:",
    r"Jibu\s*:",
    r"Idahun\s*:",
    r"Ìdáhùn\s*:",
    r"Idáhùn\s*:",
    r"Àmọ̀nà\s*:",
    r"Àdáhùn\s*:",
    r"Ànúgọ\s*:",
    r"Àṣàyàn\s*:",
]

MULTILINGUAL_ANSWER_PATTERN = (
    r"(?i)(" + "|".join(MULTILINGUAL_ANSWER_REGEXES) + r")\s*(?P<content>.+)"
)

# llm_eval/utils/prompt_template.py

JUDGE_PROMPTS = {
    JudgeType.RUBRIC_AND_RESPONSE: """You are an expert evaluator. Evaluate the following response based on the rubric.

Rubric:
{rubric}

Response to evaluate:
{response}

Provide detailed feedback based on the rubric, then end with your verdict in this EXACT format:
[RESULT] X where X is an integer between 1 and 5.""",

    JudgeType.RUBRIC_RESPONSE_AND_GOLD: """You are an expert evaluator. Compare the response with the reference answer.

Reference Answer:
{gold}

Response to evaluate:
{response}

IMPORTANT: Your response MUST be in this EXACT format:
[[true]] if the response is correct, or [[false]] if it is incorrect.""",

    JudgeType.RESPONSE_COMPARISON: """You are an expert evaluator. Compare the two responses and determine which one is better.

Response A:
{response}

Response B:
{response_b}

IMPORTANT: Your response MUST be in this EXACT format:
[[A]] if Response A is better, or [[B]] if Response B is better.

First provide your detailed comparison, then end with your verdict in the specified format.""",

    JudgeType.K2_EVAL: """
You are tasked with evaluating the responses generated by a Korean language model.
Please evaluate the response according to the following criteria:
1. Completeness: Does the response fully address the question?
2. Logical Consistency: Is the reasoning clear and logically sound?
3. Fluency: Is the language natural and coherent?
4. Cultural Appropriateness: Does the response adhere to Korean linguistic and cultural norms?

Provide your evaluation in 4-5 sentences, then assign a score from 1 to 5 in the following format:
Evaluation: [Your evaluation here]
Score: [Numeric score here]

Below is the question and the model's response. Please evaluate the response based on the criteria provided.

**Question:**
{input}

**Model's Response:**
{response}

[[End of conversation. Begin evaluation.]]"""
    }
