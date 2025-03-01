import re
from typing import List, Dict, Any, Optional, Callable, Tuple, Union

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
    and returns an empty string for the chain-of-thought portion.

    Returns:
        Tuple[str, str]: A tuple (chain_of_thought, final_answer).
    """
    final_answer = extract_final_answer(raw_output)
    chain_of_thought = ""  # Optionally, add logic here to extract the chain-of-thought portion from raw_output.
    return chain_of_thought, final_answer

    
MULTILINGUAL_ANSWER_REGEXES = [
    r"Answer\s*:",
    r"Final\s*Answer\s*:",
    r"답변\s*:",
    r"정답\s*:",
    r"답\s*:",
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

JUDGE_PROMPTS = {
    "RUBRIC_AND_RESPONSE": """You are an expert evaluator. Your task is to evaluate the given response based on the rubric and provide a score.

IMPORTANT: You must format your response exactly like this example:
Based on the rubric, this response deserves [[score: 7]].

Rubric:
{rubric}

Response to evaluate:
{response}

Provide your evaluation with the score in the specified format:""",

    "RUBRIC_RESPONSE_AND_GOLD": """You are an expert evaluator. Please evaluate if the following response matches the gold standard answer.
Compare step by step and provide your verdict as [[true]] if correct or [[false]] step: [X] if incorrect.

Rubric:
{rubric}

Gold Response:
{gold_response}

Model Response:
{response}

Provide your evaluation in the specified format:""",

    "RESPONSE_COMPARISON": """You are an expert evaluator. Your task is to compare two responses and choose the better one.

IMPORTANT: You must format your verdict exactly like this:
- Use [[A]] to choose the first response
- Use [[B]] to choose the second response
- Use [[C]] if they are equally good

Response A:
{response_a}

Response B:
{response_b}

Provide your verdict in the specified format:"""
}
