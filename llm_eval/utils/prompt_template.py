import re

def extract_final_answer(raw_output: str) -> str:
    """
    Chain-of-thought(CoT) 등 불필요한 서술이 섞인 raw_output에서
    MULTILINGUAL_ANSWER_PATTERN을 사용해 실제 최종 답만 추출.

    1) 패턴 매칭: "정답:", "답변:", "Answer:", etc. 뒤에 나오는 텍스트를 그룹(content)으로 캡처
    2) 찾지 못하면 raw_output 그대로 반환
    3) 찾았다면, match.group("content")를 줄바꿈 등으로 split해 첫 부분만 사용하거나,
       strip()만 적용하는 식으로 가공 가능.

    Returns:
        str: 추출된 최종 답변(또는 raw_output 전체)
    """
    match = re.search(MULTILINGUAL_ANSWER_PATTERN, raw_output, flags=re.DOTALL)
    if match:
        # group("content")가 실제로 캡처된 "최종 답안 부분"
        content = match.group("content")
        # 최종적으로, 첫 줄만 떼거나 전체를 살리거나 -> 상황에 맞게 조정
        # 여기서는 단순히 strip()만 해준다.
        return content.strip()
    else:
        # 패턴이 없는 경우 원본 반환
        return raw_output
    
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
