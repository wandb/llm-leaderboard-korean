JUDGE_PROMPTS = {
    "RUBRIC_AND_RESPONSE": """
Please evaluate the following response according to the provided rubric.
Rate on a scale of 0-10 and provide your verdict in this format: [[score: X]]

Rubric:
{rubric}

Response to evaluate:
{response}
""",

    "RUBRIC_RESPONSE_AND_GOLD": """
Please evaluate the following response against the gold standard answer.
Compare step by step and provide your verdict as [[true]] if correct or [[false]] step: [X] if incorrect.

Question:
{rubric}

Gold Answer:
{gold_response}

Model Response:
{response}
""",

    "RESPONSE_COMPARISON": """
Please compare the following two responses and choose the better one.
Provide your verdict as [[A]] for first response, [[B]] for second response, or [[C]] for tie.

Response A:
{response_a}

Response B:
{response_b}
"""
}
