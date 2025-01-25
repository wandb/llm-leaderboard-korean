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
