"""
데이터 변환 함수들
"""

from typing import Any, Callable

def identity(answer_value: Any) -> Any:
    """변환 없이 그대로 반환"""
    return answer_value


def to_string(answer_value: Any) -> str:
    """문자열로 변환"""
    return str(answer_value)

def index_0(answer_value: Any) -> str:
    return chr(ord('A') + int(answer_value))

def index_1(answer_value: Any) -> str:
    return chr(ord('A') + int(answer_value) - 1)

def letter(answer_value: Any) -> str:
    return str(answer_value).upper()

def boolean(answer_value: Any) -> str:
    return "A" if eval(answer_value) else "B"

def text(answer_value: Any, choices: Any) -> str:
    if choices is None:
        raise ValueError("choices required for 'text' answer_format")
    try:
        idx = choices.index(answer_value)
        return chr(ord('A') + idx)
    except ValueError:
        raise ValueError(f"Answer '{answer_value}' not found in choices: {choices}")

def unknown(answer_format: Any) -> str:
    raise ValueError(f"Unknown answer_format: {answer_format}")


ANSWER_FORMAT: dict[str, Callable] = {
    "index_0": index_0,
    "index_1": index_1,
    "letter": letter,
    "boolean": boolean,
    "text": text,
    "identity": identity,
    "to_string": to_string,
    "unknown": unknown,
}

