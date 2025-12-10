"""
한국어 벤치마크 모듈

각 벤치마크는 Inspect AI Task로 구현되어 있습니다.
"""

from horangi.benchmarks.korean_qa import korean_qa
from horangi.benchmarks.korean_reasoning import korean_reasoning
from horangi.benchmarks.korean_knowledge import korean_knowledge
from horangi.benchmarks.korean_commonsense import korean_commonsense

__all__ = [
    "korean_qa",
    "korean_reasoning",
    "korean_knowledge",
    "korean_commonsense",
]

