"""
KoAIME2025 - 한국어 AIME 2025 수학 벤치마크

inspect_evals.aime2025를 상속하여 dataset만 override합니다.
"""

CONFIG = {
    "base": "inspect_evals.aime2025.aime2025",
    "data_type": "jsonl",
    "data_source": "ko_aime2025.jsonl",
}

