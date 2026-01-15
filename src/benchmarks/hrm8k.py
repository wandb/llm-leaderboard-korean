"""
HRM8K - HAE-RAE Math 8K

한국어-영어 이중 언어 수학 추론 벤치마크
- GSM8K: 기초 수학 (1,319개)
- KSM: 한국 수학 올림피아드/경시대회 (1,428개)
- MATH: 중급 수학 (2,885개)
- MMMLU: 수학 관련 선택형 (470개)
- OMNI_MATH: 고급 수학 (1,909개)

출처: https://huggingface.co/datasets/HAERAE-HUB/HRM8K
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/HRM8K_mini:YiSCSfQWVw5k5QUeyXgkiawMs3lZxiYghev2p7SYX7E",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
    },
    answer_format="identity",  # 수학 문제이므로 answer를 그대로 사용
    solver="generate",
    scorer="math_grader",
    system_message="다음 수학 문제를 풀어주세요. 풀이 과정과 함께 최종 답을 '\\boxed{정답}' 형식으로 명확하게 제시해주세요. 만약 문제가 객관식이라면 객관식의 정답번호를 최종 답으로 제시해주세요.\n최종 답 예시\n- 주관식인 경우: '\\boxed{7}'\n- 지문이 1, 2, 3, 4인 객관식인 경우: '\\boxed{2}'\n\n문제:",
)

