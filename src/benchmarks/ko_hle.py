"""
KoHLE Standalone - 한국어 Humanity's Last Exam 벤치마크 (독립 버전)

inspect_evals.hle를 상속하지 않고 독립적으로 구현.
커스텀 hle_grader 사용 (judge 모델 별도 지정 가능).
"""

CONFIG = {
    # base 없음 - 독립 벤치마크
    "data_type": "weave",
    "data_source": "weave:///wandb-korea/evaluation-job/object/KoHLE:dgwAXckQdKIr3DuibE9Bvo5D4Sa2QIKK43ztZJ37F6I",
    "field_mapping": {
        "id": "id",
        "input": "question",
        "target": "answer",
    },
    "answer_format": "identity",  # 답변 그대로 사용
    "solver": "generate",
    "scorer": "hle_grader",
    "system_message": """답변은 다음 형식으로 작성해 주십시오.
설명: {답변 선택에 대한 설명}
답변: {선택한 답변}
확신도: {답변에 대한 확신도 (0%~100%)}""",
}

# """
# KoHLE - 한국어 Humanity's Last Exam 벤치마크

# inspect_evals.hle를 상속하여 dataset만 override합니다.
# LLM grader로 채점합니다.
# """

# CONFIG = {
#     "base": "inspect_evals.hle.hle",  # dataset override
#     "data_type": "weave",
#     "data_source": "weave:///wandb-korea/evaluation-job/object/KoHLE:dgwAXckQdKIr3DuibE9Bvo5D4Sa2QIKK43ztZJ37F6I",
#     # 원본 HLE record_to_sample에서 필요하지만 KoHLE에 없는 필드
#     "default_fields": {
#         "image": None,
#         "author_name": "",
#         "raw_subject": "",
#     },
# }

