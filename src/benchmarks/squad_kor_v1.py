"""
Squad-Kor-V1 - 한국어 Squad 벤치마크

inspect_evals.squad를 상속하여 dataset만 override합니다.
"""

CONFIG = {
    "base": "inspect_evals.squad.squad",
    "data_type": "weave",
    "data_source": "weave:///wandb-korea/evaluation-job/object/squad_kor_v1:2OPwXAfZ0y4zgqPHWXoFl6BAqf7OkkDqS0jaAB9kWOI",
}

