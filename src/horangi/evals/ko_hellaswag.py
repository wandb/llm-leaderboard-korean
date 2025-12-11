"""
KoHellaSwag - 한국어 HellaSwag 벤치마크

inspect_evals.hellaswag를 상속하여 dataset만 override합니다.
"""

CONFIG = {
    "base": "inspect_evals.hellaswag.hellaswag",
    "data_type": "weave",
    "data_source": "weave:///wandb-korea/evaluation-job/object/KoHellaSwag:PY229AMRxLFoCLqKsaEguY4jVCuvyoMnQ5wJ1wkrXfU",
    "split": "train",
}

