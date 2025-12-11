"""
IFEval-Ko - 한국어 IFEval 벤치마크

inspect_evals.ifeval를 상속하여 dataset만 override합니다.
"""

CONFIG = {
    "base": "inspect_evals.ifeval.ifeval",
    "data_type": "weave",
    "data_source": "weave:///wandb-korea/evaluation-job/object/IFEval-Ko:SGtm8r2dBuXUnkS402O7vYwzrBGrzYCfrLu7WS2u2No",
}

