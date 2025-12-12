"""
한국어 LLM 평가 벤치마크 정의

각 벤치마크는 개별 파일로 관리됩니다.
새 벤치마크를 추가하려면:
1. 이 폴더에 새 파일 생성 (예: ko_new.py)
2. CONFIG 딕셔너리 정의
3. 이 파일의 BENCHMARKS에 추가
4. eval_tasks.py에 @task 함수 추가
"""

from horangi.evals.ko_hellaswag import CONFIG as ko_hellaswag
from horangi.evals.ko_aime2025 import CONFIG as ko_aime2025
from horangi.evals.ifeval_ko import CONFIG as ifeval_ko
from horangi.evals.ko_balt_700 import CONFIG as ko_balt_700
from horangi.evals.haerae_bench_v1_rc import CONFIG as haerae_bench_v1_rc
from horangi.evals.haerae_bench_v1_wo_rc import CONFIG as haerae_bench_v1_wo_rc
from horangi.evals.kmmlu import CONFIG as kmmlu
from horangi.evals.kmmlu_pro import CONFIG as kmmlu_pro
from horangi.evals.squad_kor_v1 import CONFIG as squad_kor_v1
from horangi.evals.ko_truthful_qa import CONFIG as ko_truthful_qa
from horangi.evals.ko_moral import CONFIG as ko_moral
from horangi.evals.ko_arc_agi import CONFIG as ko_arc_agi
from horangi.evals.ko_gsm8k import CONFIG as ko_gsm8k
from horangi.evals.korean_hate_speech import CONFIG as korean_hate_speech
from horangi.evals.kobbq import CONFIG as kobbq
from horangi.evals.ko_hle import CONFIG as ko_hle
# HalluLens 벤치마크
from horangi.evals.ko_hallulens_wikiqa import CONFIG as ko_hallulens_wikiqa
from horangi.evals.ko_hallulens_longwiki import CONFIG as ko_hallulens_longwiki
from horangi.evals.ko_hallulens_generated import CONFIG as ko_hallulens_generated
from horangi.evals.ko_hallulens_mixed import CONFIG as ko_hallulens_mixed
from horangi.evals.ko_hallulens_nonexistent import CONFIG as ko_hallulens_nonexistent
# BFCL 벤치마크
from horangi.evals.bfcl_extended import CONFIG as bfcl_extended
from horangi.evals.bfcl_text import CONFIG as bfcl_text
# MT-Bench
from horangi.evals.mtbench_ko import CONFIG as mtbench_ko
# SWE-bench
from horangi.evals.swebench_verified_official_80 import CONFIG as swebench_verified_official_80

# 모든 벤치마크 설정
BENCHMARKS: dict = {
    "ko_hellaswag": ko_hellaswag,
    "ko_aime2025": ko_aime2025,
    "ifeval_ko": ifeval_ko,
    "ko_balt_700": ko_balt_700,
    "haerae_bench_v1_rc": haerae_bench_v1_rc,
    "haerae_bench_v1_wo_rc": haerae_bench_v1_wo_rc,
    "kmmlu": kmmlu,
    "kmmlu_pro": kmmlu_pro,
    "squad_kor_v1": squad_kor_v1,
    "ko_truthful_qa": ko_truthful_qa,
    "ko_moral": ko_moral,
    "ko_arc_agi": ko_arc_agi,
    "ko_gsm8k": ko_gsm8k,
    "korean_hate_speech": korean_hate_speech,
    "kobbq": kobbq,
    "ko_hle": ko_hle,
    # HalluLens 벤치마크
    "ko_hallulens_wikiqa": ko_hallulens_wikiqa,
    "ko_hallulens_longwiki": ko_hallulens_longwiki,
    "ko_hallulens_generated": ko_hallulens_generated,
    "ko_hallulens_mixed": ko_hallulens_mixed,
    "ko_hallulens_nonexistent": ko_hallulens_nonexistent,  # Generated + Mixed 통합
    # BFCL 벤치마크
    "bfcl_extended": bfcl_extended,  # Native tool calling (OpenAI, Claude 등)
    "bfcl_text": bfcl_text,          # Text-based (EXAONE, 오픈소스 등)
    # MT-Bench
    "mtbench_ko": mtbench_ko,        # 멀티턴 대화 평가 (LLM Judge)
    # SWE-bench
    "swebench_verified_official_80": swebench_verified_official_80,  # SWE-bench Verified 80
}

def get_benchmark_config(name: str) -> dict:
    """벤치마크 설정 가져오기"""
    if name not in BENCHMARKS:
        available = ", ".join(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    return BENCHMARKS[name]


def list_benchmarks() -> list[str]:
    """사용 가능한 벤치마크 목록"""
    return list(BENCHMARKS.keys())


__all__ = ["BENCHMARKS", "get_benchmark_config", "list_benchmarks"]
