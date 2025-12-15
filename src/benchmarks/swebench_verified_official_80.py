"""
SWE-bench Verified (Official 80)

이 벤치마크는 Docker 하네스 대신 "채점 서버"를 호출해서 채점합니다.

데이터:
- Weave Dataset/Object를 직접 로드합니다.
- 추가 필드들(repo, version, test_patch, FAIL_TO_PASS, code 등)은 자동으로 metadata에 포함됨
"""

CONFIG = {
    "data_type": "weave",
    "data_source": "weave:///wandb-korea/evaluation-job/object/swebench_verified_official_80:uXFA9NSgw4xjeZIH6GpEBtH5FUiYGJF4f6jpgKqYnWw",
    "field_mapping": {
        "id": "instance_id",
        "input": "problem_statement",
        # target은 서버 채점으로 판단하므로 사용하지 않음
        # repo, version, test_patch, FAIL_TO_PASS, code 등은 자동으로 metadata에 포함
    },
    "solver": "swebench_patch_solver",
    "scorer": "swebench_server_scorer",
    # system_message는 solver에서 처리하므로 여기서는 생략
    "metadata": {
        "benchmark_type": "swebench",
        "split": "verified_test",
        "subset": "official_80",
    },
}

