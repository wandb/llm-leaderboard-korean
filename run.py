#!/usr/bin/env python3
"""
실험 실행 스크립트

Usage:
    python run.py --config configs/gpt4_experiment.yaml
    python run.py --config configs/claude_experiment.yaml
"""

import argparse
import yaml
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from llm_eval.evaluator import Evaluator
from llm_eval.utils.logging import get_logger

logger = get_logger(name="experiment_runner", level=logging.INFO)

# 사용 가능한 데이터셋 목록
# 사용 가능한 데이터셋 목록과 각각에 적합한 평가 방법
DATASET_EVALUATION_MAP = {
    "haerae_bench": "string_match",      # 객관식 문제 - 정확한 문자열 매칭
    "kmmlu": "string_match",             # 객관식 문제 - 정확한 문자열 매칭
    "click": "string_match",             # 일반적으로 문자열 매칭
    "hrm8k": "math_eval",                # 수학 문제 - 수학적 동등성 평가
    "k2_eval": "llm_judge",              # 생성 태스크 - LLM 판단 필요
    "KUDGE": "llm_judge",                # 판단/평가 태스크 - LLM 판단 필요
    "benchhub": "string_match",          # 일반적으로 문자열 매칭
    "hrc": "string_match",               # 일반적으로 문자열 매칭
    "kbl": "string_match",               # 일반적으로 문자열 매칭
    "kormedmcqa": "string_match",        # 의료 객관식 - 정확한 문자열 매칭
    "aime2025": "math_eval",             # 수학 문제 - 수학적 동등성 평가
    "generic_file": "string_match"       # 기본값은 문자열 매칭
}

AVAILABLE_DATASETS = list(DATASET_EVALUATION_MAP.keys())

def load_config(config_path: str) -> Dict[str, Any]:
    """YAML 설정 파일을 로드합니다. base_config.yaml을 먼저 로드하고 모델별 config로 오버라이드합니다."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    # 1. base_config.yaml 로드
    base_config_path = "configs/base_config.yaml"
    base_config = {}
    
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        logger.info(f"기본 설정 파일 로드 완료: {base_config_path}")
    else:
        logger.warning(f"기본 설정 파일을 찾을 수 없습니다: {base_config_path}")
    
    # 2. 모델별 config 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    # 3. 설정 병합 (모델별 config가 base_config를 오버라이드)
    config = merge_configs(base_config, model_config)
    
    logger.info(f"모델별 설정 파일 로드 및 병합 완료: {config_path}")
    return config

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """두 설정을 재귀적으로 병합합니다. override_config가 base_config를 덮어씁니다."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # 딕셔너리인 경우 재귀적으로 병합
            merged[key] = merge_configs(merged[key], value)
        else:
            # 그 외의 경우 덮어쓰기
            merged[key] = value
    
    return merged

def get_enabled_datasets(config: Dict[str, Any]) -> List[str]:
    """설정에서 활성화된 데이터셋 목록을 반환합니다."""
    datasets_config = config.get('datasets', {})
    enabled_datasets = []
    
    for dataset_name in AVAILABLE_DATASETS:
        if datasets_config.get(dataset_name, False):
            enabled_datasets.append(dataset_name)
    
    logger.info(f"활성화된 데이터셋: {enabled_datasets}")
    return enabled_datasets

def get_dataset_evaluation_method(dataset_name: str, config: Dict[str, Any]) -> str:
    """데이터셋에 적합한 평가 방법을 반환합니다."""
    # 1. 설정에서 명시적으로 지정된 평가 방법 확인
    explicit_method = config.get('evaluation', {}).get('method')
    if explicit_method:
        logger.info(f"{dataset_name}: 설정에서 명시된 평가 방법 사용 - {explicit_method}")
        return explicit_method
    
    # 2. 데이터셋별 특별 설정에서 평가 방법 확인
    dataset_specific = config.get('dataset_specific', {}).get(dataset_name, {})
    dataset_method = dataset_specific.get('evaluation_method')
    if dataset_method:
        logger.info(f"{dataset_name}: 데이터셋별 특별 설정 평가 방법 사용 - {dataset_method}")
        return dataset_method
    
    # 3. 데이터셋에 최적화된 기본 평가 방법 사용
    default_method = DATASET_EVALUATION_MAP.get(dataset_name, "string_match")
    logger.info(f"{dataset_name}: 데이터셋 최적화 평가 방법 사용 - {default_method}")
    return default_method

def create_output_directory(config: Dict[str, Any]) -> str:
    """결과 저장용 디렉토리를 생성합니다."""
    model_name = config.get('model', {}).get('name', 'unknown_model')
    model_path = config.get('model', {}).get('params', {}).get('model_name_or_path', 'unknown')
    
    # 모델 경로에서 마지막 부분만 추출 (예: "openai/gpt-4" -> "gpt-4")
    if '/' in model_path:
        model_path = model_path.split('/')[-1]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/{model_name}_{model_path}_{timestamp}"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"결과 저장 디렉토리 생성: {output_dir}")
    
    return output_dir

def run_single_dataset_experiment(
    evaluator: Evaluator, 
    dataset_name: str, 
    config: Dict[str, Any], 
    output_dir: str
) -> Dict[str, Any]:
    """단일 데이터셋에 대한 실험을 실행합니다."""
    logger.info(f"=== {dataset_name} 데이터셋 실험 시작 ===")
    
    try:
        # 데이터셋별 특별 설정 가져오기
        dataset_specific_config = config.get('dataset_specific', {}).get(dataset_name, {})
        
        # 기본 데이터셋 설정과 특별 설정 병합
        dataset_params = {**config.get('dataset', {}).get('params', {}), **dataset_specific_config}
        
        # 데이터셋에 적합한 평가 방법 결정
        evaluation_method = get_dataset_evaluation_method(dataset_name, config)
        
        # 실험 실행
        result = evaluator.run(
            model=config.get('model', {}).get('name'),
            judge_model=config.get('judge_model', {}).get('name'),
            reward_model=config.get('reward_model', {}).get('name'),
            dataset=dataset_name,
            subset=dataset_specific_config.get('subset'),
            split=config.get('dataset', {}).get('split', 'test'),
            scaling_method=config.get('scaling', {}).get('name'),
            evaluation_method=evaluation_method,  # 데이터셋별 최적화된 평가 방법 사용
            dataset_params=dataset_params,
            model_params=config.get('model', {}).get('params', {}),
            judge_params=config.get('judge_model', {}).get('params', {}),
            reward_params=config.get('reward_model', {}).get('params', {}),
            scaling_params=config.get('scaling', {}).get('params', {}),
            evaluator_params=config.get('evaluation', {}).get('params', {}),
            language_penalize=config.get('language_penalize', True),
            target_lang=config.get('target_lang', 'ko'),
            custom_cot_parser=config.get('custom_cot_parser'),
            num_few_shot=config.get('few_shot', {}).get('num', 0),
            few_shot_split=config.get('few_shot', {}).get('split'),
            few_shot_instruction=config.get('few_shot', {}).get('instruction'),
            few_shot_example_template=config.get('few_shot', {}).get('example_template'),
        )
        
        # 결과 저장
        result_dict = result.to_dict()
        result_file = os.path.join(output_dir, f"{dataset_name}_results.json")
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"{dataset_name} 실험 완료. 결과 저장: {result_file}")
        logger.info(f"{dataset_name} 메트릭: {result.metrics}")
        
        return {
            'dataset': dataset_name,
            'status': 'success',
            'metrics': result.metrics,
            'result_file': result_file
        }
        
    except Exception as e:
        error_msg = f"{dataset_name} 실험 실패: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            'dataset': dataset_name,
            'status': 'failed',
            'error': str(e),
            'result_file': None
        }

def run_experiment(config_path: str) -> None:
    """전체 실험을 실행합니다."""
    logger.info("=" * 50)
    logger.info("실험 실행 시작")
    logger.info("=" * 50)
    
    # 설정 로드
    config = load_config(config_path)
    
    # 활성화된 데이터셋 가져오기
    enabled_datasets = get_enabled_datasets(config)
    
    if not enabled_datasets:
        logger.warning("활성화된 데이터셋이 없습니다. 실험을 종료합니다.")
        return
    
    # 결과 저장 디렉토리 생성
    output_dir = create_output_directory(config)
    
    # 설정 파일 백업
    config_backup_path = os.path.join(output_dir, "experiment_config.yaml")
    with open(config_backup_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # Evaluator 초기화
    evaluator = Evaluator(
        default_model_backend=config.get('model', {}).get('name', 'huggingface'),
        default_judge_backend=config.get('judge_model', {}).get('name'),
        default_reward_backend=config.get('reward_model', {}).get('name'),
        default_scaling_method=config.get('scaling', {}).get('name'),
        default_evaluation_method=config.get('evaluation', {}).get('method', 'string_match'),
        default_split=config.get('dataset', {}).get('split', 'test'),
        default_cot_parser=config.get('custom_cot_parser'),
        default_num_few_shot=config.get('few_shot', {}).get('num', 0),
        default_few_shot_split=config.get('few_shot', {}).get('split'),
        default_few_shot_instruction=config.get('few_shot', {}).get('instruction'),
        default_few_shot_example_template=config.get('few_shot', {}).get('example_template'),
    )
    
    # 실험 결과 수집
    experiment_results = []
    total_datasets = len(enabled_datasets)
    
    # 각 데이터셋에 대해 실험 실행
    for i, dataset_name in enumerate(enabled_datasets, 1):
        logger.info(f"진행상황: {i}/{total_datasets}")
        result = run_single_dataset_experiment(evaluator, dataset_name, config, output_dir)
        experiment_results.append(result)
    
    # 전체 실험 결과 저장
    summary_file = os.path.join(output_dir, "experiment_summary.json")
    summary_data = {
        'config_path': config_path,
        'timestamp': datetime.now().isoformat(),
        'total_datasets': total_datasets,
        'successful_experiments': len([r for r in experiment_results if r['status'] == 'success']),
        'failed_experiments': len([r for r in experiment_results if r['status'] == 'failed']),
        'results': experiment_results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    # 실험 완료 로그
    logger.info("=" * 50)
    logger.info("실험 완료!")
    logger.info(f"총 데이터셋: {total_datasets}")
    logger.info(f"성공: {summary_data['successful_experiments']}")
    logger.info(f"실패: {summary_data['failed_experiments']}")
    logger.info(f"결과 저장 위치: {output_dir}")
    logger.info("=" * 50)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="LLM 평가 실험 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="실험 설정 YAML 파일 경로 (예: configs/gpt4_experiment.yaml)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="로그 레벨 설정"
    )
    
    args = parser.parse_args()
    
    # 로그 레벨 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        run_experiment(args.config)
    except Exception as e:
        logger.critical(f"실험 실행 중 치명적 오류 발생: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
