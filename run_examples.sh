#!/bin/bash

# 실험 실행 예시 스크립트

echo "=== LLM 평가 실험 실행 예시 ==="
echo ""
echo "🔧 설정 구조:"
echo "- base_config.yaml: 공통 기본 설정"
echo "- 모델별 config: base_config를 오버라이드하는 모델 특화 설정"
echo ""
echo "⚡ 평가 방법 자동 선택:"
echo "- 객관식 (haerae_bench, kmmlu, kormedmcqa): string_match"
echo "- 수학 문제 (hrm8k, aime2025): math_eval"
echo "- 생성 태스크 (k2_eval, KUDGE): llm_judge"
echo ""

echo "1. 빠른 테스트 (GPT-2로 단일 데이터셋, string_match 자동 선택)"
echo "python run.py --config configs/quick_test.yaml"
echo ""

echo "2. GPT-4 실험 (다양한 데이터셋, 각각 최적 평가 방법 자동 선택)"
echo "python run.py --config configs/gpt4_experiment.yaml"
echo ""

echo "3. Claude 실험 (더 많은 데이터셋, 2-shot 학습)"
echo "python run.py --config configs/claude_experiment.yaml"
echo ""

echo "4. 평가 방법 데모 (자동 선택과 수동 오버라이드 예시)"
echo "python run.py --config configs/evaluation_method_demo.yaml"
echo ""

echo "5. 디버그 모드로 실행 (평가 방법 선택 과정 확인)"
echo "python run.py --config configs/quick_test.yaml --log-level DEBUG"
echo ""

echo "📊 결과 저장:"
echo "- results/ 디렉토리에 모델명_타임스탬프 폴더 생성"
echo "- 각 데이터셋별 결과 JSON 파일 (사용된 평가 방법 포함)"
echo "- experiment_summary.json (전체 실험 요약)"
echo "- experiment_config.yaml (사용된 설정 백업)"
echo ""

echo "💡 평가 방법 커스터마이징:"
echo "- 전역 설정: evaluation.method 필드 설정"
echo "- 데이터셋별: dataset_specific.[dataset].evaluation_method 설정"

