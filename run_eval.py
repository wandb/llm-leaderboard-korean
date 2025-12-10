#!/usr/bin/env python3
"""
한국어 벤치마크 평가 실행 스크립트

이 스크립트는 inspect-wandb를 사용하여 한국어 벤치마크 평가를 실행하고
결과를 WandB/Weave에 자동으로 기록합니다.

사용법:
    # 모든 벤치마크 실행
    python run_eval.py --model openai/gpt-4o

    # 특정 벤치마크만 실행
    python run_eval.py --model openai/gpt-4o --benchmark korean_qa

    # WandB 프로젝트 지정
    python run_eval.py --model anthropic/claude-sonnet-4-0 --wandb-project my-korean-eval
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def setup_wandb(project: str, entity: str | None = None):
    """WandB 설정 초기화"""
    os.environ.setdefault("WANDB_PROJECT", project)
    if entity:
        os.environ["WANDB_ENTITY"] = entity

    # inspect-wandb가 자동으로 wandb/weave 로깅을 처리합니다
    print(f"WandB 프로젝트: {project}")
    if entity:
        print(f"WandB 엔티티: {entity}")


def run_benchmark(
    benchmark: str,
    model: str,
    use_cot: bool = False,
    limit: int | None = None,
):
    """
    벤치마크 실행

    Args:
        benchmark: 벤치마크 이름 (korean_qa, korean_reasoning, korean_knowledge, korean_commonsense)
        model: 평가할 모델 (예: openai/gpt-4o, anthropic/claude-sonnet-4-0)
        use_cot: Chain-of-thought 사용 여부
        limit: 평가할 샘플 수 제한 (None이면 전체)
    """
    from inspect_ai import eval

    # 벤치마크 Task 임포트
    if benchmark == "korean_qa":
        from horangi.benchmarks import korean_qa
        task = korean_qa(use_cot=use_cot)
    elif benchmark == "korean_reasoning":
        from horangi.benchmarks import korean_reasoning
        task = korean_reasoning(use_cot=use_cot)
    elif benchmark == "korean_knowledge":
        from horangi.benchmarks import korean_knowledge
        task = korean_knowledge()
    elif benchmark == "korean_commonsense":
        from horangi.benchmarks import korean_commonsense
        task = korean_commonsense(use_cot=use_cot)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    print(f"\n{'='*60}")
    print(f"벤치마크: {benchmark}")
    print(f"모델: {model}")
    print(f"Chain-of-Thought: {'활성화' if use_cot else '비활성화'}")
    print(f"{'='*60}\n")

    # 평가 실행
    eval_args = {
        "model": model,
    }
    
    if limit:
        eval_args["limit"] = limit

    results = eval(task, **eval_args)
    
    return results


def run_all_benchmarks(
    model: str,
    use_cot: bool = False,
    limit: int | None = None,
):
    """모든 벤치마크 실행"""
    benchmarks = [
        "korean_qa",
        "korean_reasoning",
        "korean_knowledge",
        "korean_commonsense",
    ]

    all_results = {}
    for benchmark in benchmarks:
        print(f"\n[{benchmarks.index(benchmark) + 1}/{len(benchmarks)}] {benchmark} 평가 중...")
        try:
            results = run_benchmark(benchmark, model, use_cot, limit)
            all_results[benchmark] = results
        except Exception as e:
            print(f"⚠️ {benchmark} 평가 중 오류 발생: {e}")
            all_results[benchmark] = None

    # 결과 요약
    print("\n" + "=" * 60)
    print("평가 결과 요약")
    print("=" * 60)
    
    for benchmark, results in all_results.items():
        if results:
            print(f"  {benchmark}: 완료")
        else:
            print(f"  {benchmark}: 실패")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="한국어 LLM 벤치마크 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # GPT-4o로 한국어 QA 평가
  python run_eval.py --model openai/gpt-4o --benchmark korean_qa

  # Claude로 모든 벤치마크 평가
  python run_eval.py --model anthropic/claude-sonnet-4-0

  # Chain-of-Thought 활성화
  python run_eval.py --model openai/gpt-4o --benchmark korean_reasoning --cot

  # 샘플 수 제한하여 테스트
  python run_eval.py --model openai/gpt-4o --limit 5
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="평가할 모델 (예: openai/gpt-4o, anthropic/claude-sonnet-4-0)",
    )
    parser.add_argument(
        "--benchmark", "-b",
        choices=["korean_qa", "korean_reasoning", "korean_knowledge", "korean_commonsense", "all"],
        default="all",
        help="실행할 벤치마크 (기본: all)",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Chain-of-Thought 프롬프팅 활성화",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="평가할 샘플 수 제한 (테스트용)",
    )
    parser.add_argument(
        "--wandb-project", "-p",
        default="korean-llm-benchmark",
        help="WandB 프로젝트 이름 (기본: korean-llm-benchmark)",
    )
    parser.add_argument(
        "--wandb-entity", "-e",
        default=None,
        help="WandB 엔티티 (팀 또는 사용자 이름)",
    )

    args = parser.parse_args()

    # WandB 설정
    setup_wandb(args.wandb_project, args.wandb_entity)

    # 벤치마크 실행
    if args.benchmark == "all":
        results = run_all_benchmarks(args.model, args.cot, args.limit)
    else:
        results = run_benchmark(args.benchmark, args.model, args.cot, args.limit)

    print("\n✅ 평가 완료!")
    print("📊 WandB 대시보드에서 결과를 확인하세요.")


if __name__ == "__main__":
    main()

