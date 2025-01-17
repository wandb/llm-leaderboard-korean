from llm_judge import LLMJudge, OpenAIClient, JudgeInput, JudgeType
import dotenv
import os

# .env 파일에서 환경 변수 로드
dotenv.load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY가 환경 변수에 설정되어 있지 않습니다.")

# OpenAI 클라이언트 및 judge 초기화
llm_client = OpenAIClient(api_key=api_key)
judge = LLMJudge(llm_client)

def test_rubric_evaluation():
    """루브릭 기반 단일 응답 평가 테스트"""
    test_input = JudgeInput(
        judge_type=JudgeType.RUBRIC_AND_RESPONSE,
        rubric="""평가 기준:
1. 명확성 (5점): 설명이 이해하기 쉽고 명확한가?
2. 정확성 (5점): 제공된 정보가 기술적으로 정확한가?""",
        model_response="""파이썬은 가독성이 높고 배우기 쉬운 프로그래밍 언어입니다."""
    )
    result = judge.judge(test_input)
    print("\n=== 루브릭 기반 평가 결과 ===")
    print(result)

def test_response_comparison():
    """두 응답 비교 평가 테스트"""
    test_input = JudgeInput(
        judge_type=JudgeType.RESPONSE_COMPARISON,
        model_response="""파이썬은 프로그래밍 언어입니다.""",
        model_response_b="""파이썬은 가독성이 높고 배우기 쉬운 고수준 프로그래밍 언어로,
동적 타이핑을 지원하며 객체 지향 프로그래밍이 가능합니다."""
    )
    result = judge.judge(test_input)
    print("\n=== 응답 비교 평가 결과 ===")
    print(result)

def test_gold_comparison():
    """정답과 비교 평가 테스트"""
    test_input = JudgeInput(
        judge_type=JudgeType.RUBRIC_RESPONSE_AND_GOLD,
        rubric="파이썬의 주요 특징을 설명하시오.",
        gold_response="""파이썬의 주요 특징:
1. 읽기 쉽고 명확한 문법
2. 동적 타이핑 지원
3. 객체 지향 프로그래밍 지원
4. 풍부한 표준 라이브러리""",
        model_response="""파이썬은 읽기 쉽고 명확한 문법을 가진 언어입니다.
동적 타이핑을 지원하며 객체 지향 프로그래밍이 가능합니다."""
    )
    result = judge.judge(test_input)
    print("\n=== 정답 비교 평가 결과 ===")
    print(result)

if __name__ == "__main__":
    # 세 가지 평가 방식 모두 테스트
    test_rubric_evaluation()
    test_response_comparison()
    test_gold_comparison()