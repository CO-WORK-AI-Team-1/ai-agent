import time

from llm.client import generate_response


TEST_CASES = [
    {
        "name": "일반 질의",
        "prompt": """
월드비전 AI 서비스가 필요한 이유를
3가지로 간단하게 설명해주세요.
""",
    },
    {
        "name": "요약",
        "prompt": """
다음 내용을 핵심 내용만 3줄로 요약해주세요.

월드비전 AI 서비스는 내부 문서 검색,
AI 기반 문서 작성, 회의 업무 지원 기능을 제공한다.
이를 통해 분산된 사내 지식을 쉽게 검색하고
반복적인 업무 시간을 줄이는 것을 목표로 한다.
""",
    },
    {
        "name": "구조화 출력",
        "prompt": """
다음 정보를 반드시 JSON 형식으로만 출력해주세요.

부서명: IT팀
요청사항: 내부 문서 검색 기능 개발
우선순위: 높음
""",
    },
    {
        "name": "Hallucination 방지",
        "prompt": """
제공된 정보만 사용해서 답변해주세요.
확인할 수 없는 내용은 추측하지 말고
"제공된 정보에서 확인할 수 없습니다."라고 답해주세요.

제공된 정보:
월드비전 AI 서비스는 내부 업무 효율화를 목적으로 한다.

질문:
월드비전의 2027년 총매출은 얼마인가요?
""",
    },
    {
        "name": "Context 준수",
        "prompt": """
아래 Context만 사용해서 질문에 답해주세요.

Context:
월드비전 AI 서비스의 MVP 기능은
사내 지식 검색 Agent,
AI 문서 작성 Agent,
회의 업무 Agent이다.

질문:
MVP 기능을 모두 알려주세요.
""",
    },
]


def main():
    print("OpenAI LLM Evaluation 시작\n")

    total_time = 0

    for index, test in enumerate(TEST_CASES, start=1):
        print("=" * 60)
        print(f"Test {index}: {test['name']}")
        print("=" * 60)

        try:
            start_time = time.perf_counter()

            response = generate_response(test["prompt"])

            elapsed_time = time.perf_counter() - start_time
            total_time += elapsed_time

            print(response)

            print(f"\n응답 시간: {elapsed_time:.2f}초")

        except Exception as e:
            print(f"테스트 실패: {e}")

        print()

    average_time = total_time / len(TEST_CASES)

    print("=" * 60)
    print("Evaluation 완료")
    print(f"평균 응답 시간: {average_time:.2f}초")
    print("=" * 60)


if __name__ == "__main__":
    main()