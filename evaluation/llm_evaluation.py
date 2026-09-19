import time

from llm.client import generate_response


TEST_PROMPT = """
월드비전 AI 서비스가 필요한 이유를
3가지로 간단하게 설명해주세요.
"""


def main():
    try:
        print("OpenAI API 테스트를 시작합니다.\n")

        start_time = time.perf_counter()

        response = generate_response(TEST_PROMPT)

        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        print("=== LLM 응답 ===")
        print(response)

        print("\n=== 테스트 결과 ===")
        print("API 연결: 성공")
        print(f"응답 시간: {elapsed_time:.2f}초")

    except Exception as e:
        print("=== 테스트 실패 ===")
        print(e)


if __name__ == "__main__":
    main()