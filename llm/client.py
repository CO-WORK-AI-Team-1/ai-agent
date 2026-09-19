import os

from dotenv import load_dotenv
from openai import OpenAI


# .env 파일의 환경변수를 호출
load_dotenv()


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY가 설정되지 않았습니다. "
            ".env 파일을 확인해주세요."
        )

    return OpenAI(api_key=api_key)


def generate_response(prompt: str) -> str:
    if not prompt.strip():
        raise ValueError("질문이 비어 있습니다.")

    client = get_openai_client()

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
    )

    return response.output_text