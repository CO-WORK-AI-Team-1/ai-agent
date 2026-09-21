import os

from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()


def get_claude_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY가 설정되지 않았습니다. "
            ".env 파일을 확인해주세요."
        )

    return Anthropic(api_key=api_key)


def generate_claude_response(prompt: str) -> str:
    if not prompt.strip():
        raise ValueError("질문이 비어 있습니다.")

    client = get_claude_client()

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.content[0].text