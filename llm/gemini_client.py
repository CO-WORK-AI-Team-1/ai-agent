import os
import time

from dotenv import load_dotenv
from google import genai


load_dotenv()


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY가 설정되지 않았습니다. "
            ".env 파일을 확인해주세요."
        )

    return genai.Client(api_key=api_key)


def generate_gemini_response(prompt: str, max_retries: int = 3) -> str:
    if not prompt.strip():
        raise ValueError("질문이 비어 있습니다.")

    client = get_gemini_client()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
            )

            return response.text

        except Exception as e:
            error_message = str(e)

            if "503" in error_message or "UNAVAILABLE" in error_message:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt

                    print(
                        f"Gemini 서버가 혼잡합니다. "
                        f"{wait_time}초 후 재시도합니다..."
                    )

                    time.sleep(wait_time)
                    continue

            raise