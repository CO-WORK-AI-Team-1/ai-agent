# WorldVision AI Agent Demo

PDF 또는 TXT 문서를 업로드하고 질문하면 FAISS와 BM25 Hybrid Search로 관련 문맥을 검색한 뒤,
LLM이 검색 근거를 바탕으로 답변하는 Streamlit 데모입니다.

## 주요 기능

- PDF/TXT 텍스트 추출
- 페이지별 문서 Chunking
- OpenAI Embedding
- FAISS Vector Search
- BM25 Keyword Search
- FAISS 70% + BM25 30% Hybrid Retrieval
- 검색 문맥 기반 LLM 답변
- 답변 출처, PDF 페이지와 검색 원문 표시

## 로컬 실행

Python 3.11 이상을 권장합니다.

### 1. 가상환경 생성 및 활성화

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. 패키지 설치

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 환경변수 설정

`.env.example`을 `.env`로 복사한 뒤 개인 API 키를 입력합니다.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

필수 환경변수:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL_DEFAULT=gpt-5.6-luna
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 4. 앱 실행

`demo` 폴더 안에서 실행할 경우:

```bash
streamlit run app.py
```

팀 저장소 루트에서 실행할 경우:

```bash
streamlit run demo/app.py
```

## 클라우드 배포 정보

- Main file path: `demo/app.py`
- Dependency file: `demo/requirements.txt`
- 최대 업로드 크기: 20MB
- 최대 추출 텍스트: 200,000자
- Chunk size: 1,000자
- Chunk overlap: 150자
- 검색 결과 수 `k`: 5

배포 플랫폼의 Secrets에 다음 값을 등록합니다.

```toml
OPENAI_API_KEY = "your_api_key_here"
OPENAI_MODEL_DEFAULT = "gpt-5.6-luna"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
```

실제 `.env`, `.venv`, `.streamlit/secrets.toml`, API 키와 내부 문서는 Git에 올리지 마세요.

## 테스트

```bash
pytest
```

현재 테스트는 Chunking과 FAISS+BM25 Hybrid Search의 기본 동작을 확인합니다.
