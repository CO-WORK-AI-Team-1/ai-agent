# AI Agent Project

World Vesion RFP 기반 요구사항을 바탕으로 AI Agent를 설계하고 구현하는 팀 프로젝트입니다.

단일 AI Agent 프로토타입 구현을 시작으로 RAG(Retrieval-Augmented Generation)와
Multi-Agent 구조를 적용하여 최종 AI Agent 서비스를 구현하고 배포하는 것을 목표로 합니다.

---

## 👥 Team

| Role | Member |
|------|--------|
| Product Owner | 윤여빈 |
| AI Engineer | 정은영 |
| AI Engineer | 임진영 |
| AI Solution Architect | 이수현 |
| Cloud Engineer | 은휘찬 |

---

## 🎯 Project Goals

- RFP 기반 요구사항 분석
- AI Agent 기능 및 Persona 설계
- RAG 기반 문서 검색 및 질의응답
- Multi-Agent Architecture 설계 및 구현
- Agent 간 협업 Workflow 구현
- 서비스 배포 및 운영 환경 구축

---

## 🛠 Tech Stack

### AI / Application

- Python
- LangChain
- LangGraph
- OpenAI API
- FAISS
- Streamlit

### Infrastructure

- TBD

### Collaboration

- Git
- GitHub

---

## 🏗 System Architecture

> Multi-Agent Architecture 설계 후 업데이트 예정

---

## 📁 Project Structure

> 프로젝트 구조 확정 후 업데이트 예정

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

`.env.example`을 복사하여 `.env` 파일을 생성합니다.

```text
OPENAI_API_KEY=your_api_key
```

> `.env` 파일 및 API Key와 같은 Secret 정보는 Git에 Commit하지 않습니다.

### 6. Run Application

```bash
streamlit run app.py
```

---

## 🌿 Branch Convention

```text
main
└── develop
    └── feature/*
```

### Branch

| Branch | Description |
|--------|-------------|
| `main` | 배포 가능한 안정 버전 |
| `develop` | 개발 통합 브랜치 |
| `feature/*` | 기능 단위 개발 브랜치 |

### Example

```text
feature/rag
feature/analyzer-agent
feature/writer-agent
feature/reviewer-agent
feature/deploy
```

기능 개발은 `feature/*` 브랜치에서 진행하고 Pull Request를 통해 `develop`에 병합합니다.

---

## 📝 Commit Convention

| Type | Description |
|------|-------------|
| `feat` | 새로운 기능 추가 |
| `fix` | 버그 수정 |
| `refactor` | 코드 리팩토링 |
| `docs` | 문서 수정 |
| `test` | 테스트 코드 |
| `chore` | 설정 및 기타 작업 |

### Example

```text
feat: add RFP analyzer agent
fix: handle PDF parsing error
refactor: separate RAG module
docs: update README
chore: update dependencies
```

---

## 🔀 Pull Request

1. `develop`에서 새로운 `feature/*` 브랜치를 생성합니다.
2. 기능 단위로 개발 및 Commit을 진행합니다.
3. 원격 Repository에 Push합니다.
4. `feature/*` → `develop` Pull Request를 생성합니다.
5. Code Review 후 Merge합니다.
6. 배포 가능한 버전은 `develop` → `main`으로 Merge합니다.

---

## 🔐 Security

다음 정보는 Git Repository에 Commit하지 않습니다.

- API Key
- `.env`
- Password / Secret
- 개인 인증 정보

환경변수가 필요한 경우 `.env.example`에 변수 이름만 추가합니다.

---

## 📄 License

TBD
