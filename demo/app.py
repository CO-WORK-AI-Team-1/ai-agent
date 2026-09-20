import hashlib

import pymupdf
import streamlit as st

from src.config import get_settings
from src.document_loader import MAX_UPLOAD_BYTES, load_document
from src.rag import build_document_index
from src.workflow import run_document_workflow

st.set_page_config(page_title=" 월드비전 챗봇 ", page_icon="📄", layout="wide")


@st.cache_data(show_spinner=False)
def parse_uploaded_document(filename: str, data: bytes):
    return load_document(filename, data)


settings = get_settings()

st.title("월드비전 챗봇")
st.caption("PDF 또는 TXT 문서를 업로드하고 문서 내용에 관해 질문해 보세요.")

with st.sidebar:
    st.subheader("1. 문서 업로드")
    st.caption("지원 형식: PDF, TXT · 최대 20MB")
    consent = st.checkbox("문서 처리 권한과 민감정보 포함 여부를 확인했습니다.")
    uploaded_file = st.file_uploader(
        "분석할 파일을 선택하세요",
        type=["pdf", "txt"],
        )

document = None
if uploaded_file is None:
    st.info("왼쪽 사이드바에서 PDF 또는 TXT 파일을 업로드하세요.")
elif not consent:
    st.warning("문서 처리 권한과 민감정보 여부를 먼저 확인해 주세요.")
else:
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        st.error("파일 크기가 20MB를 초과합니다.")
    else:
        try:
            with st.spinner("문서에서 텍스트를 추출하는 중입니다..."):
                document = parse_uploaded_document(uploaded_file.name, file_bytes)
            file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
            st.success("문서 업로드와 텍스트 추출이 완료되었습니다.")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("파일", document.filename)
            col2.metric("형식", document.file_type.upper())
            col3.metric("페이지", document.page_count)
            col4.metric("추출 문자", f"{document.character_count:,}")
            st.caption(f"문서 식별자: `{file_hash}`")

            with st.expander("추출된 텍스트 미리보기"):
                preview = document.text[:2000]
                st.text(preview)
                if document.character_count > len(preview):
                    st.caption("미리보기는 앞부분 2,000자까지만 표시합니다.")
        except (ValueError, pymupdf.FileDataError) as error:
            st.error(str(error))

st.subheader("2. 문서 기반 요청")
request = st.text_area(
    "질문 또는 요청",
    value="이 문서의 주요 내용과 확인이 필요한 정보를 정리해 주세요.",
    height=130,
    disabled=document is None,
)

can_run = document is not None and bool(request.strip())

if st.button("문서 분석하기", type="primary", disabled=not can_run):
    with st.spinner("문서 검색 인덱스를 준비하고 관련 내용을 분석하고 있습니다..."):
        try:
            index_key = f"{file_hash}:{settings.embedding_model}"
            if st.session_state.get("document_index_key") != index_key:
                st.session_state.document_index = build_document_index(
                    filename=document.filename,
                    pages=document.pages,
                    settings=settings,
                )
                st.session_state.document_index_key = index_key
            result = run_document_workflow(
                document_index=st.session_state.document_index,
                question=request,
                settings=settings,
            )
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
        except Exception:
            st.error("서비스 연결 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
        else:
            if result["success"]:
                st.success("문서 분석이 완료되었습니다.")
                st.markdown("#### 분석 결과")
                st.write(result["answer"])
                st.caption(
                    f"출처 문서: {document.filename} · 답변의 [S번호]는 아래 검색 근거입니다."
                )
                with st.expander("검색 근거 확인"):
                    for source in result["sources"]:
                        st.markdown(
                            f"**[{source['source_id']}] {source['filename']} · "
                            f"{source['page']}페이지**"
                        )
                        st.caption(f"유사도: {source['score']:.3f}")
                        st.write(source["text"])
                        st.divider()
                if result["is_incomplete"]:
                    st.warning(
                        "응답 길이 제한으로 일부 내용이 생략되었을 수 있습니다. "
                        "질문 범위를 좁혀 다시 요청해 주세요."
                    )
            else:
                st.error(result["error"])

st.divider()