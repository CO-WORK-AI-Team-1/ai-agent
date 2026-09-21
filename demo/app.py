import hashlib

import pymupdf
import streamlit as st

from src.config import get_settings
from src.document_loader import MAX_UPLOAD_BYTES, load_document
from src.rag import build_document_index
from src.services.transcription import (
    MAX_AUDIO_UPLOAD_BYTES,
    SUPPORTED_AUDIO_SUFFIXES,
)
from src.workflow import (
    run_document_workflow,
    run_document_writer_workflow,
    run_meeting_workflow,
)

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
    consent = st.checkbox("문서와 회의 녹음 처리 권한, 민감정보 포함 여부를 확인했습니다.")
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

search_tab, writer_tab, meeting_tab = st.tabs(
    ["사내 지식 검색", "AI 문서 작성", "회의 업무"]
)

with search_tab:
    st.subheader("2. 문서 기반 질문")
    request = st.text_area(
        "질문 또는 요청",
        value="이 문서의 주요 내용과 확인이 필요한 정보를 정리해 주세요.",
        height=130,
        disabled=document is None,
        key="search_request",
    )
    can_search = document is not None and bool(request.strip())

    if st.button("문서 분석하기", type="primary", disabled=not can_search):
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

with writer_tab:
    st.subheader("2. 내부 자료 기반 문서 초안")
    st.caption("업로드한 자료에서 관련 근거를 검색해 보고서 또는 업무 문서 초안을 만듭니다.")
    document_type = st.selectbox(
        "문서 유형",
        ["업무 보고서", "사업 계획서", "제안서", "공지문", "기타 업무 문서"],
        disabled=document is None,
    )
    purpose = st.text_area(
        "작성 목적과 포함할 내용",
        placeholder="예: 2026년 교육 지원 사업의 추진 현황과 다음 분기 계획을 정리한 보고서를 작성해 주세요.",
        height=120,
        disabled=document is None,
    )
    audience = st.text_input(
        "예상 독자",
        value="월드비전 내부 구성원",
        disabled=document is None,
    )
    additional_requirements = st.text_area(
        "추가 요청 사항 (선택)",
        placeholder="예: 한 페이지 분량으로 간결하게 작성하고, 예산 정보는 별도 소제목으로 정리해 주세요.",
        height=90,
        disabled=document is None,
    )
    can_write = document is not None and bool(purpose.strip())

    if st.button("문서 초안 만들기", type="primary", disabled=not can_write):
        with st.spinner("관련 자료를 검색하고 문서 초안을 작성하고 있습니다..."):
            try:
                index_key = f"{file_hash}:{settings.embedding_model}"
                if st.session_state.get("document_index_key") != index_key:
                    st.session_state.document_index = build_document_index(
                        filename=document.filename,
                        pages=document.pages,
                        settings=settings,
                    )
                    st.session_state.document_index_key = index_key
                result = run_document_writer_workflow(
                    document_index=st.session_state.document_index,
                    purpose=purpose,
                    document_type=document_type,
                    audience=audience,
                    additional_requirements=additional_requirements,
                    settings=settings,
                )
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
            except Exception:
                st.error("서비스 연결 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
            else:
                if result["success"]:
                    st.success("문서 초안이 생성되었습니다.")
                    st.markdown("#### 문서 초안")
                    st.markdown(result["draft"])
                    st.download_button(
                        "Markdown 파일로 저장",
                        data=result["draft"],
                        file_name="worldvision_document_draft.md",
                        mime="text/markdown",
                    )
                    st.caption(
                        "초안의 [S번호]는 아래 검색 근거와 연결됩니다. "
                        "배포 전 사실과 표현을 검토해 주세요."
                    )
                    with st.expander("작성에 사용한 검색 근거"):
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
                            "요청 범위를 좁혀 다시 작성해 주세요."
                        )
                else:
                    st.error(result["error"])

with meeting_tab:
    st.subheader("2. 회의 녹음 기반 회의록")
    st.caption(
        "녹음 파일을 텍스트로 전사한 뒤, 핵심 요약·결정 사항·실행 항목을 포함한 회의록을 만듭니다."
    )
    meeting_file = st.file_uploader(
        "회의 녹음 파일",
        type=[suffix.removeprefix(".") for suffix in SUPPORTED_AUDIO_SUFFIXES],
        key="meeting_audio",
        help=(
            "지원 형식: MP3, MP4, MPEG, MPGA, M4A, WAV, WEBM · "
            f"최대 {MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)}MB"
        ),
    )
    meeting_title = st.text_input(
        "회의 제목",
        placeholder="예: AI 서비스 플랫폼 MVP 주간 회의",
        disabled=meeting_file is None,
    )
    meeting_focus = st.text_area(
        "정리 중점 (선택)",
        placeholder="예: MVP 기능별 담당자와 다음 주까지의 실행 항목을 중심으로 정리해 주세요.",
        height=90,
        disabled=meeting_file is None,
    )
    can_process_meeting = (
        consent
        and meeting_file is not None
        and bool(meeting_title.strip())
    )
    if not consent:
        st.info("왼쪽 사이드바에서 녹음 처리 권한과 민감정보 여부를 먼저 확인해 주세요.")

    if st.button(
        "회의록 만들기",
        type="primary",
        disabled=not can_process_meeting,
    ):
        with st.spinner("음성을 전사하고 회의록을 작성하고 있습니다..."):
            try:
                result = run_meeting_workflow(
                    filename=meeting_file.name,
                    audio_data=meeting_file.getvalue(),
                    meeting_title=meeting_title,
                    focus=meeting_focus,
                    settings=settings,
                )
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
            except Exception:
                st.error("회의 업무 서비스 연결 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
            else:
                if result["success"]:
                    st.success("회의록이 생성되었습니다.")
                    st.markdown("#### 회의록")
                    st.markdown(result["minutes"])
                    st.download_button(
                        "회의록 Markdown 파일로 저장",
                        data=result["minutes"],
                        file_name="worldvision_meeting_minutes.md",
                        mime="text/markdown",
                    )
                    with st.expander("전사 원문 확인"):
                        st.text(result["transcript"])
                    st.download_button(
                        "전사 원문 TXT 파일로 저장",
                        data=result["transcript"],
                        file_name="worldvision_meeting_transcript.txt",
                        mime="text/plain",
                    )
                    if result["is_incomplete"]:
                        st.warning(
                            "응답 길이 제한으로 일부 내용이 생략되었을 수 있습니다. "
                            "녹음 파일을 나누거나 정리 중점을 좁혀 다시 시도해 주세요."
                        )
                else:
                    st.error(result["error"])

st.divider()
