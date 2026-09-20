from dataclasses import dataclass
from pathlib import Path

import pymupdf

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class LoadedDocument:
    filename: str
    file_type: str
    page_count: int
    text: str
    pages: tuple[str, ...]

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def citation_context(self) -> str:
        return "\n\n".join(
            f"[P{page_number}]\n{page_text}"
            for page_number, page_text in enumerate(self.pages, start=1)
            if page_text
        )


def load_document(filename: str, data: bytes) -> LoadedDocument:
    if not data:
        raise ValueError("빈 파일은 업로드할 수 없습니다.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("파일 크기는 20MB를 초과할 수 없습니다.")

    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        with pymupdf.open(stream=data, filetype="pdf") as document:
            pages = tuple(page.get_text("text").strip() for page in document)
        text = "\n\n".join(page for page in pages if page)
        page_count = len(pages)
    elif suffix == ".txt":
        text = data.decode("utf-8-sig").strip()
        pages = (text,)
        page_count = 1
    else:
        raise ValueError("현재는 PDF와 TXT 파일만 지원합니다.")

    if not text:
        raise ValueError("파일에서 텍스트를 추출하지 못했습니다. 스캔 PDF 여부를 확인하세요.")

    return LoadedDocument(
        filename=filename,
        file_type=suffix.removeprefix("."),
        page_count=page_count,
        text=text,
        pages=pages,
    )
