from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import pymupdf

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
NUMBER_PATTERN = re.compile(r"^\d[\d,]*(?:\.\d+)?$")
UNIT_PATTERN = re.compile(
    r"^(?:명|개|개국|원|%|퍼센트|톤|헥타르|킬로그램|제곱미터|곳|대|년|월|일)$"
)


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


def clean_extracted_text(text: str) -> str:
    """PDF 추출 과정의 제어문자, 중복 숫자 조각과 단위 줄바꿈을 정리합니다."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(
        character
        for character in normalized
        if character in "\n\t" or unicodedata.category(character) != "Cc"
    )
    raw_lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in normalized.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]

    deduplicated: list[str] = []
    for line in raw_lines:
        if line and deduplicated and line == deduplicated[-1]:
            continue
        deduplicated.append(line)

    def is_number_fragment(index: int) -> bool:
        line = deduplicated[index]
        if not line.isdigit() or len(line) > 3:
            return False
        for adjacent_index in (index - 1, index + 1):
            if not 0 <= adjacent_index < len(deduplicated):
                continue
            adjacent = deduplicated[adjacent_index]
            if not NUMBER_PATTERN.fullmatch(adjacent) or "," not in adjacent:
                continue
            adjacent_digits = adjacent.replace(",", "").split(".", maxsplit=1)[0]
            if adjacent_digits.startswith(line) or adjacent_digits.endswith(line):
                return True
        return False

    filtered = [
        line
        for index, line in enumerate(deduplicated)
        if not is_number_fragment(index)
    ]

    combined: list[str] = []
    index = 0
    while index < len(filtered):
        line = filtered[index]
        if (
            line
            and NUMBER_PATTERN.fullmatch(line)
            and index + 1 < len(filtered)
            and UNIT_PATTERN.fullmatch(filtered[index + 1])
        ):
            combined.append(f"{line} {filtered[index + 1]}")
            index += 2
            continue
        combined.append(line)
        index += 1

    cleaned = "\n".join(combined)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_document(filename: str, data: bytes) -> LoadedDocument:
    if not data:
        raise ValueError("빈 파일은 업로드할 수 없습니다.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("파일 크기는 20MB를 초과할 수 없습니다.")

    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        with pymupdf.open(stream=data, filetype="pdf") as document:
            pages = tuple(clean_extracted_text(page.get_text("text")) for page in document)
        text = "\n\n".join(page for page in pages if page)
        page_count = len(pages)
    elif suffix == ".txt":
        text = clean_extracted_text(data.decode("utf-8-sig"))
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
