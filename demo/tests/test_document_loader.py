from src.document_loader import clean_extracted_text


def test_clean_extracted_text_removes_duplicate_number_fragments():
    raw = "156\n156,223\n223\n명\n57\n57\n개"

    cleaned = clean_extracted_text(raw)

    assert cleaned == "156,223 명\n57 개"


def test_clean_extracted_text_normalizes_spaces_and_control_characters():
    raw = "총수입\u2002  386,585,434,203\n원\x00"

    cleaned = clean_extracted_text(raw)

    assert cleaned == "총수입 386,585,434,203\n원"
