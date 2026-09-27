"""2025 한국월드비전 연차보고서 PDF에서 검색 평가 데이터셋을 생성합니다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_document


QUESTIONS = [
    {"id": "q01", "question": "2025년 한국월드비전이 전 세계에서 도운 사람은 총 몇 명이며, 아동과 성인은 각각 몇 명인가요?", "answerable": True, "expected_pages": [3]},
    {"id": "q02", "question": "2025년 한국월드비전은 몇 개국에서 몇 개 사업을 진행했나요?", "answerable": True, "expected_pages": [3]},
    {"id": "q03", "question": "국제사업, 국내사업, 옹호사업을 통해 다시 일어날 희망을 선물 받은 사람은 각각 몇 명인가요?", "answerable": True, "expected_pages": [4]},
    {"id": "q04", "question": "10년 이상, 20년 이상, 30년 이상 장기 후원자는 각각 몇 명인가요?", "answerable": True, "expected_pages": [5]},
    {"id": "q05", "question": "교육시설과 학습 환경이 개선된 학교는 몇 개이며, 기초 문해력 평가 기준을 충족한 아동은 몇 명인가요?", "answerable": True, "expected_pages": [7]},
    {"id": "q06", "question": "국제 교육사업의 총수혜자는 아동과 성인이 각각 몇 명인가요?", "answerable": True, "expected_pages": [7]},
    {"id": "q07", "question": "식수위생 사업에서 설치하거나 수리한 식수시설과 화장실은 각각 몇 개인가요?", "answerable": True, "expected_pages": [8]},
    {"id": "q08", "question": "예방접종이나 필수 의료 서비스를 받은 아동과 영양 상태가 향상된 아동은 각각 몇 명인가요?", "answerable": True, "expected_pages": [8]},
    {"id": "q09", "question": "아동권리·옹호 활동에 적극적으로 참여한 아동 그룹은 몇 개이며 관련 인식 개선 활동에 참여하거나 접한 주민은 몇 명인가요?", "answerable": True, "expected_pages": [9]},
    {"id": "q10", "question": "소득증대 사업을 통해 새로운 기술을 습득한 사람과 두 가지 이상의 소득원을 보유하게 된 가구는 각각 얼마인가요?", "answerable": True, "expected_pages": [9]},
    {"id": "q11", "question": "2025년 인도적지원 사업은 몇 개국에서 몇 개 진행됐으며 총수혜자는 몇 명인가요?", "answerable": True, "expected_pages": [10]},
    {"id": "q12", "question": "인도적지원 중 긴급 식량 지원, 식수위생 지원, 다목적 현금 지원을 받은 사람은 각각 몇 명인가요?", "answerable": True, "expected_pages": [10]},
    {"id": "q13", "question": "기후변화 사업으로 보호되거나 복원된 토지 면적과 재난 대응 능력이 향상된 주민 수는 얼마인가요?", "answerable": True, "expected_pages": [11]},
    {"id": "q14", "question": "북한사업 30년사 백서는 언제 발간될 예정이며 어떤 용도로 활용될 계획인가요?", "answerable": True, "expected_pages": [11]},
    {"id": "q15", "question": "국내 꿈 지원 사업의 총수혜자 수와 꿈디자이너 참여 아동 수는 얼마인가요?", "answerable": True, "expected_pages": [12]},
    {"id": "q16", "question": "위기아동 지원의 총수혜자 수와 긴급위기 지원을 받은 아동 수는 각각 몇 명인가요?", "answerable": True, "expected_pages": [13]},
    {"id": "q17", "question": "아침머꼬, 주말에뭐먹니, 사랑의도시락, 쿡n쑥쑥의 수혜 아동은 각각 몇 명인가요?", "answerable": True, "expected_pages": [13]},
    {"id": "q18", "question": "기적의 존 사업의 지원 시설 수와 연간 온실가스 감축 기대 효과는 얼마인가요?", "answerable": True, "expected_pages": [14]},
    {"id": "q19", "question": "E-share 사업을 통해 지원한 전기차와 충전기는 각각 몇 대와 몇 개인가요?", "answerable": True, "expected_pages": [15]},
    {"id": "q20", "question": "2025년 아동권리 옹호 사업에서 추진한 공공정책 개선 활동의 주요 사례를 설명해 주세요.", "answerable": True, "expected_pages": [16]},
    {"id": "q21", "question": "세계시민학교의 전체 참여자와 교사가 진행하는 세계시민교육 참여자는 각각 몇 명인가요?", "answerable": True, "expected_pages": [17]},
    {"id": "q22", "question": "2025년 한국월드비전의 총수입은 얼마인가요?", "answerable": True, "expected_pages": [18]},
    {"id": "q23", "question": "2025년 수입에서 후원금, 사업준비금, 정부 보조금, 기타 수입의 비율은 각각 얼마인가요?", "answerable": True, "expected_pages": [18]},
    {"id": "q24", "question": "2025년 총지출과 사업비는 각각 얼마이며 사업비 비율은 몇 퍼센트인가요?", "answerable": True, "expected_pages": [19]},
    {"id": "q25", "question": "해외사업비 중 자립마을지원사업과 인도적지원사업 지출액은 각각 얼마인가요?", "answerable": True, "expected_pages": [19]},
    {"id": "q26", "question": "2025년 국제 총사업과 국내 총사업은 각각 몇 개인가요?", "answerable": True, "expected_pages": [20]},
    {"id": "q27", "question": "동아프리카에서 케냐와 에티오피아의 사업 수는 각각 몇 개인가요?", "answerable": True, "expected_pages": [20]},
    {"id": "q28", "question": "2025 한국월드비전 연차보고서의 발행일과 발행·편집인은 누구인가요?", "answerable": True, "expected_pages": [21]},
    {"id": "q29", "question": "한국월드비전 회장의 개인 휴대전화 번호를 알려주세요.", "answerable": False, "expected_pages": []},
    {"id": "q30", "question": "2024년 한국월드비전의 총수입은 얼마였나요?", "answerable": False, "expected_pages": []},
    {"id": "q31", "question": "2026년 월드비전 국제사업의 실제 총수혜자는 몇 명인가요?", "answerable": False, "expected_pages": []},
    {"id": "q32", "question": "2025년 한국월드비전 직원의 평균 연봉은 얼마인가요?", "answerable": False, "expected_pages": []},
    {"id": "q33", "question": "월드비전 서울서부사업본부의 평일 운영시간은 어떻게 되나요?", "answerable": False, "expected_pages": []},
    {"id": "q34", "question": "2025년에 신규 가입한 후원자의 평균 연령은 몇 세인가요?", "answerable": False, "expected_pages": []}
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loaded_document = load_document(args.pdf.name, args.pdf.read_bytes())
    pages = list(loaded_document.pages)

    dataset = {
        "documents": [
            {
                "id": "worldvision_2025_annual_report",
                "filename": args.pdf.name,
                "pages": pages,
            }
        ],
        "questions": [
            {
                **question,
                "document_id": "worldvision_2025_annual_report",
            }
            for question in QUESTIONS
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"created: {args.output} ({len(pages)} pages, {len(QUESTIONS)} questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
