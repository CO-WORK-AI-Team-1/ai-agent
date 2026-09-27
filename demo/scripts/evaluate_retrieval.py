"""사내 검색 Agent의 검색 설정 조합을 실제 평가 세트로 비교합니다."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.grounding import (
    SOURCE_CITATION_PATTERN,
    assess_answerability,
    validate_numeric_date_facts,
)
from src.prompts.week2_rag import render_week2_prompt
from src.rag import build_document_index, render_retrieved_context, search_document
from src.services.llm import invoke_response
from src.workflow import validate_source_citations


class CachedEmbeddings:
    """동일 모델·텍스트의 임베딩 API 호출을 평가 실행 중 재사용합니다."""

    def __init__(self, embeddings: Any):
        self._embeddings = embeddings
        self._cache: dict[tuple[str, str], list[float]] = {}

    def create(self, *, model: str, input: list[str]):
        missing = [text for text in input if (model, text) not in self._cache]
        if missing:
            response = self._embeddings.create(model=model, input=missing)
            for text, item in zip(missing, response.data, strict=True):
                self._cache[(model, text)] = item.embedding
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=self._cache[(model, text)])
                for text in input
            ]
        )


class EvaluationClient:
    def __init__(self, api_key: str):
        api = OpenAI(api_key=api_key, timeout=60.0, max_retries=2)
        self.embeddings = CachedEmbeddings(api.embeddings)
        self.responses = api.responses


@dataclass(frozen=True)
class EvaluationConfig:
    chunk_size: int
    overlap: int
    top_k: int
    min_score: float
    min_vector_score: float | None
    vector_weight: float
    bm25_weight: float


@dataclass(frozen=True)
class EvaluationMetrics:
    question_count: int
    answerable_count: int
    unanswerable_count: int
    hit_at_k: float
    mrr: float
    balanced_accuracy: float
    false_rejection_rate: float
    not_found_accuracy: float
    unanswerable_false_positive_rate: float
    answerability_accuracy: float | None
    end_to_end_balanced_accuracy: float | None
    end_to_end_answerable_success_rate: float | None
    end_to_end_not_found_accuracy: float | None
    average_verification_latency_ms: float | None
    fact_validation_checked_count: int
    fact_validation_pass_rate: float | None
    average_answer_latency_ms: float | None


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_optional_float_list(value: str) -> list[float | None]:
    values: list[float | None] = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if not normalized:
            continue
        values.append(None if normalized in {"none", "off"} else float(normalized))
    return values


def _parse_weights(value: str) -> list[tuple[float, float]]:
    weights: list[tuple[float, float]] = []
    for item in value.split(","):
        vector_text, bm25_text = item.strip().split(":", maxsplit=1)
        vector_weight = float(vector_text)
        bm25_weight = float(bm25_text)
        if vector_weight < 0 or bm25_weight < 0 or vector_weight + bm25_weight <= 0:
            raise ValueError("가중치는 음수가 아니어야 하며 합계가 0보다 커야 합니다.")
        total = vector_weight + bm25_weight
        weights.append((vector_weight / total, bm25_weight / total))
    return weights


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    documents = dataset.get("documents")
    questions = dataset.get("questions")
    if not isinstance(documents, list) or not documents:
        raise ValueError("평가 세트에 documents 배열이 필요합니다.")
    if not isinstance(questions, list) or not questions:
        raise ValueError("평가 세트에 questions 배열이 필요합니다.")

    document_ids: set[str] = set()
    for document in documents:
        document_id = document.get("id")
        pages = document.get("pages")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("모든 문서에는 비어 있지 않은 id가 필요합니다.")
        if document_id in document_ids:
            raise ValueError(f"중복된 문서 id입니다: {document_id}")
        if not isinstance(pages, list) or not all(isinstance(page, str) for page in pages):
            raise ValueError(f"문서 {document_id}의 pages는 문자열 배열이어야 합니다.")
        document_ids.add(document_id)

    for question in questions:
        question_id = question.get("id", "(unknown)")
        if question.get("document_id") not in document_ids:
            raise ValueError(f"질문 {question_id}의 document_id를 찾을 수 없습니다.")
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"질문 {question_id}의 question이 비어 있습니다.")
        if not isinstance(question.get("answerable"), bool):
            raise ValueError(f"질문 {question_id}의 answerable은 boolean이어야 합니다.")
        expected_pages = question.get("expected_pages", [])
        if question["answerable"] and (
            not isinstance(expected_pages, list)
            or not expected_pages
            or not all(isinstance(page, int) and page > 0 for page in expected_pages)
        ):
            raise ValueError(f"답변 가능 질문 {question_id}에는 expected_pages가 필요합니다.")

    return dataset


def evaluate_config(
    *,
    dataset: dict[str, Any],
    config: EvaluationConfig,
    settings: Any,
    client: Any | None = None,
    index_cache: dict[tuple[int, int], dict[str, Any]] | None = None,
    evaluate_answerability: bool = False,
    answerability_cache: dict[tuple[str, str], Any] | None = None,
    fact_check_question_ids: set[str] | None = None,
) -> tuple[EvaluationMetrics, list[dict[str, Any]]]:
    cache_key = (config.chunk_size, config.overlap)
    indexes = index_cache.get(cache_key) if index_cache is not None else None
    if indexes is None:
        indexes = {}
        for document in dataset["documents"]:
            indexes[document["id"]] = build_document_index(
                filename=document.get("filename", f"{document['id']}.txt"),
                pages=tuple(document["pages"]),
                chunk_size=config.chunk_size,
                overlap=config.overlap,
                settings=settings,
                client=client,
            )
        if index_cache is not None:
            index_cache[cache_key] = indexes

    hits = 0
    reciprocal_rank_sum = 0.0
    false_rejections = 0
    correct_not_found = 0
    unanswerable_false_positives = 0
    answerable_count = 0
    unanswerable_count = 0
    answerability_correct = 0
    answerability_checked = 0
    end_to_end_answerable_successes = 0
    end_to_end_unanswerable_rejections = 0
    verification_latencies: list[int] = []
    fact_validation_results: list[bool] = []
    answer_latencies: list[int] = []
    details: list[dict[str, Any]] = []

    for question in dataset["questions"]:
        results = search_document(
            document_index=indexes[question["document_id"]],
            question=question["question"],
            k=config.top_k,
            min_score=config.min_score,
            min_vector_score=config.min_vector_score,
            vector_weight=config.vector_weight,
            bm25_weight=config.bm25_weight,
            settings=settings,
            client=client,
        )
        retrieved_pages = [result.chunk.page for result in results]
        expected_pages = set(question.get("expected_pages", []))
        first_relevant_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if result.chunk.page in expected_pages
            ),
            None,
        )
        answerability_decision = None
        answerability_reason = None
        verification_latency_ms = None
        fact_validation_passed = None
        unsupported_facts: list[str] = []
        answer_latency_ms = None
        if evaluate_answerability and results:
            context = render_retrieved_context(results)
            cache_key = (question["question"], context)
            decision = (
                answerability_cache.get(cache_key)
                if answerability_cache is not None
                else None
            )
            if decision is None:
                decision = assess_answerability(
                    context=context,
                    question=question["question"],
                    settings=settings,
                    client=client,
                )
                if answerability_cache is not None:
                    answerability_cache[cache_key] = decision
            answerability_decision = decision.answerable
            answerability_reason = decision.reason
            verification_latency_ms = decision.latency_ms
            verification_latencies.append(decision.latency_ms)
            expected_decision = bool(
                question["answerable"] and first_relevant_rank is not None
            )
            answerability_checked += 1
            if answerability_decision == expected_decision:
                answerability_correct += 1

        final_answerable = bool(
            results
            and first_relevant_rank is not None
            and (not evaluate_answerability or answerability_decision is True)
        )

        if (
            fact_check_question_ids
            and question.get("id") in fact_check_question_ids
            and results
            and (not evaluate_answerability or answerability_decision is True)
        ):
            context = render_retrieved_context(results)
            answer_result = invoke_response(
                prompt=render_week2_prompt(
                    context=context,
                    question=question["question"],
                ),
                model=settings.model_default,
                settings=settings,
                client=client,
            )
            answer_latency_ms = answer_result.latency_ms
            answer_latencies.append(answer_result.latency_ms)
            if answer_result.success:
                allowed_source_ids = {result.source_id for result in results}
                citations_valid, _ = validate_source_citations(
                    answer=answer_result.output_text,
                    allowed_source_ids=allowed_source_ids,
                )
                cited_source_ids = {
                    citation[1:-1]
                    for citation in SOURCE_CITATION_PATTERN.findall(
                        answer_result.output_text
                    )
                }
                cited_texts = [
                    result.chunk.text
                    for result in results
                    if result.source_id in cited_source_ids
                ]
                facts_valid, unsupported_facts = validate_numeric_date_facts(
                    answer=answer_result.output_text,
                    cited_source_texts=cited_texts,
                )
                fact_validation_passed = citations_valid and facts_valid
            else:
                fact_validation_passed = False
            fact_validation_results.append(bool(fact_validation_passed))

        if question["answerable"]:
            answerable_count += 1
            if not results:
                false_rejections += 1
            if first_relevant_rank is not None:
                hits += 1
                reciprocal_rank_sum += 1.0 / first_relevant_rank
            if final_answerable:
                end_to_end_answerable_successes += 1
        else:
            unanswerable_count += 1
            if results:
                unanswerable_false_positives += 1
            else:
                correct_not_found += 1
            if not results or (
                evaluate_answerability and answerability_decision is False
            ):
                end_to_end_unanswerable_rejections += 1

        details.append(
            {
                "question_id": question.get("id"),
                "answerable": question["answerable"],
                "expected_pages": sorted(expected_pages),
                "retrieved_pages": retrieved_pages,
                "scores": [round(result.score, 6) for result in results],
                "cosine_similarities": [
                    round(result.cosine_similarity, 6)
                    for result in results
                ],
                "first_relevant_rank": first_relevant_rank,
                "not_found": not results,
                "answerability_decision": answerability_decision,
                "answerability_reason": answerability_reason,
                "verification_latency_ms": verification_latency_ms,
                "fact_validation_passed": fact_validation_passed,
                "unsupported_facts": unsupported_facts,
                "answer_latency_ms": answer_latency_ms,
            }
        )

    hit_at_k = hits / answerable_count if answerable_count else 0.0
    not_found_accuracy = (
        correct_not_found / unanswerable_count if unanswerable_count else 0.0
    )
    end_to_end_answerable_rate = (
        end_to_end_answerable_successes / answerable_count
        if answerable_count
        else 0.0
    )
    end_to_end_not_found = (
        end_to_end_unanswerable_rejections / unanswerable_count
        if unanswerable_count
        else 0.0
    )
    metrics = EvaluationMetrics(
        question_count=len(dataset["questions"]),
        answerable_count=answerable_count,
        unanswerable_count=unanswerable_count,
        hit_at_k=hit_at_k,
        mrr=(reciprocal_rank_sum / answerable_count if answerable_count else 0.0),
        balanced_accuracy=(hit_at_k + not_found_accuracy) / 2.0,
        false_rejection_rate=(
            false_rejections / answerable_count if answerable_count else 0.0
        ),
        not_found_accuracy=not_found_accuracy,
        unanswerable_false_positive_rate=(
            unanswerable_false_positives / unanswerable_count
            if unanswerable_count
            else 0.0
        ),
        answerability_accuracy=(
            answerability_correct / answerability_checked
            if evaluate_answerability and answerability_checked
            else None
        ),
        end_to_end_balanced_accuracy=(
            (end_to_end_answerable_rate + end_to_end_not_found) / 2.0
            if evaluate_answerability
            else None
        ),
        end_to_end_answerable_success_rate=(
            end_to_end_answerable_rate if evaluate_answerability else None
        ),
        end_to_end_not_found_accuracy=(
            end_to_end_not_found if evaluate_answerability else None
        ),
        average_verification_latency_ms=(
            sum(verification_latencies) / len(verification_latencies)
            if verification_latencies
            else None
        ),
        fact_validation_checked_count=len(fact_validation_results),
        fact_validation_pass_rate=(
            sum(fact_validation_results) / len(fact_validation_results)
            if fact_validation_results
            else None
        ),
        average_answer_latency_ms=(
            sum(answer_latencies) / len(answer_latencies)
            if answer_latencies
            else None
        ),
    )
    return metrics, details


def build_configs(args: argparse.Namespace) -> list[EvaluationConfig]:
    configs = []
    for chunk_size in _parse_int_list(args.chunk_sizes):
        for overlap in _parse_int_list(args.overlaps):
            if overlap >= chunk_size:
                continue
            for top_k in _parse_int_list(args.top_ks):
                for min_score in _parse_float_list(args.min_scores):
                    for min_vector_score in _parse_optional_float_list(
                        args.min_vector_scores
                    ):
                        for vector_weight, bm25_weight in _parse_weights(args.weights):
                            configs.append(
                                EvaluationConfig(
                                    chunk_size=chunk_size,
                                    overlap=overlap,
                                    top_k=top_k,
                                    min_score=min_score,
                                    min_vector_score=min_vector_score,
                                    vector_weight=vector_weight,
                                    bm25_weight=bm25_weight,
                                )
                            )
    if not configs:
        raise ValueError("실행 가능한 검색 설정 조합이 없습니다.")
    return configs


def print_summary(rows: list[dict[str, Any]]) -> None:
    header = (
        "rank chunk overlap k hybrid cosine weights balanced gate-bal Hit@K   MRR  "
        "reject(no-answer) false-reject(answer)"
    )
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(rows, start=1):
        config = row["config"]
        metrics = row["metrics"]
        gate_balanced = metrics["end_to_end_balanced_accuracy"]
        gate_display = f"{gate_balanced:.1%}" if gate_balanced is not None else "-"
        print(
            f"{rank:>4} {config['chunk_size']:>5} {config['overlap']:>7} "
            f"{config['top_k']:>1} {config['min_score']:>6.2f} "
            f"{str(config['min_vector_score']):>6} "
            f"{config['vector_weight']:.1f}:{config['bm25_weight']:.1f} "
            f"{metrics['balanced_accuracy']:>8.1%} "
            f"{gate_display:>8} "
            f"{metrics['hit_at_k']:>7.1%} {metrics['mrr']:>6.3f} "
            f"{metrics['not_found_accuracy']:>17.1%} "
            f"{metrics['false_rejection_rate']:>20.1%}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="평가 세트 JSON 파일")
    parser.add_argument("--chunk-sizes", default="500,800,1000")
    parser.add_argument("--overlaps", default="100,150")
    parser.add_argument("--top-ks", default="3,5")
    parser.add_argument("--min-scores", default="0.45,0.55,0.65")
    parser.add_argument(
        "--min-vector-scores",
        default="none,0.30,0.35,0.40,0.45",
        help="절대 cosine 임계값 목록. 비활성화는 none",
    )
    parser.add_argument("--weights", default="0.7:0.3,0.5:0.5,0.8:0.2")
    parser.add_argument("--output", type=Path, help="상세 결과를 저장할 JSON 경로")
    parser.add_argument(
        "--evaluate-answerability",
        action="store_true",
        help="검색 후 LLM 답변 가능성 판정까지 평가",
    )
    parser.add_argument(
        "--fact-check-question-ids",
        default="",
        help="최종 답변과 수치·날짜 근거를 검증할 질문 ID 목록",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dataset = load_dataset(args.dataset)
        configs = build_configs(args)
        settings = get_settings()
        if not settings.api_key_configured:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

        rows = []
        index_cache: dict[tuple[int, int], dict[str, Any]] = {}
        client = EvaluationClient(settings.openai_api_key)
        answerability_cache: dict[tuple[str, str], Any] = {}
        fact_check_question_ids = {
            item.strip()
            for item in args.fact_check_question_ids.split(",")
            if item.strip()
        }
        for position, config in enumerate(configs, start=1):
            print(f"[{position}/{len(configs)}] 평가 중: {config}", file=sys.stderr)
            metrics, details = evaluate_config(
                dataset=dataset,
                config=config,
                settings=settings,
                client=client,
                index_cache=index_cache,
                evaluate_answerability=args.evaluate_answerability,
                answerability_cache=answerability_cache,
                fact_check_question_ids=fact_check_question_ids,
            )
            rows.append(
                {
                    "config": asdict(config),
                    "metrics": asdict(metrics),
                    "details": details,
                }
            )

        rows.sort(
            key=lambda row: (
                (
                    row["metrics"]["end_to_end_balanced_accuracy"]
                    if row["metrics"]["end_to_end_balanced_accuracy"] is not None
                    else row["metrics"]["balanced_accuracy"]
                ),
                -row["metrics"]["false_rejection_rate"],
                row["metrics"]["mrr"],
                row["metrics"]["hit_at_k"],
            ),
            reverse=True,
        )
        print_summary(rows)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as file:
                json.dump(rows, file, ensure_ascii=False, indent=2)
            print(f"상세 결과 저장: {args.output}", file=sys.stderr)
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"평가 실패: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
