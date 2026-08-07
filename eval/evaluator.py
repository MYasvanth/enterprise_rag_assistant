"""
Evaluation Harness for Enterprise RAG Knowledge Assistant.

Metrics:
  - Recall@k        : fraction of expected sources found in top-k retrieved docs
  - Faithfulness    : LLM-as-judge — is the answer grounded in the retrieved context?
  - Answer Correctness : ROUGE-L F1 between generated answer and ground truth
  - Latency         : wall-clock time per query (seconds)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from rouge_score import rouge_scorer
from langchain_openai import ChatOpenAI

# Allowed base directory for all eval file I/O
_EVAL_BASE_DIR = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    id: str
    question: str
    ground_truth: str
    expected_sources: List[str]
    type: str = "factual"
    tags: List[str] = field(default_factory=list)


@dataclass
class SampleResult:
    id: str
    question: str
    type: str
    tags: List[str]
    generated_answer: str
    ground_truth: str
    retrieved_sources: List[str]
    expected_sources: List[str]
    recall_at_k: float          # 0.0 – 1.0
    faithfulness: float         # 0.0 – 1.0  (LLM judge)
    answer_correctness: float   # ROUGE-L F1
    latency_seconds: float
    faithfulness_reason: str = ""
    error: Optional[str] = None


@dataclass
class EvalReport:
    total: int
    passed: int                 # faithfulness >= threshold AND recall >= threshold
    avg_recall_at_k: float
    avg_faithfulness: float
    avg_answer_correctness: float
    avg_latency_seconds: float
    by_type: Dict[str, Dict[str, float]]
    samples: List[SampleResult]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Evaluates a RAG pipeline or agent against a golden dataset.

    Usage:
        evaluator = RAGEvaluator(query_fn=pipeline.query, api_key="sk-...")
        report = evaluator.run(golden_dataset_path="eval/golden_dataset.json")
    """

    FAITHFULNESS_THRESHOLD = 0.7
    RECALL_THRESHOLD = 0.5

    def __init__(
        self,
        query_fn,                          # callable: (question: str) -> {"answer": str, "source_documents": list}
        api_key: str,
        judge_model: str = "gpt-3.5-turbo",
        k: int = 5,
    ):
        self._query_fn = query_fn
        self._k = k
        self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self._judge = ChatOpenAI(
            model_name=judge_model,
            openai_api_key=api_key,
            temperature=0,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, golden_dataset_path: str) -> EvalReport:
        """Run evaluation over the full golden dataset and return a report."""
        samples = self._load_dataset(golden_dataset_path)
        logger.info("Starting evaluation on %d samples", len(samples))

        results: List[SampleResult] = []
        for sample in samples:
            result = self._evaluate_sample(sample)
            results.append(result)
            logger.info(
                "[%s] recall=%.2f faith=%.2f rouge=%.2f latency=%.2fs",
                result.id, result.recall_at_k, result.faithfulness,
                result.answer_correctness, result.latency_seconds,
            )

        return self._build_report(results)

    # ------------------------------------------------------------------
    # Per-sample evaluation
    # ------------------------------------------------------------------

    def _evaluate_sample(self, sample: EvalSample) -> SampleResult:
        try:
            t0 = time.perf_counter()
            response = self._query_fn(sample.question)
            latency = time.perf_counter() - t0

            answer = response.get("answer", "")
            source_docs = response.get("source_documents", [])
            retrieved_sources = [
                doc.metadata.get("source", "") for doc in source_docs
            ]

            recall = self._recall_at_k(sample.expected_sources, retrieved_sources)
            context = "\n\n".join(doc.page_content for doc in source_docs)
            faithfulness, reason = self._faithfulness(sample.question, answer, context)
            correctness = self._answer_correctness(answer, sample.ground_truth)

            return SampleResult(
                id=sample.id,
                question=sample.question,
                type=sample.type,
                tags=sample.tags,
                generated_answer=answer,
                ground_truth=sample.ground_truth,
                retrieved_sources=retrieved_sources,
                expected_sources=sample.expected_sources,
                recall_at_k=recall,
                faithfulness=faithfulness,
                faithfulness_reason=reason,
                answer_correctness=correctness,
                latency_seconds=round(latency, 3),
            )

        except Exception as e:
            logger.error("Error evaluating sample %s: %s", sample.id, e)
            return SampleResult(
                id=sample.id,
                question=sample.question,
                type=sample.type,
                tags=sample.tags,
                generated_answer="",
                ground_truth=sample.ground_truth,
                retrieved_sources=[],
                expected_sources=sample.expected_sources,
                recall_at_k=0.0,
                faithfulness=0.0,
                answer_correctness=0.0,
                latency_seconds=0.0,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _recall_at_k(self, expected: List[str], retrieved: List[str]) -> float:
        """
        Fraction of expected sources that appear in the top-k retrieved sources.
        Matches on filename substring to handle full path vs basename differences.
        """
        if not expected:
            return 1.0
        hits = sum(
            1 for exp in expected
            if any(exp in ret or ret in exp for ret in retrieved[: self._k])
        )
        return round(hits / len(expected), 4)

    def _faithfulness(self, question: str, answer: str, context: str) -> tuple[float, str]:
        """
        LLM-as-judge: score 0.0–1.0 for how well the answer is grounded in context.
        Returns (score, reason).
        """
        prompt = f"""You are an evaluation judge. Score how faithfully the answer is grounded in the provided context.

Rules:
- Score 1.0  : every claim in the answer is directly supported by the context
- Score 0.5  : answer is partially supported; some claims lack context support
- Score 0.0  : answer contradicts or ignores the context entirely

Respond in this exact format (no extra text):
SCORE: <float between 0.0 and 1.0>
REASON: <one sentence>

Question: {question}
Context: {context[:2000]}
Answer: {answer}"""

        try:
            response = self._judge.invoke(prompt).content.strip()
            score_line = next(l for l in response.splitlines() if l.startswith("SCORE:"))
            reason_line = next((l for l in response.splitlines() if l.startswith("REASON:")), "REASON: N/A")
            score = float(score_line.split(":", 1)[1].strip())
            reason = reason_line.split(":", 1)[1].strip()
            return round(min(max(score, 0.0), 1.0), 4), reason
        except Exception as e:
            logger.warning("Faithfulness judge failed: %s", e)
            return 0.0, f"Judge error: {e}"

    def _answer_correctness(self, generated: str, ground_truth: str) -> float:
        """ROUGE-L F1 between generated answer and ground truth."""
        if not generated or not ground_truth:
            return 0.0
        scores = self._rouge.score(ground_truth, generated)
        return round(scores["rougeL"].fmeasure, 4)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _build_report(self, results: List[SampleResult]) -> EvalReport:
        total = len(results)
        if total == 0:
            raise ValueError("No results to report.")

        passed = sum(
            1 for r in results
            if r.faithfulness >= self.FAITHFULNESS_THRESHOLD
            and r.recall_at_k >= self.RECALL_THRESHOLD
        )

        def avg(values): return round(sum(values) / len(values), 4) if values else 0.0

        # Aggregate by question type
        types: Dict[str, List[SampleResult]] = {}
        for r in results:
            types.setdefault(r.type, []).append(r)

        by_type: Dict[str, Dict] = {}
        for qtype, group in types.items():
            by_type[qtype] = {
                "count": len(group),
                "avg_recall": avg([r.recall_at_k for r in group]),
                "avg_faithfulness": avg([r.faithfulness for r in group]),
                "avg_correctness": avg([r.answer_correctness for r in group]),
                "avg_latency": avg([r.latency_seconds for r in group]),
            }

        return EvalReport(
            total=total,
            passed=passed,
            avg_recall_at_k=avg([r.recall_at_k for r in results]),
            avg_faithfulness=avg([r.faithfulness for r in results]),
            avg_answer_correctness=avg([r.answer_correctness for r in results]),
            avg_latency_seconds=avg([r.latency_seconds for r in results]),
            by_type=by_type,
            samples=results,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_safe_path(path: str) -> Path:
        """Resolve path and ensure it stays within the project directory."""
        resolved = Path(path).resolve()
        if not str(resolved).startswith(str(_EVAL_BASE_DIR)):
            raise ValueError(
                f"Access denied: path '{resolved}' is outside the project directory."
            )
        return resolved

    @staticmethod
    def _load_dataset(path: str) -> List[EvalSample]:
        safe_path = RAGEvaluator._resolve_safe_path(path)
        with safe_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return [EvalSample(**item) for item in raw]

    @staticmethod
    def save_report(report: EvalReport, output_path: str) -> None:
        """Serialize the full report to JSON."""
        safe_path = RAGEvaluator._resolve_safe_path(output_path)
        os.makedirs(safe_path.parent, exist_ok=True)
        data = asdict(report)
        with safe_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Report saved to %s", safe_path)
