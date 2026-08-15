"""
CLI entry point for the RAG evaluation harness.

Usage:
    python -m eval.run_eval --api-key sk-... [--dataset eval/golden_dataset.json]
                            [--output eval/report.json] [--judge gpt-4o] [--k 5]
"""

import argparse
import logging
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.embedding.embedding import EmbeddingManager
from src.retrieval.retrieval import RAGPipeline
from eval.evaluator import RAGEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def build_query_fn(api_key: str):
    """Initialise the RAG pipeline and return its query callable."""
    embedding_manager = EmbeddingManager(
        embedding_provider="openai",
        vector_store="chroma",
        api_key=api_key,
    )
    persist_dir = "./data/chroma_db"
    embedding_manager.create_vector_store(persist_dir)

    pipeline = RAGPipeline(
        embedding_manager=embedding_manager,
        llm_provider="openai",
        api_key=api_key,
    )
    pipeline.create_qa_chain()
    return pipeline.query


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation harness")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--dataset", default="eval/golden_dataset.json", help="Path to golden dataset")
    parser.add_argument("--output", default="eval/report.json", help="Path to write JSON report")
    parser.add_argument("--judge", default="gpt-3.5-turbo", help="Judge model for faithfulness scoring")
    parser.add_argument("--k", type=int, default=5, help="Top-k for recall calculation")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("OpenAI API key required: use --api-key or set OPENAI_API_KEY")

    logger.info("Building RAG pipeline…")
    query_fn = build_query_fn(args.api_key)

    evaluator = RAGEvaluator(
        query_fn=query_fn,
        api_key=args.api_key,
        judge_model=args.judge,
        k=args.k,
    )

    report = evaluator.run(args.dataset)

    # Console summary
    print("\n" + "=" * 60)
    print(f"  Evaluation complete — {report.passed}/{report.total} samples passed")
    print(f"  Recall@{args.k}        : {report.avg_recall_at_k:.4f}")
    print(f"  Faithfulness     : {report.avg_faithfulness:.4f}")
    print(f"  Answer Correctness: {report.avg_answer_correctness:.4f}")
    print(f"  Avg Latency      : {report.avg_latency_seconds:.3f}s")
    print("=" * 60)

    if report.by_type:
        print("\nBy question type:")
        for qtype, stats in report.by_type.items():
            print(
                f"  {qtype:15s}  n={stats['count']}  "
                f"recall={stats['avg_recall']:.3f}  "
                f"faith={stats['avg_faithfulness']:.3f}  "
                f"rouge={stats['avg_correctness']:.3f}"
            )

    RAGEvaluator.save_report(report, args.output)
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
