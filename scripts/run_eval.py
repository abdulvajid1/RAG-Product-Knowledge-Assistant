import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import setup_logging
from app.evaluation.runner import EvaluationRunner

logger = logging.getLogger("scripts.run_eval")


async def main():
    setup_logging()
    data_eval_dir = Path("data/eval").resolve()
    data_eval_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_eval_dir / "eval_questions.json"
    if not dataset_path.exists():
        print(f"Error: Evaluation dataset not found at {dataset_path}")
        sys.exit(1)

    runner = EvaluationRunner(dataset_path=dataset_path)

    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK EVALUATION (BASELINE vs HYBRID)")
    print("=" * 60)

    # 1. Run Baseline
    print("\n[1/2] Evaluating RETRIEVAL_MODE=baseline (Vector-only)...")
    baseline_out = data_eval_dir / "results_baseline.json"
    baseline_results = await runner.run(mode="baseline", top_k=5, output_file=baseline_out)

    # 2. Run Hybrid (Improved)
    print("\n[2/2] Evaluating RETRIEVAL_MODE=hybrid (BM25 + Vector + RRF)...")
    improved_out = data_eval_dir / "results_improved.json"
    improved_results = await runner.run(mode="hybrid", top_k=5, output_file=improved_out)

    # 3. Print Comparison Table
    print("\n" + "=" * 60)
    print("RETRIEVAL & ACCURACY COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<28} | {'Baseline (Vector)':<18} | {'Improved (Hybrid RRF)':<20} | {'Delta':<10}")
    print("-" * 84)

    metrics_to_compare = [
        ("Recall@1", "mean_recall_at_1"),
        ("Recall@3", "mean_recall_at_3"),
        ("Recall@5", "mean_recall_at_5"),
        ("Mean Reciprocal Rank (MRR)", "mean_mrr"),
        ("Hit Rate@5", "hit_rate_at_5"),
        ("Answer Accuracy", "answerable_accuracy"),
        ("Unanswerable Refusal Rate", "unanswerable_refusal_rate"),
        ("Hallucination Count", "hallucination_count"),
    ]

    for label, key in metrics_to_compare:
        base_val = baseline_results.get(key, 0.0)
        imp_val = improved_results.get(key, 0.0)
        if isinstance(base_val, float):
            delta = imp_val - base_val
            sign = "+" if delta > 0 else ""
            print(f"{label:<28} | {base_val:<18.4f} | {imp_val:<20.4f} | {sign}{delta:.4f}")
        else:
            delta = imp_val - base_val
            print(f"{label:<28} | {base_val:<18} | {imp_val:<20} | {delta}")

    print("=" * 60)
    print(f"Results saved to:\n - {baseline_out}\n - {improved_out}\n")


if __name__ == "__main__":
    asyncio.run(main())
