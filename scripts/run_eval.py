import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import setup_logging
from app.observability import setup_observability
from app.evaluation.langsmith_eval import ExperimentRunner, sync_eval_dataset_to_langsmith

logger = logging.getLogger("scripts.run_eval")


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG benchmark evaluation and experiment tracking.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "baseline", "hybrid"],
        default="both",
        help="Retrieval mode to evaluate: 'both' (runs baseline and hybrid comparison), 'baseline', or 'hybrid'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks (top-k) to pass to LLM (default: 5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Score threshold for retrieval filtering (default: from config).",
    )
    parser.add_argument(
        "--push-langsmith",
        action="store_true",
        help="Upload evaluation dataset and experiment run to LangSmith platform if configured.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment name to attach to this evaluation run.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    setup_logging()
    setup_observability()

    data_eval_dir = Path("data/eval").resolve()
    data_eval_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_eval_dir / "eval_questions.json"
    if not dataset_path.exists():
        print(f"Error: Evaluation dataset not found at {dataset_path}")
        sys.exit(1)

    runner = ExperimentRunner(dataset_path=dataset_path)

    if args.push_langsmith:
        print("\nSyncing evaluation dataset to LangSmith...")
        sync_eval_dataset_to_langsmith(dataset_path)

    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK EVALUATION & EXPERIMENT TRACKING")
    print(f"Settings: top_k={args.top_k}, threshold={args.threshold or 'default'}")
    print("=" * 60)

    if args.mode == "both":
        # 1. Run Baseline
        print("\n[1/2] Evaluating RETRIEVAL_MODE=baseline (Vector-only)...")
        baseline_out = data_eval_dir / "results_baseline.json"
        baseline_results = await runner.run_experiment(
            retrieval_mode="baseline",
            top_k=args.top_k,
            score_threshold=args.threshold,
            experiment_name="baseline-vector-top5" if not args.experiment_name else f"{args.experiment_name}-baseline",
            push_langsmith=args.push_langsmith,
            output_file=baseline_out,
        )

        # 2. Run Hybrid
        print("\n[2/2] Evaluating RETRIEVAL_MODE=hybrid (BM25 + Vector + RRF)...")
        improved_out = data_eval_dir / "results_improved.json"
        improved_results = await runner.run_experiment(
            retrieval_mode="hybrid",
            top_k=args.top_k,
            score_threshold=args.threshold,
            experiment_name="hybrid-rrf-top5" if not args.experiment_name else f"{args.experiment_name}-hybrid",
            push_langsmith=args.push_langsmith,
            output_file=improved_out,
        )

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

    else:
        # Run single mode
        out_file = data_eval_dir / f"results_{args.mode}.json"
        print(f"\nEvaluating RETRIEVAL_MODE={args.mode}...")
        results = await runner.run_experiment(
            retrieval_mode=args.mode,
            top_k=args.top_k,
            score_threshold=args.threshold,
            experiment_name=args.experiment_name or f"{args.mode}-k{args.top_k}",
            push_langsmith=args.push_langsmith,
            output_file=out_file,
        )

        print("\n" + "=" * 60)
        print(f"EXPERIMENT RESULTS ({args.mode.upper()})")
        print("=" * 60)
        print(f"Mean Recall@1:             {results['mean_recall_at_1']}")
        print(f"Mean Recall@3:             {results['mean_recall_at_3']}")
        print(f"Mean Recall@5:             {results['mean_recall_at_5']}")
        print(f"Mean Reciprocal Rank:      {results['mean_mrr']}")
        print(f"Hit Rate@5:                {results['hit_rate_at_5']}")
        print(f"Answer Accuracy:           {results['answerable_accuracy']}")
        print(f"Unanswerable Refusal Rate: {results['unanswerable_refusal_rate']}")
        print(f"Hallucination Count:       {results['hallucination_count']}")
        print("=" * 60)
        print(f"Results saved to: {out_file}\n")


if __name__ == "__main__":
    asyncio.run(main())
