import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.retrieval.retriever import BaselineRetriever
from app.retrieval.hybrid import HybridRetriever
from app.generation.prompt import PromptBuilder
from app.generation.llm import get_llm_client
from app.evaluation.metrics import compute_retrieval_metrics, evaluate_answer_correctness

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Runs automated benchmark evaluation on baseline and improved hybrid retrieval."""

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or Path("data/eval/eval_questions.json")

    def load_dataset(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def run(
        self,
        mode: str = "baseline",
        top_k: int = 5,
        output_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute evaluation suite for a specific retrieval mode (baseline or hybrid).
        """
        questions = self.load_dataset()
        logger.info(f"Loaded {len(questions)} evaluation questions. Running mode '{mode}'...")

        if mode.lower() == "hybrid":
            retriever = HybridRetriever(top_k=top_k)
        else:
            retriever = BaselineRetriever(top_k=top_k)

        llm = get_llm_client()
        prompt_builder = PromptBuilder()

        item_results: List[Dict[str, Any]] = []

        total_r1 = 0.0
        total_r3 = 0.0
        total_r5 = 0.0
        total_mrr = 0.0
        total_hit = 0.0

        answerable_count = 0
        correct_answerable_count = 0

        unanswerable_count = 0
        refused_unanswerable_count = 0
        hallucination_count = 0

        for item in questions:
            q_id = item["id"]
            query = item["question"]
            filters = item.get("filters")
            expected_sources = item.get("expected_sources", [])
            expected_answer = item.get("expected_answer", "")
            is_answerable = item.get("answerable", True)

            # 1. Retrieve
            res = retriever.retrieve(query=query, filters=filters, top_k=top_k)

            retrieved_sources_dicts = [s.model_dump() for s in res.sources]

            # 2. Retrieval Metrics
            ret_metrics = compute_retrieval_metrics(retrieved_sources_dicts, expected_sources)
            total_r1 += ret_metrics["recall_at_1"]
            total_r3 += ret_metrics["recall_at_3"]
            total_r5 += ret_metrics["recall_at_5"]
            total_mrr += ret_metrics["mrr"]
            total_hit += ret_metrics["hit_rate_at_5"]

            # 3. Generation
            if res.has_context:
                messages = prompt_builder.build_messages(query, res.context)
                answer = await llm.generate(messages)
            else:
                answer = "The requested information is not available in the provided knowledge base."

            # 4. Answer Quality Evaluation
            ans_metrics = evaluate_answer_correctness(answer, expected_answer, is_answerable)

            if is_answerable:
                answerable_count += 1
                if ans_metrics["correct"]:
                    correct_answerable_count += 1
            else:
                unanswerable_count += 1
                if ans_metrics["refused"]:
                    refused_unanswerable_count += 1
                if ans_metrics["hallucination"]:
                    hallucination_count += 1

            item_results.append({
                "id": q_id,
                "category": item.get("category"),
                "question": query,
                "answerable": is_answerable,
                "expected_answer": expected_answer,
                "actual_answer": answer,
                "retrieved_sources": retrieved_sources_dicts,
                "retrieval_metrics": ret_metrics,
                "answer_metrics": ans_metrics,
            })

        n_questions = len(questions)
        summary = {
            "mode": mode,
            "total_questions": n_questions,
            "mean_recall_at_1": round(total_r1 / n_questions, 4),
            "mean_recall_at_3": round(total_r3 / n_questions, 4),
            "mean_recall_at_5": round(total_r5 / n_questions, 4),
            "mean_mrr": round(total_mrr / n_questions, 4),
            "hit_rate_at_5": round(total_hit / n_questions, 4),
            "answerable_accuracy": round(correct_answerable_count / answerable_count, 4) if answerable_count else 1.0,
            "unanswerable_refusal_rate": round(refused_unanswerable_count / unanswerable_count, 4) if unanswerable_count else 1.0,
            "hallucination_count": hallucination_count,
            "detailed_results": item_results,
        }

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved evaluation results to {output_file}")

        return summary
