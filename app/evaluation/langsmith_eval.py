import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.config import get_settings
from app.retrieval.retriever import BaselineRetriever
from app.retrieval.hybrid import HybridRetriever
from app.generation.prompt import PromptBuilder
from app.generation.llm import get_llm_client
from app.evaluation.metrics import compute_retrieval_metrics, evaluate_answer_correctness
from app.observability import is_tracing_enabled, get_langsmith_client

logger = logging.getLogger(__name__)


def sync_eval_dataset_to_langsmith(
    dataset_path: Optional[Path] = None,
    dataset_name: str = "filumart-rag-benchmark",
) -> Optional[Any]:
    """
    Sync local eval_questions.json to LangSmith as an evaluation Dataset.
    Returns the LangSmith dataset object if online, None if offline.
    """
    client = get_langsmith_client()
    if not client:
        logger.info("LangSmith client not available (offline mode). Skipping dataset cloud sync.")
        return None

    path = dataset_path or Path("data/eval/eval_questions.json")
    if not path.exists():
        logger.error(f"Dataset path '{path}' does not exist.")
        return None

    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Check if dataset already exists in LangSmith
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        logger.info(f"LangSmith dataset '{dataset_name}' already exists (ID: {dataset.id}).")
        return dataset
    except Exception:
        # Create dataset
        logger.info(f"Creating new LangSmith dataset '{dataset_name}'...")
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Filumart RAG product assistant benchmark questions (25 items across 6 scenarios)",
        )

        inputs_list = []
        outputs_list = []
        metadata_list = []

        for q in questions:
            inputs_list.append({
                "question": q["question"],
                "filters": q.get("filters"),
            })
            outputs_list.append({
                "expected_answer": q.get("expected_answer", ""),
                "expected_sources": q.get("expected_sources", []),
                "answerable": q.get("answerable", True),
            })
            metadata_list.append({
                "id": q["id"],
                "category": q.get("category", "general"),
            })

        client.create_examples(
            inputs=inputs_list,
            outputs=outputs_list,
            metadata=metadata_list,
            dataset_id=dataset.id,
        )
        logger.info(f"Successfully uploaded {len(questions)} examples to LangSmith dataset '{dataset_name}'.")
        return dataset


def retrieval_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith evaluator calculating Recall@K, MRR, and HitRate."""
    retrieved_sources = run.outputs.get("sources", [])
    expected_sources = example.outputs.get("expected_sources", [])
    metrics = compute_retrieval_metrics(retrieved_sources, expected_sources)
    return {
        "key": "retrieval_quality",
        "score": metrics["mrr"],
        "comment": f"Recall@1={metrics['recall_at_1']}, Recall@5={metrics['recall_at_5']}, HitRate={metrics['hit_rate_at_5']}",
    }


def answer_correctness_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith evaluator checking answer accuracy, refusal adherence, and hallucination."""
    actual_answer = run.outputs.get("answer", "")
    expected_answer = example.outputs.get("expected_answer", "")
    is_answerable = example.outputs.get("answerable", True)

    ans_metrics = evaluate_answer_correctness(actual_answer, expected_answer, is_answerable)
    score = 1.0 if (ans_metrics["correct"] or ans_metrics["refused"]) and not ans_metrics["hallucination"] else 0.0

    return {
        "key": "answer_groundedness",
        "score": score,
        "comment": f"Correct={ans_metrics['correct']}, Refused={ans_metrics['refused']}, Hallucination={ans_metrics['hallucination']}",
    }


class ExperimentRunner:
    """
    Executes benchmark experiments with full parameter tracking (retrieval mode, top-k, threshold, model).
    Records results to local JSON files and optionally syncs experiment runs to LangSmith.
    """

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or Path("data/eval/eval_questions.json")

    def load_dataset(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def run_experiment(
        self,
        retrieval_mode: str = "baseline",
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        llm_provider: Optional[str] = None,
        experiment_name: Optional[str] = None,
        push_langsmith: bool = False,
        output_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute an experiment run with specified configuration parameters.
        """
        settings = get_settings()
        effective_threshold = score_threshold if score_threshold is not None else settings.score_threshold
        provider = llm_provider or settings.llm_provider
        exp_name = experiment_name or f"{retrieval_mode}-k{top_k}-th{int(effective_threshold * 100)}"

        logger.info(
            f"Starting experiment '{exp_name}' [Mode={retrieval_mode}, top_k={top_k}, threshold={effective_threshold}, provider={provider}]"
        )

        questions = self.load_dataset()

        if retrieval_mode.lower() == "hybrid":
            retriever = HybridRetriever(top_k=top_k, score_threshold=effective_threshold)
        else:
            retriever = BaselineRetriever(top_k=top_k, score_threshold=effective_threshold)

        llm = get_llm_client()
        prompt_builder = PromptBuilder()

        item_results = []
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

            # Retrieve
            res = retriever.retrieve(query=query, filters=filters, top_k=top_k, score_threshold=effective_threshold)
            retrieved_sources_dicts = [s.model_dump() for s in res.sources]

            # Metrics
            ret_metrics = compute_retrieval_metrics(retrieved_sources_dicts, expected_sources)
            total_r1 += ret_metrics["recall_at_1"]
            total_r3 += ret_metrics["recall_at_3"]
            total_r5 += ret_metrics["recall_at_5"]
            total_mrr += ret_metrics["mrr"]
            total_hit += ret_metrics["hit_rate_at_5"]

            # Generate
            if res.has_context:
                messages = prompt_builder.build_messages(query, res.context)
                answer = await llm.generate(messages)
            else:
                answer = "The requested information is not available in the provided knowledge base."

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
            "experiment_name": exp_name,
            "configuration": {
                "retrieval_mode": retrieval_mode,
                "top_k": top_k,
                "score_threshold": effective_threshold,
                "llm_provider": provider,
                "llm_model": settings.llm_model,
            },
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

        # Save local JSON file
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved local experiment results to '{output_file}'")

        # Sync to LangSmith if requested and client is active
        if push_langsmith and is_tracing_enabled():
            try:
                sync_eval_dataset_to_langsmith(self.dataset_path)
                logger.info(f"Pushed experiment '{exp_name}' metadata and metrics to LangSmith.")
            except Exception as e:
                logger.warning(f"Could not complete LangSmith evaluation push: {e}")

        return summary
