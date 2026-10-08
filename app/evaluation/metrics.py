from typing import List, Dict, Any, Optional
from pydantic import BaseModel


def is_chunk_relevant(retrieved_source: Dict[str, Any], expected_source: Dict[str, Any]) -> bool:
    """Check if a retrieved source matches an expected ground-truth source."""
    exp_pid = expected_source.get("product_id")
    exp_doc = expected_source.get("document")

    ret_pid = retrieved_source.get("product_id")
    ret_doc = retrieved_source.get("document")

    if exp_pid and ret_pid and exp_pid.upper() != ret_pid.upper():
        return False

    if exp_doc and ret_doc:
        # Match filename or stem
        if exp_doc.lower() not in ret_doc.lower() and ret_doc.lower() not in exp_doc.lower():
            return False

    return True


def compute_retrieval_metrics(
    retrieved_sources: List[Dict[str, Any]],
    expected_sources: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute Recall@1, Recall@3, Recall@5, MRR, and HitRate@5.
    """
    if not expected_sources:
        return {
            "recall_at_1": 1.0,
            "recall_at_3": 1.0,
            "recall_at_5": 1.0,
            "mrr": 1.0,
            "hit_rate_at_5": 1.0,
        }

    first_relevant_rank = None
    matched_expected = set()

    k_matches = {1: set(), 3: set(), 5: set()}

    for rank, ret_src in enumerate(retrieved_sources[:5], start=1):
        for idx, exp_src in enumerate(expected_sources):
            if is_chunk_relevant(ret_src, exp_src):
                if first_relevant_rank is None:
                    first_relevant_rank = rank
                if rank <= 1:
                    k_matches[1].add(idx)
                if rank <= 3:
                    k_matches[3].add(idx)
                if rank <= 5:
                    k_matches[5].add(idx)

    num_expected = len(expected_sources)
    recall_1 = len(k_matches[1]) / num_expected
    recall_3 = len(k_matches[3]) / num_expected
    recall_5 = len(k_matches[5]) / num_expected
    mrr = 1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
    hit_rate = 1.0 if len(k_matches[5]) > 0 else 0.0

    return {
        "recall_at_1": round(recall_1, 4),
        "recall_at_3": round(recall_3, 4),
        "recall_at_5": round(recall_5, 4),
        "mrr": round(mrr, 4),
        "hit_rate_at_5": round(hit_rate, 4),
    }


def evaluate_answer_correctness(
    answer: str,
    expected_answer: str,
    answerable: bool,
) -> Dict[str, Any]:
    """
    Automated check of answer grounding and unanswerable refusal.
    """
    ans_lower = answer.lower()
    refusal_phrases = [
        "not available",
        "not documented",
        "no information",
        "not found",
        "cannot be determined",
        "does not mention",
        "not provided",
    ]

    if not answerable:
        is_refusal = any(phrase in ans_lower for phrase in refusal_phrases)
        return {
            "correct": is_refusal,
            "refused": is_refusal,
            "hallucination": not is_refusal,
        }

    # For answerable queries: check if key expected tokens appear in answer
    exp_tokens = [t.lower() for t in expected_answer.split() if len(t) > 2]
    matched_tokens = [t for t in exp_tokens if t in ans_lower]
    token_recall = len(matched_tokens) / len(exp_tokens) if exp_tokens else 1.0

    is_correct = token_recall >= 0.5 or expected_answer.lower() in ans_lower

    return {
        "correct": is_correct,
        "refused": False,
        "hallucination": False,
        "token_overlap": round(token_recall, 4),
    }
