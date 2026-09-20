import json
import time
import os
import logging
from typing import List, Dict, Any
from app.graph.workflow import KestrelResearchAssistantPipeline
from app.evaluation.metrics import (
    compute_retrieval_recall_at_k,
    compute_citation_precision,
    compute_heuristic_faithfulness,
    compute_heuristic_relevance,
    compute_e2e_correctness
)
from app.evaluation.judge import run_llm_judge
from app.retrieval.ingestion import run_ingestion

logger = logging.getLogger(__name__)

def run_evaluation_suite(
    questions_file: str = "results/eval_questions.jsonl",
    results_file: str = "results/eval_results.jsonl",
    summary_file: str = "results/metrics_summary.json",
    primary_provider: str = "groq"
) -> Dict[str, Any]:
    """Executes evaluation suite over all questions and generates result files."""

    # Ensure ingestion has run
    run_ingestion()

    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    pipeline = KestrelResearchAssistantPipeline(primary_provider=primary_provider)

    questions = []
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line.strip()))

    logger.info(f"Starting evaluation run over {len(questions)} test questions...")

    eval_results = []
    start_wall_clock = time.time()
    total_latency = 0.0

    # Type breakdowns
    type_metrics: Dict[str, Dict[str, List[float]]] = {}

    for idx, q_data in enumerate(questions, 1):
        q_id = q_data["question_id"]
        question = q_data["question"]
        q_type = q_data["type"]
        exp_answer = q_data.get("expected_answer")
        exp_chunk_ids = q_data.get("expected_chunk_ids", [])

        logger.info(f"[{idx}/{len(questions)}] Evaluating Question ({q_type}): '{question}'")

        # Handle multi-turn context
        messages = []
        if q_type == "follow_up":
            # Add synthetic prior turn for context
            if "beacons" in q_id:
                messages = [
                    {"role": "user", "content": "What is the Beacon feature?"},
                    {"role": "assistant", "content": "Beacons are alerting rules attached to metrics."}
                ]
            elif "cohort" in q_id:
                messages = [
                    {"role": "user", "content": "What is a behavioural cohort?"},
                    {"role": "assistant", "content": "A behavioural cohort is a rule-based set of users recomputed hourly."}
                ]
            elif "trails" in q_id:
                messages = [
                    {"role": "user", "content": "What are Trails in Kestrel?"},
                    {"role": "assistant", "content": "Trails is a per-user chronological event timeline."}
                ]

        state = pipeline.run(
            question=question,
            conversation_id=q_data.get("conversation_id", f"eval_{q_id}"),
            messages=messages
        )

        final_answer = state["final_answer"]
        retrieved_chunk_ids = state["retrieved_chunk_ids"]
        cited_ids = [c["chunk_id"] for c in state.get("citations", [])]
        verdict = state["verifier_verdict"]
        latency = state["latency_seconds"]
        total_latency += latency

        # Compute Metrics
        recall = compute_retrieval_recall_at_k(retrieved_chunk_ids, exp_chunk_ids)
        precision = compute_citation_precision(cited_ids, exp_chunk_ids)
        faithfulness = compute_heuristic_faithfulness(final_answer, verdict, state["retrieved_chunks"])
        relevance = compute_heuristic_relevance(question, final_answer)
        e2e_correctness = compute_e2e_correctness(recall, precision, faithfulness, relevance)

        # Run LLM-as-Judge
        judge_res = run_llm_judge(
            question=question,
            answer=final_answer,
            expected_answer=exp_answer,
            retrieved_chunks=state["retrieved_chunks"],
            llm_provider=pipeline.provider
        )

        scores = {
            "retrieval_recall": recall,
            "citation_precision": precision,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "e2e_correctness": e2e_correctness,
            "llm_judge_score": judge_res["overall_judge_score"],
            "judge_breakdown": judge_res
        }

        # Track type breakdown
        if q_type not in type_metrics:
            type_metrics[q_type] = {"recall": [], "precision": [], "correctness": [], "judge": []}
        type_metrics[q_type]["recall"].append(recall)
        type_metrics[q_type]["precision"].append(precision)
        type_metrics[q_type]["correctness"].append(e2e_correctness)
        type_metrics[q_type]["judge"].append(judge_res["overall_judge_score"])

        langsmith_url = f"https://smith.langchain.com/o/project/{pipeline.provider.provider_name}/{state['conversation_id']}"

        result_entry = {
            "question_id": q_id,
            "question": question,
            "type": q_type,
            "answer": final_answer,
            "citations": cited_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "verifier_verdict": verdict,
            "scores": scores,
            "latency_seconds": latency,
            "langsmith_run_url": langsmith_url
        }

        eval_results.append(result_entry)

    wall_clock_time = round(time.time() - start_wall_clock, 2)

    # Save eval_results.jsonl
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        for res in eval_results:
            f.write(json.dumps(res) + "\n")

    # Aggregate Scores
    avg_recall = round(sum(r["scores"]["retrieval_recall"] for r in eval_results) / len(eval_results), 4)
    avg_precision = round(sum(r["scores"]["citation_precision"] for r in eval_results) / len(eval_results), 4)
    avg_faithfulness = round(sum(r["scores"]["faithfulness"] for r in eval_results) / len(eval_results), 4)
    avg_relevance = round(sum(r["scores"]["relevance"] for r in eval_results) / len(eval_results), 4)
    avg_correctness = round(sum(r["scores"]["e2e_correctness"] for r in eval_results) / len(eval_results), 4)
    avg_judge = round(sum(r["scores"]["llm_judge_score"] for r in eval_results) / len(eval_results), 4)

    breakdown_by_type = {}
    for t, m in type_metrics.items():
        breakdown_by_type[t] = {
            "count": len(m["recall"]),
            "avg_recall": round(sum(m["recall"]) / len(m["recall"]), 4),
            "avg_precision": round(sum(m["precision"]) / len(m["precision"]), 4),
            "avg_correctness": round(sum(m["correctness"]) / len(m["correctness"]), 4),
            "avg_llm_judge": round(sum(m["judge"]) / len(m["judge"]), 4)
        }

    summary = {
        "total_questions": len(eval_results),
        "aggregate_scores": {
            "retrieval_recall_at_k": avg_recall,
            "citation_precision": avg_precision,
            "answer_faithfulness": avg_faithfulness,
            "answer_relevance": avg_relevance,
            "end_to_end_correctness": avg_correctness,
            "llm_as_judge_score": avg_judge
        },
        "breakdown_by_question_type": breakdown_by_type,
        "performance": {
            "total_wall_clock_seconds": wall_clock_time,
            "average_latency_per_question_seconds": round(total_latency / len(eval_results), 2),
            "estimated_token_usage": len(eval_results) * 1250
        },
        "models_used": {
            "generation_model": pipeline.provider.model_name,
            "provider": pipeline.provider.last_provider_used,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Evaluation complete. Summary written to {summary_file}")
    return summary
