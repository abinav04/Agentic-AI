import logging
import sys
import argparse
from app.evaluation.runner import run_evaluation_suite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Command-line entry point for running the complete benchmark evaluation suite.
# Parses provider arguments, invokes run_evaluation_suite, and prints benchmark score reports.
def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Suite for Kestrel Multi-Agent Research Assistant")
    parser.add_argument("--provider", type=str, default="groq", choices=["groq", "gemini"], help="Primary LLM provider")
    args = parser.parse_args()

    print("============================================================")
    print("      Kestrel Multi-Agent Research Assistant Evaluation     ")
    print("============================================================")

    try:
        summary = run_evaluation_suite(primary_provider=args.provider)
        print("\n--- Benchmark Execution Summary ---")
        print(f"Total Questions Evaluated : {summary['total_questions']}")
        print(f"Retrieval Recall@K        : {summary['aggregate_scores']['retrieval_recall_at_k']}")
        print(f"Citation Precision        : {summary['aggregate_scores']['citation_precision']}")
        print(f"Answer Faithfulness       : {summary['aggregate_scores']['answer_faithfulness']}")
        print(f"Answer Relevance          : {summary['aggregate_scores']['answer_relevance']}")
        print(f"End-to-End Correctness    : {summary['aggregate_scores']['end_to_end_correctness']}")
        print(f"LLM-as-Judge Score (1-5)  : {summary['aggregate_scores']['llm_as_judge_score']}")
        print(f"Total Wall Clock Time     : {summary['performance']['total_wall_clock_seconds']}s")
        print("============================================================")
        print("Results generated:")
        print("  - results/eval_questions.jsonl")
        print("  - results/eval_results.jsonl")
        print("  - results/metrics_summary.json")
    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
