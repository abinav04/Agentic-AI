from fastapi import APIRouter
import json
import os

router = APIRouter()

@router.get("/evaluate/results")
def get_evaluation_results():
    """Retrieves previous evaluation benchmark results if present."""
    summary_path = "results/metrics_summary.json"
    if not os.path.exists(summary_path):
        return {"status": "not_run", "message": "Evaluation summary file not found. Run python -m evaluation.run_eval to generate."}

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"status": "available", "metrics_summary": data}
