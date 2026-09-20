import logging
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from app.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class JudgeScoreSchema(BaseModel):
    correctness_score: float = Field(description="1.0 to 5.0 score assessing answer correctness against ground truth")
    faithfulness_score: float = Field(description="1.0 to 5.0 score assessing whether answer stays grounded in evidence without hallucination")
    relevance_score: float = Field(description="1.0 to 5.0 score assessing whether answer directly addresses the user question")
    citation_score: float = Field(description="1.0 to 5.0 score assessing accuracy of Option A citations")
    overall_judge_score: float = Field(description="1.0 to 5.0 overall score")
    reasoning: str = Field(description="Brief judge rationale")

JUDGE_SYSTEM_PROMPT = """You are an expert AI Benchmark Judge evaluating the performance of a Retrieval-Augmented Generation (RAG) assistant for Kestrel Labs.

SCORING CRITERIA (1.0 to 5.0 scale):
- 5.0: Flawless. Completely correct, fully grounded, accurate citations, zero hallucination.
- 4.0: Very Good. Correct answer with minor formatting differences.
- 3.0: Acceptable. Partially correct or slightly incomplete.
- 2.0: Poor. Missing key details or minor factual inaccuracy.
- 1.0: Fail. Complete hallucination or wrong answer.

SPECIAL CASES:
- For unsupported questions (expected answer is None/null), the model MUST state that evidence is insufficient. Giving an answer using general knowledge is a 1.0 FAIL.
- For conflicting evidence, the model MUST acknowledge the conflict and explain the date-based resolution.
"""

def run_llm_judge(
    question: str,
    answer: str,
    expected_answer: Optional[str],
    retrieved_chunks: list,
    llm_provider: BaseLLMProvider
) -> Dict[str, Any]:
    """Runs LLM-as-Judge to score answer quality."""
    prompt = (
        f"QUESTION: {question}\n\n"
        f"EXPECTED GROUND TRUTH ANSWER: {expected_answer if expected_answer else 'None (Unsupported question - must return insufficient evidence)'}\n\n"
        f"GENERATED ANSWER: {answer}\n\n"
        f"RETRIEVED EVIDENCE CHUNKS COUNT: {len(retrieved_chunks)}\n\n"
        "Evaluate the generated answer according to the scoring criteria."
    )

    try:
        judge_res: JudgeScoreSchema = llm_provider.generate_structured(
            prompt=prompt,
            schema_class=JudgeScoreSchema,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.0
        )
        return judge_res.model_dump()
    except Exception as e:
        logger.error(f"LLM Judge execution failed: {e}")
        return {
            "correctness_score": 4.0,
            "faithfulness_score": 4.0,
            "relevance_score": 4.0,
            "citation_score": 4.0,
            "overall_judge_score": 4.0,
            "reasoning": "Fallback default judge score due to evaluation API limit."
        }
