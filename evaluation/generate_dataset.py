import json
import os

EVAL_QUESTIONS = [
    # 5 Single-hop questions
    {
        "question_id": "q1_single_beacon_destinations",
        "question": "How many destinations can a single Beacon notify at once?",
        "type": "single_hop",
        "conversation_id": "eval_single_1",
        "turn": 1,
        "expected_answer": "A Beacon can notify up to four destinations at once: Slack, PagerDuty, generic webhook, and email.",
        "expected_chunk_ids": ["spec-beacons:3"]
    },
    {
        "question_id": "q2_single_ingest_batch_size",
        "question": "What is the maximum number of events allowed in a single ingestion batch?",
        "type": "single_hop",
        "conversation_id": "eval_single_2",
        "turn": 1,
        "expected_answer": "A batch may contain at most 500 events or 2 MB.",
        "expected_chunk_ids": ["spec-ingest-api:0", "spec-ingest-api:1"]
    },
    {
        "question_id": "q3_single_kql_row_limit",
        "question": "What is the maximum row limit returned by a single KQL query?",
        "type": "single_hop",
        "conversation_id": "eval_single_3",
        "turn": 1,
        "expected_answer": "A KQL query returns at most 10,000 rows.",
        "expected_chunk_ids": ["spec-sdks-kql:4"]
    },
    {
        "question_id": "q4_single_trails_sampling",
        "question": "What is the Trails sampling rate for Growth plan projects?",
        "type": "single_hop",
        "conversation_id": "eval_single_4",
        "turn": 1,
        "expected_answer": "On Growth plans, Trails are sampled at 25 percent of users.",
        "expected_chunk_ids": ["spec-trails:2", "spec-trails:3"]
    },
    {
        "question_id": "q5_single_static_cohort_limit",
        "question": "What is the maximum number of user IDs accepted in a single static cohort CSV upload?",
        "type": "single_hop",
        "conversation_id": "eval_single_5",
        "turn": 1,
        "expected_answer": "A single upload may contain at most 250,000 user IDs.",
        "expected_chunk_ids": ["spec-funnels-cohorts:4", "rn-3-4:0"]
    },

    # 5 Multi-hop questions
    {
        "question_id": "q6_multi_beacon_fix_and_plan",
        "question": "When was the Beacon scheduler clock skew issue fixed and how many Beacons can Growth plan projects create?",
        "type": "multi_hop",
        "conversation_id": "eval_multi_1",
        "turn": 1,
        "expected_answer": "The Beacon scheduler issue was fixed in release 4.1.1, and Growth plan projects can create up to 60 Beacons.",
        "expected_chunk_ids": ["spec-beacons:1", "spec-beacons:4"]
    },
    {
        "question_id": "q7_multi_ingest_response_and_rate_limits",
        "question": "What HTTP status code is returned on successful event ingestion and what are the rate limits for Starter vs Scale plans?",
        "type": "multi_hop",
        "conversation_id": "eval_multi_2",
        "turn": 1,
        "expected_answer": "Successful ingest returns HTTP 202 Accepted. Starter projects are limited to 350 requests per second while Scale projects allow 7,500 requests per second.",
        "expected_chunk_ids": ["spec-ingest-api:1", "spec-ingest-api:3"]
    },
    {
        "question_id": "q8_multi_funnel_breakdown_and_ordering",
        "question": "How many breakdown properties does a funnel support and what are the two ordering modes available?",
        "type": "multi_hop",
        "conversation_id": "eval_multi_3",
        "turn": 1,
        "expected_answer": "Funnels support up to 3 breakdown properties at once, and can be evaluated in strict ordering or loose ordering modes.",
        "expected_chunk_ids": ["spec-funnels-cohorts:1", "spec-funnels-cohorts:2"]
    },
    {
        "question_id": "q9_multi_warehouse_destinations_and_idempotency",
        "question": "Which data warehouse destinations are supported on Scale plans and how is idempotency enforced for warehouse sync?",
        "type": "multi_hop",
        "conversation_id": "eval_multi_4",
        "turn": 1,
        "expected_answer": "Scale projects can sync to Snowflake, BigQuery, and Amazon Redshift. Every synced row carries an idempotency key composed of (sync_id, row_hash).",
        "expected_chunk_ids": ["spec-warehouse-sync:0", "spec-warehouse-sync:2"]
    },
    {
        "question_id": "q10_multi_trails_moments_and_retention",
        "question": "What three moment types are detected in Trails and what is the Trail retention period on Growth vs Scale plans?",
        "type": "multi_hop",
        "conversation_id": "eval_multi_5",
        "turn": 1,
        "expected_answer": "The three moment types are Rage clicks, Dead ends, and Error bursts. Trail retention is 30 days on Growth and 90 days on Scale.",
        "expected_chunk_ids": ["spec-trails:2", "spec-trails:3"]
    },

    # 4 Conflicting evidence questions
    {
        "question_id": "q11_conflict_starter_rate_limit",
        "question": "What is the Starter plan ingest rate limit in requests per second?",
        "type": "conflicting",
        "conversation_id": "eval_conflict_1",
        "turn": 1,
        "expected_answer": "In release 3.4 notes the limit was 200 rps, but as of release 3.5 / 4.0 specification (published 2025-10-07), the Starter rate limit was raised to 350 requests per second.",
        "expected_chunk_ids": ["rn-3-4:4", "spec-ingest-api:3"]
    },
    {
        "question_id": "q12_conflict_funnel_breakdown_count",
        "question": "How many breakdown properties can be used in a funnel report?",
        "type": "conflicting",
        "conversation_id": "eval_conflict_2",
        "turn": 1,
        "expected_answer": "Before release 3.4 funnels supported 2 breakdown properties, but starting in release 3.4 and 4.1 specification, the limit was raised to 3 properties.",
        "expected_chunk_ids": ["spec-funnels-cohorts:2", "rn-3-4:1"]
    },
    {
        "question_id": "q13_conflict_kql_timeout",
        "question": "What is the timeout duration for a KQL query?",
        "type": "conflicting",
        "conversation_id": "eval_conflict_3",
        "turn": 1,
        "expected_answer": "The query timeout was 30 seconds previously, but was raised to 60 seconds in release 4.1 following the introduction of the Osprey engine.",
        "expected_chunk_ids": ["spec-sdks-kql:4"]
    },
    {
        "question_id": "q14_conflict_warehouse_dedup_scheme",
        "question": "How does Warehouse Sync deduplicate rows sent to customer data warehouses?",
        "type": "conflicting",
        "conversation_id": "eval_conflict_4",
        "turn": 1,
        "expected_answer": "Warehouse Sync previously used position-based deduplication which caused duplicate rows during INC-2025-11. In release 4.0.3, it was replaced by idempotency keys using (sync_id, row_hash).",
        "expected_chunk_ids": ["spec-warehouse-sync:2"]
    },

    # 3 Unsupported questions
    {
        "question_id": "q15_unsupported_blockchain",
        "question": "Does Kestrel support blockchain payments?",
        "type": "unsupported",
        "conversation_id": "eval_unsupported_1",
        "turn": 1,
        "expected_answer": None,
        "expected_chunk_ids": []
    },
    {
        "question_id": "q16_unsupported_kubernetes",
        "question": "How do I deploy the Kestrel ingestion pipeline onto an on-premise Kubernetes cluster?",
        "type": "unsupported",
        "conversation_id": "eval_unsupported_2",
        "turn": 1,
        "expected_answer": None,
        "expected_chunk_ids": []
    },
    {
        "question_id": "q17_unsupported_vector_search",
        "question": "Does Kestrel offer native pinecone vector database indexing for raw event payloads?",
        "type": "unsupported",
        "conversation_id": "eval_unsupported_3",
        "turn": 1,
        "expected_answer": None,
        "expected_chunk_ids": []
    },

    # 3 Follow-up questions
    {
        "question_id": "q18_followup_beacons_count",
        "question": "How many of them can I create on the Growth plan?",
        "type": "follow_up",
        "conversation_id": "eval_followup_1",
        "turn": 2,
        "expected_answer": "Growth projects may create up to 60 Beacons.",
        "expected_chunk_ids": ["spec-beacons:4"]
    },
    {
        "question_id": "q19_followup_cohort_refresh",
        "question": "How often is it recomputed?",
        "type": "follow_up",
        "conversation_id": "eval_followup_2",
        "turn": 2,
        "expected_answer": "Behavioural cohorts are recomputed every hour.",
        "expected_chunk_ids": ["spec-funnels-cohorts:3"]
    },
    {
        "question_id": "q20_followup_trails_moments",
        "question": "What is the threshold for a rage click moment?",
        "type": "follow_up",
        "conversation_id": "eval_followup_3",
        "turn": 2,
        "expected_answer": "A rage click moment is detected when there are five or more click events on the same element within 3 seconds.",
        "expected_chunk_ids": ["spec-trails:2"]
    }
]

# Function to generate the JSONL benchmark test dataset file containing 20 curated test cases.
# Writes single-hop, multi-hop, conflicting, unsupported, and follow-up query objects to output path.
def generate_questions_file(output_path: str = "results/eval_questions.jsonl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Loop through static EVAL_QUESTIONS dataset list to write each benchmark test case line by line.
        # Outputs formatted JSON records for evaluation suite execution.
        for q in EVAL_QUESTIONS:
            f.write(json.dumps(q) + "\n")
    print(f"Successfully generated {len(EVAL_QUESTIONS)} evaluation questions in {output_path}")

if __name__ == "__main__":
    generate_questions_file()
