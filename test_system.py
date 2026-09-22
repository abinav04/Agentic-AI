import requests
import json
import sys

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"

# Test function validating system health status and vector store document count.
# Sends HTTP GET request to /api/health and asserts response status 200 and chunk count 154.
def test_health_check():
    print("\n--- 1. Testing GET /api/health Endpoint ---")
    url = f"{BASE_URL}/api/health"
    try:
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print("Health Response:")
        print(json.dumps(data, indent=2))
        assert response.status_code == 200, "Health check failed!"
        assert data.get("chroma_chunks_count") == 154, f"Expected 154 chunks, got {data.get('chroma_chunks_count')}"
        print("[PASS] Health Check Passed!")
        return True
    except Exception as e:
        print(f"[FAIL] Health Check Error: {e}")
        return False

# Test helper function sending POST request queries to the /api/chat endpoint.
# Asserts valid answer payloads, outputs provider/latency stats, and logs execution steps.
def test_chat_query(name: str, question: str, expected_type: str, conversation_id: str = "test_conv", messages: list = None):
    print(f"\n--- Testing {name} ({expected_type}) ---")
    print(f"Question: '{question}'")
    url = f"{BASE_URL}/api/chat"
    payload = {
        "question": question,
        "conversation_id": conversation_id,
        "messages": messages or [],
        "primary_provider": "groq",
        "fallback_enabled": True
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"Error Output: {response.text}")
            return False

        data = response.json()
        print(f"Provider Used: {data.get('llm_provider').upper()} (Model: {data.get('llm_model')})")
        print(f"Fallback Occurred: {data.get('fallback_occurred')}")
        print(f"Verifier Verdict: {data.get('verifier_verdict')}")
        print(f"Latency: {data.get('latency_seconds')}s")
        print("Agent Steps Executed:")
        # Loop over returned agent step descriptions to print execution trace checkmarks.
        # Encodes unicode step symbols safely for Windows terminal output compatibility.
        for step in data.get("agent_steps", []):
            safe_step = step.encode('ascii', errors='replace').decode('ascii')
            print(f"  {safe_step}")

        print("\nFinal Answer Preview:")
        ans_preview = data.get("final_answer", "")[:300].encode('ascii', errors='replace').decode('ascii')
        print(ans_preview + ("..." if len(data.get("final_answer", "")) > 300 else ""))

        print("\nCitations:")
        cit_preview = data.get("formatted_citations", "").encode('ascii', errors='replace').decode('ascii')
        print(cit_preview)

        # Assertions
        assert data.get("final_answer"), "Final answer should not be empty!"
        print(f"[PASS] {name} Passed!")
        return data
    except Exception as e:
        print(f"[FAIL] {name} Error: {e}")
        return False

# Main test runner executing health checks, single-hop, multi-hop, conflicting, and follow-up test cases.
# Prints comprehensive pass/fail summary results for backend API verification.
def run_all_tests():
    print("============================================================")
    print("      Kestrel Multi-Agent Research Assistant Test Suite     ")
    print("============================================================")

    # 1. Health Check
    health_ok = test_health_check()

    # 2. Single-Hop Question
    q1_data = test_chat_query(
        name="Single-Hop Query",
        question="When was the Beacon issue fixed?",
        expected_type="single_hop",
        conversation_id="conv_test_1"
    )

    # 3. Multi-Hop Question
    q2_data = test_chat_query(
        name="Multi-Hop Query",
        question="When was the Beacon issue fixed and who gets the feature?",
        expected_type="multi_hop",
        conversation_id="conv_test_2"
    )

    # 4. Conflicting Evidence Question
    q3_data = test_chat_query(
        name="Conflicting Evidence Query",
        question="What is the Starter plan ingest rate limit in requests per second?",
        expected_type="conflicting",
        conversation_id="conv_test_3"
    )

    # 5. Unsupported Question
    q4_data = test_chat_query(
        name="Unsupported Query",
        question="Does Kestrel support blockchain payments?",
        expected_type="unsupported",
        conversation_id="conv_test_4"
    )

    # 6. Multi-Turn Follow-Up Question
    print("\n--- Testing Multi-Turn Follow-Up Query ---")
    turn1_data = test_chat_query(
        name="Multi-Turn Turn 1",
        question="What is the Beacon feature?",
        expected_type="single_hop",
        conversation_id="conv_test_multiturn"
    )

    history = [
        {"role": "user", "content": "What is the Beacon feature?"},
        {"role": "assistant", "content": turn1_data.get("final_answer", "")}
    ] if turn1_data else []

    turn2_data = test_chat_query(
        name="Multi-Turn Turn 2 (Follow-Up)",
        question="How many of them can I create on the Growth plan?",
        expected_type="follow_up",
        conversation_id="conv_test_multiturn",
        messages=history
    )

    print("\n============================================================")
    print("                    TEST SUMMARY RESULTS                    ")
    print("============================================================")
    print(f"Health Check              : {'PASSED' if health_ok else 'FAILED'}")
    print(f"Single-Hop Query          : {'PASSED' if q1_data else 'FAILED'}")
    print(f"Multi-Hop Query           : {'PASSED' if q2_data else 'FAILED'}")
    print(f"Conflicting Evidence Query: {'PASSED' if q3_data else 'FAILED'}")
    print(f"Unsupported Query         : {'PASSED' if q4_data else 'FAILED'}")
    print(f"Multi-Turn Follow-Up      : {'PASSED' if turn2_data else 'FAILED'}")
    print("============================================================")

if __name__ == "__main__":
    run_all_tests()
