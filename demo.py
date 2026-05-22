import argparse
import json
from pathlib import Path

from app.main import ConversationStateManager


SEMANTIC_EQUIVALENTS = {
    "issue": {
        "billing": {"billing", "double_charge", "duplicate_billing", "refund", "payment_issue"},
        "api_support": {"api_support", "api_integration", "webhook_timeout", "webhook_delivery_timeout"},
    },
    "intent": {
        "subscription_cancellation": {
            "subscription_cancellation",
            "cancel_subscription",
            "cancellation",
            "cancel_subscription_request",
        },
    },
}


def values_match(field, actual_value, expected_value):
    if actual_value == expected_value:
        return True

    if isinstance(actual_value, str) and isinstance(expected_value, str):
        actual = actual_value.lower().replace(" ", "_").replace("-", "_")
        expected = expected_value.lower().replace(" ", "_").replace("-", "_")
        equivalents = SEMANTIC_EQUIVALENTS.get(field, {}).get(expected, set())
        return actual in equivalents

    return False


def memory_matches(actual, expected):
    return all(values_match(key, actual.get(key), value) for key, value in expected.items())


def expected_memory_field_report(actual, expected):
    report = {}
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        report[key] = {
            "expected": expected_value,
            "actual": actual_value,
            "matched": values_match(key, actual_value, expected_value),
        }
    return report


def expected_fields_preserved(report):
    total = len(report)
    matched = sum(1 for item in report.values() if item["matched"])
    return matched, total


def main():
    parser = argparse.ArgumentParser(description="Conversation State Manager demo/evaluation")
    parser.add_argument("--samples", default="sample_conversations.json")
    parser.add_argument("--threshold", type=int, default=250)
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument(
        "--no-llm-summary",
        action="store_true",
        help="Disable OpenRouter summarization even if OPENROUTER_API_KEY is set.",
    )
    parser.add_argument(
        "--no-llm-memory",
        action="store_true",
        help="Disable OpenRouter memory extraction even if OPENROUTER_API_KEY is set.",
    )
    args = parser.parse_args()

    manager = ConversationStateManager(
        token_threshold=args.threshold,
        recent_window=args.recent_window,
        use_llm_summary=not args.no_llm_summary,
        use_llm_memory=not args.no_llm_memory,
    )
    samples = json.loads(Path(args.samples).read_text(encoding="utf-8"))

    results = []
    for sample in samples:
        state = manager.process_conversation(sample["conversation_id"], sample["messages"])
        expected = sample.get("expected_memory", {})
        field_report = expected_memory_field_report(state["memory"], expected)
        matched_fields, total_fields = expected_fields_preserved(field_report)
        result = {
            "conversation_id": state["conversation_id"],
            "active_topic": state["active_topic"],
            "summarization_triggered": state["summarization_triggered"],
            "summary_provider": state["summary_provider"],
            "summary_fallback_reason": state["summary_fallback_reason"],
            "memory_provider": state["memory_provider"],
            "memory_fallback_reason": state["memory_fallback_reason"],
            "recent_message_count": len(state["recent_messages"]),
            "token_before": state["token_before"],
            "token_after": state["token_after"],
            "token_reduction_percent": state["token_reduction_percent"],
            "expected_memory_fields_preserved": f"{matched_fields}/{total_fields}",
            "memory_field_report": field_report,
            "memory_matched_expected": memory_matches(state["memory"], expected),
            "memory": state["memory"],
            "summary": state["conversation_summary"],
        }
        results.append(result)

    print("Conversation State Manager Evaluation")
    print("=" * 45)
    for result in results:
        status = "PASS" if result["memory_matched_expected"] else "FAIL"
        print(f"[{status}] {result['conversation_id']}")
        print(f"  topic: {result['active_topic']}")
        print(
            "  tokens: "
            f"{result['token_before']} -> {result['token_after']} "
            f"({result['token_reduction_percent']}% reduction)"
        )
        print(
            "  summarization: "
            f"{result['summarization_triggered']} via {result['summary_provider']}"
        )
        if result["summary_provider"] != "openrouter":
            print(f"  summary_fallback_reason: {result['summary_fallback_reason']}")
        print(f"  memory_extraction: via {result['memory_provider']}")
        if result["memory_provider"] != "openrouter":
            print(f"  memory_fallback_reason: {result['memory_fallback_reason']}")
        print(f"  recent_messages: {result['recent_message_count']}")
        print(f"  expected_memory_fields_preserved: {result['expected_memory_fields_preserved']}")
        for field, item in result["memory_field_report"].items():
            field_status = "ok" if item["matched"] else "miss"
            print(f"    - {field}: {field_status}")
        print(f"  memory: {json.dumps(result['memory'], ensure_ascii=False)}")
        if result["summary"]:
            print(f"  summary: {result['summary']}")
        print()


if __name__ == "__main__":
    main()
