import json
from pathlib import Path

import pytest

from main import ConversationStateManager, DEFAULT_OPENROUTER_MODEL


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "test_conversations.json"


def load_cases():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case["case_id"])
def test_conversation_state_pipeline_cases(case):
    manager = ConversationStateManager(
        token_threshold=case["token_threshold"],
        recent_window=case["recent_window"],
        use_llm_summary=False,
        use_llm_memory=False,
    )

    state = manager.process_conversation(case["conversation_id"], case["messages"])

    assert state["active_topic"] == case["expected_topic"]
    assert len(state["recent_messages"]) == case["expected_recent_count"]
    assert state["summarization_triggered"] is case["expected_summarization_triggered"]
    assert state["summary_provider"] == case["expected_summary_provider"]
    assert state["memory_provider"] == "fallback"

    for key, value in case["expected_memory"].items():
        assert state["memory"].get(key) == value

    if case.get("expected_summary_empty"):
        assert state["conversation_summary"] == ""
    elif state["summarization_triggered"] and not case.get("expected_summary_can_be_empty"):
        assert state["conversation_summary"]

    for excluded_text in case.get("expected_summary_excludes", []):
        assert excluded_text not in state["conversation_summary"]

    if case.get("expect_token_reduction"):
        assert state["token_after"] < state["token_before"]
        assert state["token_reduction_percent"] > 0


def test_merges_previous_state_without_losing_existing_memory():
    manager = ConversationStateManager(
        token_threshold=9999,
        recent_window=6,
        use_llm_summary=False,
        use_llm_memory=False,
    )
    previous = {"memory": {"plan": "enterprise"}, "active_topic": "subscription_management"}

    state = manager.process_conversation(
        "subscription-merge",
        [{"role": "user", "content": "Need invoice export and cancellation effective end of term."}],
        previous_state=previous,
    )

    assert state["memory"]["plan"] == "enterprise"
    assert state["memory"]["needs_invoice_export"] is True
    assert state["memory"]["cancel_end_of_term"] is True


def test_openrouter_model_can_be_configured_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")

    manager = ConversationStateManager()

    assert manager.openrouter_model == "google/gemini-2.0-flash-exp:free"


def test_openrouter_model_has_free_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    manager = ConversationStateManager()

    assert manager.openrouter_model == DEFAULT_OPENROUTER_MODEL


def test_fallback_summarization_without_openrouter_key(monkeypatch):
    manager = ConversationStateManager(
        token_threshold=1,
        recent_window=1,
        use_llm_summary=True,
        use_llm_memory=False,
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    state = manager.process_conversation(
        "fallback",
        [
            {"role": "user", "content": "My API integration fails with a webhook timeout."},
            {"role": "assistant", "content": "I will preserve the technical context."},
        ],
    )

    assert state["summarization_triggered"] is True
    assert state["summary_provider"] == "fallback"


def test_helper_llm_memory_extraction_is_preferred_and_normalized(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    manager = ConversationStateManager(token_threshold=9999, use_llm_summary=False, use_llm_memory=True)

    def fake_memory_call(messages, existing_memory):
        return (
            """
            {
              "issue": "API Support",
              "plan": "annual enterprise plan",
              "region": "EU region",
              "oauth_enabled": "enabled",
              "invoice_numbers": ["invoice 1842", 1843],
              "unsupported_guess": "do not keep this",
              "additional_facts": {
                "Webhook timeout": "10 seconds"
              }
            }
            """,
            "",
        )

    monkeypatch.setattr(manager.memory_extractor, "_extract_memory_with_openrouter", fake_memory_call)

    state = manager.process_conversation(
        "llm-memory",
        [{"role": "user", "content": "Webhook delivery times out after 10 seconds in EU."}],
    )

    assert state["memory_provider"] == "openrouter"
    assert state["memory"]["issue"] == "api_support"
    assert state["memory"]["plan"] == "enterprise"
    assert state["memory"]["region"] == "EU"
    assert state["memory"]["oauth_enabled"] is True
    assert state["memory"]["invoice_numbers"] == [1842, 1843]
    assert "unsupported_guess" not in state["memory"]
    assert state["memory"]["additional_facts"]["webhook_timeout"] == "10 seconds"


def test_malformed_helper_memory_json_falls_back(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    manager = ConversationStateManager(token_threshold=9999, use_llm_summary=False, use_llm_memory=True)
    monkeypatch.setattr(manager.memory_extractor, "_extract_memory_with_openrouter", lambda messages, existing: ("not json", ""))

    state = manager.process_conversation(
        "bad-memory-json",
        [{"role": "user", "content": "OAuth enabled for our API integration."}],
    )

    assert state["memory_provider"] == "fallback"
    assert state["memory_fallback_reason"].startswith("memory_json_error")
    assert state["memory"]["oauth_enabled"] is True
