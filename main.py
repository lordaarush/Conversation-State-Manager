from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from memory_extractor import MemoryExtractor
from openrouter_client import DEFAULT_OPENROUTER_MODEL, OpenRouterClient
from summarizer import ConversationSummarizer


Message = Dict[str, str]


@dataclass
class ConversationState:
    conversation_id: str
    active_topic: str = "general_support"
    conversation_summary: str = ""
    recent_messages: List[Message] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)


class ConversationStateManager:
    """
    Maintains compact support-conversation state by separating:
    - conversation_summary: compressed older context
    - recent_messages: most recent turns
    - memory: durable support facts
    - active_topic: current support subject
    """

    TOPIC_KEYWORDS = {
        "billing": ["billing", "charged", "charge", "invoice", "refund", "payment", "receipt"],
        "technical_support": ["api", "webhook", "timeout", "integration", "oauth", "error", "delivery"],
        "subscription_management": ["cancel", "cancellation", "subscription", "plan", "end of term", "renewal"],
    }

    def __init__(
        self,
        token_threshold: int = 12000,
        recent_window: int = 6,
        use_llm_summary: bool = True,
        use_llm_memory: bool = True,
        openrouter_model: Optional[str] = None,
    ) -> None:
        self.token_threshold = token_threshold
        self.recent_window = recent_window
        self.openrouter_client = OpenRouterClient(model=openrouter_model)
        self.openrouter_model = self.openrouter_client.model
        self.memory_extractor = MemoryExtractor(self.openrouter_client, use_llm=use_llm_memory)
        self.summarizer = ConversationSummarizer(self.openrouter_client, use_llm=use_llm_summary)

    def estimate_token_count(self, value: Any) -> int:
        """Approximate token count without requiring tokenizer downloads."""
        text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        return max(1, len(text) // 4)

    def detect_topic(self, messages: Iterable[Message], existing_topic: str = "general_support") -> str:
        scores = {topic: 0 for topic in self.TOPIC_KEYWORDS}
        for message in messages:
            content = message.get("content", "").lower()
            for topic, keywords in self.TOPIC_KEYWORDS.items():
                scores[topic] += sum(1 for keyword in keywords if keyword in content)

        best_topic, best_score = max(scores.items(), key=lambda item: item[1])
        return best_topic if best_score > 0 else existing_topic

    def prune_recent_messages(self, messages: List[Message]) -> List[Message]:
        return messages[-self.recent_window :]

    def process_conversation(
        self,
        conversation_id: str,
        messages: List[Message],
        previous_state: Optional[ConversationState | Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = self._coerce_state(conversation_id, previous_state)
        token_before = self.estimate_token_count(
            {
                "summary": state.conversation_summary,
                "memory": state.memory,
                "messages": messages,
            }
        )

        active_topic = self.detect_topic(messages, state.active_topic)
        memory, memory_provider, memory_fallback_reason = self.memory_extractor.extract_with_provider(
            messages,
            state.memory,
        )

        should_summarize = token_before > self.token_threshold
        if should_summarize:
            older_messages = messages[: -self.recent_window]
            conversation_summary, summary_provider, summary_fallback_reason = self.summarizer.summarize_with_provider(
                older_messages,
                state.conversation_summary,
                memory,
            )
            recent_messages = self.prune_recent_messages(messages)
        else:
            conversation_summary = state.conversation_summary
            summary_provider = "none"
            summary_fallback_reason = "summarization_not_triggered"
            recent_messages = list(messages)

        compact_context = {
            "active_topic": active_topic,
            "conversation_summary": conversation_summary,
            "recent_messages": recent_messages,
            "memory": memory,
        }
        token_after = self.estimate_token_count(compact_context) if should_summarize else token_before

        return {
            "conversation_id": conversation_id,
            **compact_context,
            "token_before": token_before,
            "token_after": token_after,
            "summarization_triggered": should_summarize,
            "summary_provider": summary_provider,
            "summary_fallback_reason": summary_fallback_reason,
            "memory_provider": memory_provider,
            "memory_fallback_reason": memory_fallback_reason,
            "token_reduction_percent": round(
                max(0, token_before - token_after) / max(token_before, 1) * 100,
                2,
            ),
        }

    def _coerce_state(
        self,
        conversation_id: str,
        previous_state: Optional[ConversationState | Dict[str, Any]],
    ) -> ConversationState:
        if previous_state is None:
            return ConversationState(conversation_id=conversation_id)
        if isinstance(previous_state, ConversationState):
            return previous_state
        return ConversationState(
            conversation_id=previous_state.get("conversation_id", conversation_id),
            active_topic=previous_state.get("active_topic", "general_support"),
            conversation_summary=previous_state.get("conversation_summary", ""),
            recent_messages=previous_state.get("recent_messages", []),
            memory=previous_state.get("memory", {}),
        )
