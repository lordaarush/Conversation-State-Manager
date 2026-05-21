import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openrouter_client import OpenRouterClient
from prompts import SUMMARY_SYSTEM_PROMPT


Message = Dict[str, str]


class ConversationSummarizer:
    LOW_VALUE_PATTERNS = [
        re.compile(r"^(hi|hello|hey|thanks|thank you|okay|ok|sure|noted|please repeat)[.! ]*$", re.I),
    ]

    def __init__(self, client: OpenRouterClient, use_llm: bool = True) -> None:
        self.client = client
        self.use_llm = use_llm

    def summarize_with_provider(
        self,
        messages: List[Message],
        existing_summary: str = "",
        memory: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str]:
        if not messages:
            return existing_summary, "none", "no_older_messages_to_summarize"

        if self.use_llm and self.client.has_api_key:
            summary, error = self._summarize_with_openrouter(messages, existing_summary, memory or {})
            if summary:
                return summary, "openrouter", ""
            fallback_reason = f"openrouter_error: {error}"
        elif self.use_llm:
            fallback_reason = "missing_openrouter_api_key"
        else:
            fallback_reason = "openrouter_disabled"

        return self._fallback_summary(messages, existing_summary), "fallback", fallback_reason

    def _summarize_with_openrouter(
        self,
        messages: List[Message],
        existing_summary: str,
        memory: Dict[str, Any],
    ) -> Tuple[Optional[str], str]:
        payload = {
            "messages": [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "existing_summary": existing_summary,
                            "durable_memory": memory,
                            "older_messages": messages,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 300,
        }
        return self.client.chat_completion(payload)

    def _fallback_summary(self, messages: List[Message], existing_summary: str) -> str:
        useful_lines = []
        for message in messages:
            content = message.get("content", "").strip()
            if not content or any(pattern.match(content) for pattern in self.LOW_VALUE_PATTERNS):
                continue
            if message.get("role") == "assistant" and len(content.split()) < 5:
                continue
            useful_lines.append(content)

        selected = useful_lines[-5:]
        parts = []
        if existing_summary:
            parts.append(existing_summary.strip())
        if selected:
            parts.append("Earlier: " + " ".join(selected))
        return " ".join(parts).strip()
