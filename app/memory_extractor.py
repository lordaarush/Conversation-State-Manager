import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.openrouter_client import OpenRouterClient
from app.prompts import MEMORY_EXTRACTION_SYSTEM_PROMPT


Message = Dict[str, str]


class MemoryExtractor:
    def __init__(self, client: OpenRouterClient, use_llm: bool = True) -> None:
        self.client = client
        self.use_llm = use_llm

    def extract_with_provider(
        self,
        messages: List[Message],
        existing_memory: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str, str]:
        existing = dict(existing_memory or {})

        if self.use_llm and self.client.has_api_key:
            raw_memory, error = self._extract_memory_with_openrouter(messages, existing)
            if raw_memory:
                parsed_memory, parse_error = self._parse_memory_json(raw_memory)
                if parse_error:
                    fallback = self._fallback_extract_memory_facts(messages)
                    return self._merge_memory(existing, fallback), "fallback", f"memory_json_error: {parse_error}"

                normalized = self._normalize_memory_facts(parsed_memory)
                return self._merge_memory(existing, normalized), "openrouter", ""

            fallback_reason = f"openrouter_error: {error}"
        elif self.use_llm:
            fallback_reason = "missing_openrouter_api_key"
        else:
            fallback_reason = "openrouter_memory_disabled"

        fallback = self._fallback_extract_memory_facts(messages)
        return self._merge_memory(existing, fallback), "fallback", fallback_reason

    def _extract_memory_with_openrouter(
        self,
        messages: List[Message],
        existing_memory: Dict[str, Any],
    ) -> Tuple[Optional[str], str]:
        payload = {
            "messages": [
                {"role": "system", "content": MEMORY_EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "existing_memory": existing_memory,
                            "conversation_turns": messages,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 400,
        }
        return self.client.chat_completion(payload)

    def _parse_memory_json(self, raw_memory: str) -> Tuple[Dict[str, Any], str]:
        text = raw_memory.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
            text = re.sub(r"\s*```$", "", text)

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return {}, "no_json_object_found"

        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as error:
            return {}, str(error)

        if not isinstance(parsed, dict):
            return {}, "memory_payload_is_not_object"
        return parsed, ""

    def _normalize_memory_facts(self, raw_memory: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        string_fields = {
            "issue",
            "intent",
            "plan",
            "environment",
            "region",
            "company_name",
            "workspace_name",
            "status",
        }
        bool_fields = {"oauth_enabled", "needs_invoice_export", "cancel_end_of_term"}
        list_fields = {"user_constraints", "commitments"}

        for field in string_fields:
            value = self._clean_string(raw_memory.get(field))
            if not value:
                continue
            if field == "region":
                value = self._normalize_region(value)
            elif field in {"issue", "intent", "plan", "environment", "status"}:
                value = value.lower().replace(" ", "_").replace("-", "_")
            normalized[field] = value

        for field in bool_fields:
            value = self._coerce_bool(raw_memory.get(field))
            if value is not None:
                normalized[field] = value

        invoice_numbers = self._coerce_invoice_numbers(raw_memory.get("invoice_numbers"))
        if invoice_numbers:
            normalized["invoice_numbers"] = invoice_numbers

        for field in list_fields:
            values = raw_memory.get(field)
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list):
                cleaned = [self._clean_string(value) for value in values]
                cleaned = [value for value in cleaned if value]
                if cleaned:
                    normalized[field] = cleaned

        additional_facts = raw_memory.get("additional_facts")
        if isinstance(additional_facts, dict):
            cleaned_facts = {}
            for key, value in additional_facts.items():
                clean_key = self._clean_string(key).lower().replace(" ", "_")
                clean_value = self._clean_memory_value(value)
                if clean_key and clean_value not in (None, "", [], {}):
                    cleaned_facts[clean_key] = clean_value
            if cleaned_facts:
                normalized["additional_facts"] = cleaned_facts

        return normalized

    def _fallback_extract_memory_facts(self, messages: Iterable[Message]) -> Dict[str, Any]:
        memory: Dict[str, Any] = {}
        combined = "\n".join(message.get("content", "") for message in messages)
        lowered = combined.lower()

        billing_score = sum(1 for word in ["billing", "charged", "charge", "refund", "payment"] if word in lowered)
        api_score = sum(1 for word in ["api", "webhook", "oauth", "integration", "timeout", "error"] if word in lowered)
        if billing_score or api_score:
            memory["issue"] = "api_support" if api_score > billing_score else "billing"
        if "cancel" in lowered or "cancellation" in lowered:
            memory["intent"] = "subscription_cancellation"

        plan_match = re.search(r"\b(free|starter|pro|business|enterprise|annual enterprise)\s+plan\b", lowered)
        if plan_match:
            memory["plan"] = plan_match.group(1).replace("annual ", "")
        elif "enterprise account" in lowered or "enterprise accounts" in lowered:
            memory["environment"] = "enterprise"

        invoices = sorted({int(match) for match in re.findall(r"\binvoice\s*#?\s*(\d+)\b", lowered)})
        if invoices:
            memory["invoice_numbers"] = invoices

        if "oauth enabled" in lowered or "oauth is enabled" in lowered:
            memory["oauth_enabled"] = True

        region_match = re.search(r"\b(us|usa|united states|eu|europe|uk|apac|asia)\s+region\b", lowered)
        if region_match:
            memory["region"] = self._normalize_region(region_match.group(1))

        if "invoice export" in lowered:
            memory["needs_invoice_export"] = True
        if "end of term" in lowered:
            memory["cancel_end_of_term"] = True

        return memory

    def _merge_memory(self, existing: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(existing)
        for key, value in updates.items():
            if value in (None, "", [], {}):
                continue
            if key == "invoice_numbers":
                existing_numbers = self._coerce_invoice_numbers(merged.get(key))
                merged[key] = sorted(set(existing_numbers + self._coerce_invoice_numbers(value)))
            elif key in {"user_constraints", "commitments"}:
                existing_values = merged.get(key, [])
                if not isinstance(existing_values, list):
                    existing_values = [existing_values]
                merged[key] = list(dict.fromkeys(existing_values + value))
            elif key == "additional_facts" and isinstance(value, dict):
                current = merged.get(key, {})
                if not isinstance(current, dict):
                    current = {}
                merged[key] = {**current, **value}
            else:
                merged[key] = value
        return merged

    def _clean_string(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        value = value.strip()
        if value.lower() in {"unknown", "n/a", "none", "null", "not specified", ""}:
            return ""
        return value

    def _clean_memory_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._clean_string(value)
        if isinstance(value, bool) or isinstance(value, int):
            return value
        if isinstance(value, list):
            cleaned = [self._clean_memory_value(item) for item in value]
            return [item for item in cleaned if item not in (None, "", [], {})]
        if isinstance(value, dict):
            return {
                self._clean_string(key).lower().replace(" ", "_"): self._clean_memory_value(item)
                for key, item in value.items()
                if self._clean_string(key)
            }
        return None

    def _normalize_region(self, value: str) -> str:
        lowered = value.lower().replace(" region", "").strip()
        return {
            "europe": "EU",
            "eu": "EU",
            "united states": "US",
            "usa": "US",
            "us": "US",
            "uk": "UK",
            "apac": "APAC",
            "asia": "APAC",
        }.get(lowered, value.upper() if len(value) <= 4 else value)

    def _coerce_bool(self, value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "enabled", "on"}:
                return True
            if lowered in {"false", "no", "disabled", "off"}:
                return False
        return None

    def _coerce_invoice_numbers(self, value: Any) -> List[int]:
        if value in (None, "", [], {}):
            return []
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            return sorted({int(match) for match in re.findall(r"\d+", value)})
        if isinstance(value, list):
            numbers = []
            for item in value:
                numbers.extend(self._coerce_invoice_numbers(item))
            return sorted(set(numbers))
        return []
