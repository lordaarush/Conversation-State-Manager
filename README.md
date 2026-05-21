# Conversation State Manager

Focused implementation for Part 3 of the AI Voice Support Agent assignment.

The module keeps long support conversations usable without repeatedly sending the full transcript to the model. It separates state into:

- `conversation_summary`: compressed older context
- `recent_messages`: most recent conversational turns
- `memory`: durable support facts extracted by a helper LLM, such as plan, invoice IDs, region, OAuth status, and cancellation intent
- `active_topic`: coarse support topic for continuity

Summarization is triggered by estimated token growth, not by message count alone. This avoids over-compressing short conversations that happen to contain more turns than the recent-message window.

Memory extraction is OpenRouter-first by default. If the helper LLM is unavailable or returns malformed JSON, the manager falls back to a lightweight deterministic extractor so the demo and tests still run offline.

## Setup

```bash
pip install -r requirements.txt
python demo.py
pytest
```

OpenRouter summarization is the primary path when an API key is available:

```bash
copy .env.example .env
```

Then edit `.env`:

```bash
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=google/gemini-2.0-flash-001
```

Run:

```bash
python demo.py
```

The current default helper model is `google/gemini-2.0-flash-001`. You can change it with `OPENROUTER_MODEL` or by passing `openrouter_model=...` to `ConversationStateManager`.

The implementation still runs without an API key. For memory extraction, the fallback uses simple deterministic patterns as a baseline only; the primary implementation is the helper LLM. For summarization, fallback is heuristic extractive compression. Neither fallback is TF-IDF or TextRank.

To force offline fallback even when a key exists:

```bash
python demo.py --no-llm-summary --no-llm-memory
```

## Usage

```python
from main import ConversationStateManager

manager = ConversationStateManager(token_threshold=12000, recent_window=6)
state = manager.process_conversation(
    conversation_id="conversation-123",
    messages=[
        {"role": "user", "content": "I upgraded to the Pro plan but was charged twice."},
        {"role": "user", "content": "Invoice 1842 and invoice 1843 show the same charge."},
    ],
)
```

Returned state:

```json
{
  "conversation_id": "conversation-123",
  "active_topic": "billing",
  "conversation_summary": "...",
  "recent_messages": [
    {"role": "user", "content": "I upgraded to the Pro plan but was charged twice."}
  ],
  "memory": {
    "issue": "billing",
    "plan": "pro",
    "invoice_numbers": [1842, 1843]
  },
  "token_before": 12345,
  "token_after": 3200,
  "memory_provider": "openrouter",
  "summary_provider": "openrouter",
  "token_reduction_percent": 74.08
}
```

## Code Structure

- `main.py`: orchestration pipeline, token thresholding, topic tracking, recent-message retention
- `memory_extractor.py`: helper-LLM memory extraction, JSON parsing, normalization, deterministic fallback
- `summarizer.py`: helper-LLM summarization and fallback summary generation
- `openrouter_client.py`: `.env` loading and OpenRouter API calls
- `prompts.py`: helper LLM system prompts
- `demo.py`: human-readable evaluation runner
- `tests/`: automated behavior checks

## Evaluation

`demo.py` runs the manager against `sample_conversations.json` and reports:

- token count before and after compaction
- token reduction percentage
- active topic
- whether summarization was triggered
- whether the summary came from OpenRouter or fallback heuristics
- whether memory extraction came from OpenRouter or fallback
- extracted durable memory
- field-level expected memory preservation

For demonstration, the default threshold is intentionally low (`250`) and a long sample is included so summarization/pruning is visible. In production, a threshold such as `12000` is more realistic. Very short conversations should normally remain uncompressed.

Automated tests use `tests/fixtures/test_conversations.json`, which contains structured conversations covering billing memory extraction, API topic tracking, subscription intent, short-conversation no-pruning behavior, long-conversation cost reduction, topic switching, and noisy low-value turns.

## Assumptions

- Voice transcription and normalization happen upstream.
- Message history arrives in chronological order.
- Recent turns should be preserved verbatim because they carry conversational flow.
- Older context can be compressed as long as support-critical facts are preserved separately.
- This module is called by a larger assistant workflow before prompt/context construction.

## Limitations

- Token counting is approximate to avoid requiring model-specific tokenizer packages.
- Helper-LLM memory extraction depends on API availability and JSON quality; deterministic fallback is intentionally narrow and only covers representative support facts.
- Fallback summaries use simple heuristic extraction, not TF-IDF/TextRank, and are less fluent than LLM-generated summaries.
- Active topic tracking uses keyword scoring, so ambiguous conversations may need a model-assisted classifier later.

## Future Improvements

- Replace heuristic token estimation with model-specific tokenizers.
- Track confidence per memory fact and source message IDs.
- Add expiry rules for stale memory.
- Add more topic categories and richer evaluation data.
- Store compact state in PostgreSQL tables alongside message and response records.
