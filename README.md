# Conversation State Manager

Focused implementation for Part 3 of the AI Voice Support Agent assessment.

This module improves long-conversation handling by combining:

- **Conversation Summarization**  
  Older conversation history is compressed once context size exceeds a token threshold while preserving recent conversational flow.

- **Durable Memory Extraction**  
  Stable support facts are extracted and stored separately from raw transcripts to maintain conversational continuity.

The goal is to reduce context growth, preserve important support information, and improve long-session reliability.

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure OpenRouter (recommended):

```bash
copy .env.example .env
```

Add:

```bash
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=google/gemini-2.0-flash-001
```

Run demo:

```bash
python demo.py
```

Run tests:

```bash
pytest
```

The system also runs without an API key using lightweight fallback behavior.

---

## Assumptions

- Voice transcription and transcript normalization occur upstream.
- Messages arrive in chronological order.
- Recent conversational turns should remain uncompressed.
- Older conversational context may be summarized once token growth becomes large.
- The assistant primarily handles SaaS customer support scenarios such as billing, subscription management, and technical support.

---

## Limitations

- Token estimation is approximate rather than tokenizer-based.
- Helper-LLM memory extraction depends on API availability and structured JSON output quality.
- Fallback extraction intentionally remains lightweight and does not fully replace model-based extraction.
- Topic detection currently uses keyword scoring rather than a learned classifier.

---

## Future Improvements

- Model-specific token counting.
- Confidence scoring for extracted memory facts.
- Memory expiry and freshness handling.
- Stronger topic classification.
- PostgreSQL persistence for conversation state and operational metrics.

---

## Project Structure

```
app/main.py
Conversation state orchestration

app/memory_extractor.py
Durable memory extraction

app/summarizer.py
Conversation summarization

app/openrouter_client.py
OpenRouter integration

tests/
Automated tests

demo.py
Evaluation runner
```

---

## Example

Input:

```
User:
"I upgraded to Pro but invoice 1842 and 1843 both charged me."
```

Extracted memory:

```json
{
  "plan": "pro",
  "invoice_numbers": [1842,1843]
}
```

Long conversations automatically summarize older context while preserving recent turns and durable memory.
