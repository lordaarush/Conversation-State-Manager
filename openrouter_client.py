import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib import request
from urllib.error import HTTPError, URLError


DEFAULT_OPENROUTER_MODEL = "google/gemini-2.0-flash-001"


def load_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs without requiring python-dotenv."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class OpenRouterClient:
    def __init__(self, model: Optional[str] = None) -> None:
        load_env_file()
        self.model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)

    @property
    def has_api_key(self) -> bool:
        return bool(os.getenv("OPENROUTER_API_KEY"))

    def chat_completion(self, payload: Dict[str, Any]) -> Tuple[Optional[str], str]:
        payload = {**payload, "model": payload.get("model") or self.model}
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://local-conversation-state-manager",
                "X-Title": "Conversation State Manager Demo",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=20) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip(), ""
        except HTTPError as error:
            try:
                body = error.read().decode("utf-8")
            except OSError:
                body = ""
            detail = body[:300].replace("\n", " ").strip()
            return None, f"HTTP {error.code} {error.reason}: {detail}"
        except URLError as error:
            return None, f"network_error: {error.reason}"
        except TimeoutError:
            return None, "timeout"
        except (KeyError, IndexError) as error:
            return None, f"unexpected_response_shape: {error}"
        except json.JSONDecodeError as error:
            return None, f"invalid_json_response: {error}"
        except OSError as error:
            return None, f"os_error: {error}"
