"""
key_manager.py

Manages a pool of Groq API keys loaded from groq_keys.json. If a call hits
a rate limit (HTTP 429), the manager marks that key as "cooling down",
switches to the next key, and retries - instead of the whole training/
generation loop dying because one key ran out of quota.

A separate failure mode - HTTP 413 / "rate_limit_exceeded" for a request
that is simply too large for the model's per-request token limit - is
classified distinctly (RequestTooLargeError) and is NOT retried across
keys: the payload is the same size no matter which key sends it, so
rotating keys just burns through the whole pool for a guaranteed repeat
failure. Callers that build the prompt (llm_reward_designer.py) are
expected to catch RequestTooLargeError and retry with a smaller prompt.

Usage (drop-in replacement for client.chat.completions.create(**kwargs)):

    from key_manager import get_key_manager, RequestTooLargeError

    manager = get_key_manager()
    try:
        response = manager.chat_completion(
            model="llama-3.1-8b-instant",
            messages=[...],
            max_tokens=2,
            temperature=0,
        )
    except RequestTooLargeError:
        # shrink the prompt and retry
        ...
"""

import json
import os
import time

GROQ_KEYS_PATH = "groq_keys.json"
COOLDOWN_SECONDS = 60  # how long to avoid a rate-limited key before trying it again


class RequestTooLargeError(Exception):
    """
    Raised when the API rejects a request because its payload exceeds a
    per-request size/token limit (HTTP 413, or a 4xx body reporting
    "rate_limit_exceeded" for request size rather than quota exhaustion).

    Unlike a 429 (quota exhaustion on THIS key), this is a property of the
    request itself - every key in the pool will hit the same limit on the
    same payload. Retrying with a different key wastes calls and time
    without any chance of success; the caller must shrink the request.
    """


def _is_rate_limit_error(e: Exception) -> bool:
    """
    True only for genuine per-key quota/rate-limit errors (HTTP 429), which
    *can* be worked around by switching to a different key. Request-too-
    large errors (413) are deliberately excluded here - see
    _is_request_too_large_error - because switching keys cannot fix them.
    """
    status_code = getattr(e, "status_code", None)
    if status_code == 429:
        return True
    if status_code is not None:
        # any other explicit status code (including 413) is NOT a 429,
        # regardless of what substrings appear in the message
        return False
    message = str(e).lower()
    return "rate limit" in message or "429" in message


def _is_request_too_large_error(e: Exception) -> bool:
    """
    True for "this single request is too big" errors - HTTP 413, or Groq's
    4xx body carrying an error code of "rate_limit_exceeded" together with
    request-size language (e.g. "Request too large for model ... Limit
    8000 ... Requested 8140"). Checked separately from, and BEFORE,
    _is_rate_limit_error so a message that happens to also mention a
    "limit" isn't misclassified as a retryable 429.
    """
    status_code = getattr(e, "status_code", None)
    if status_code == 413:
        return True
    message = str(e).lower()
    if "413" in message:
        return True
    if "request too large" in message:
        return True
    if "rate_limit_exceeded" in message and (
        "requested" in message and "limit" in message
    ):
        return True
    return False


class GroqKeyManager:
    def __init__(self, keys_path: str = GROQ_KEYS_PATH):
        self.keys_path = keys_path
        self.keys = self._load_keys()
        if not self.keys:
            raise RuntimeError(f"No Groq API keys found in {keys_path}")

        self.index = 0
        self.cooldown_until = {key: 0.0 for key in self.keys}

    def _load_keys(self) -> list[str]:
        if not os.path.exists(self.keys_path):
            raise FileNotFoundError(
                f"{self.keys_path} not found. Create it as:\n"
                '{"keys": ["gsk_...", "gsk_..."]}'
            )
        with open(self.keys_path) as f:
            data = json.load(f)
        keys = data.get("keys", [])
        # ignore obvious placeholder entries so a half-filled-in file doesn't
        # silently try to authenticate with the placeholder string
        return [k for k in keys if k and not k.startswith("gsk_REPLACE")]

    def _current_key(self) -> str:
        return self.keys[self.index]

    def _advance(self):
        self.index = (self.index + 1) % len(self.keys)

    def _mark_rate_limited(self, key: str):
        self.cooldown_until[key] = time.time() + COOLDOWN_SECONDS
        print(f"[key_manager] key ...{key[-6:]} rate-limited, cooling down {COOLDOWN_SECONDS}s")

    def _select_available_key(self) -> str:
        now = time.time()
        for _ in range(len(self.keys)):
            key = self._current_key()
            if self.cooldown_until[key] <= now:
                return key
            self._advance()
        # every key is currently in cooldown - use the current one anyway
        # (best effort; the caller's own error handling still applies)
        return self._current_key()

    def get_client(self):
        from groq import Groq
        key = self._select_available_key()
        return Groq(api_key=key), key

    def chat_completion(self, **kwargs):
        """
        Same signature as client.chat.completions.create(**kwargs). Tries
        up to one attempt per key in the pool, rotating past any key that
        returns a rate-limit error.

        Error handling order matters:
          1. Request-too-large (413) is checked FIRST and raised
             immediately as RequestTooLargeError, without consuming a
             rotation attempt - every key would fail identically on the
             same oversized payload, so rotating is pure waste.
          2. Genuine per-key 429s rotate to the next key and retry.
          3. Anything else is raised immediately (not fixed by switching
             keys either).
        """
        last_exception = None

        for _ in range(len(self.keys)):
            client, key = self.get_client()
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                if _is_request_too_large_error(e):
                    raise RequestTooLargeError(str(e)) from e
                last_exception = e
                if _is_rate_limit_error(e):
                    self._mark_rate_limited(key)
                    self._advance()
                    continue
                raise

        raise RuntimeError(
            f"All {len(self.keys)} Groq API keys are rate-limited or failing. "
            f"Last error: {last_exception}"
        )


_manager_singleton = None


def get_key_manager() -> GroqKeyManager:
    """Lazily builds and reuses one GroqKeyManager instance per process."""
    global _manager_singleton
    if _manager_singleton is None:
        _manager_singleton = GroqKeyManager()
    return _manager_singleton