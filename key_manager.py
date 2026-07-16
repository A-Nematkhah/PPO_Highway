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

--------------------------------------------------------------------------
P2 fix #9: error classification widened beyond free-text substring matching
--------------------------------------------------------------------------
The previous classifiers only recognized a genuine 429 if the message
literally contained the substring "rate limit" or "429", and only
recognized a 413 if the message contained "413" or "request too large" (or
a very specific "rate_limit_exceeded" + "requested"/"limit" combination).
That misses other error shapes the Groq API (and OpenAI-compatible APIs in
general) can return for the same underlying conditions, e.g. a JSON error
body with `"code": "rate_limit_exceeded"` but human-readable text that
doesn't happen to say "rate limit", or messages like "quota exceeded" /
"too many requests" that mean the same thing in different words.

Classification is now layered:
    1. `status_code` on the exception, when present, is authoritative
       (413 -> too-large, 429 -> rate-limited).
    2. A structured error `code` field (Groq/OpenAI-style SDK exceptions
       commonly expose this via `.body["error"]["code"]` or similar) is
       checked against known code strings for each category.
    3. Free-text substring matching remains as a last-resort fallback for
       SDK versions/error shapes that expose neither of the above.
Order still matters: request-too-large is always checked BEFORE
rate-limited, since a 413 can otherwise be misclassified as a retryable
429 by a loose substring match (both mention "limit").
"""

import json
import os
import time

GROQ_KEYS_PATH = "groq_keys.json"
COOLDOWN_SECONDS = 60  # how long to avoid a rate-limited key before trying it again

# Known structured error-code strings (Groq / OpenAI-compatible APIs) for
# each failure category. Matched against whatever code string we can dig
# out of the exception, independent of the human-readable message text.
_TOO_LARGE_ERROR_CODES = frozenset({
    "request_too_large",
    "context_length_exceeded",
    "string_above_max_length",
})
_RATE_LIMITED_ERROR_CODES = frozenset({
    "rate_limit_exceeded",
    "requests_rate_limit_exceeded",
    "tokens_rate_limit_exceeded",
    "quota_exceeded",
    "insufficient_quota",
    "too_many_requests",
})


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


def _extract_status_code(e: Exception):
    """Best-effort status code lookup across SDK exception shapes."""
    status_code = getattr(e, "status_code", None)
    if status_code is not None:
        return status_code
    response = getattr(e, "response", None)
    return getattr(response, "status_code", None) if response is not None else None


def _extract_error_code(e: Exception) -> str | None:
    """
    Best-effort structured error-code lookup. Groq's Python SDK (and most
    OpenAI-compatible clients) attach the parsed JSON error body to the
    exception under a `.body` attribute shaped like
    `{"error": {"code": "...", "message": "...", ...}}`; some versions
    instead expose `.body["code"]` directly, or nest it under
    `.response.json()`. Try each in order and give up gracefully (returning
    None) rather than raising - this is a diagnostic aid, not something
    that should ever itself crash key rotation.
    """
    body = getattr(e, "body", None)
    for source in (body,):
        if isinstance(source, dict):
            error = source.get("error")
            if isinstance(error, dict) and isinstance(error.get("code"), str):
                return error["code"].lower()
            if isinstance(source.get("code"), str):
                return source["code"].lower()

    response = getattr(e, "response", None)
    if response is not None:
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and isinstance(error.get("code"), str):
                return error["code"].lower()
            if isinstance(payload.get("code"), str):
                return payload["code"].lower()

    return None


def _is_request_too_large_error(e: Exception) -> bool:
    """
    True for "this single request is too big" errors - HTTP 413, a
    structured error code in _TOO_LARGE_ERROR_CODES, or Groq's 4xx body
    carrying error code "rate_limit_exceeded" together with request-size
    language (e.g. "Request too large for model ... Limit 8000 ...
    Requested 8140"). Checked separately from, and BEFORE,
    _is_rate_limit_error so a message that happens to also mention a
    "limit" isn't misclassified as a retryable 429.
    """
    if _extract_status_code(e) == 413:
        return True

    error_code = _extract_error_code(e)
    if error_code in _TOO_LARGE_ERROR_CODES:
        return True

    message = str(e).lower()
    if "413" in message:
        return True
    if "request too large" in message:
        return True
    if error_code == "rate_limit_exceeded" and "requested" in message and "limit" in message:
        return True
    if "rate_limit_exceeded" in message and "requested" in message and "limit" in message:
        return True
    return False


def _is_rate_limit_error(e: Exception) -> bool:
    """
    True for genuine per-key quota/rate-limit errors (HTTP 429, or one of
    the known structured rate-limit/quota error codes), which *can* be
    worked around by switching to a different key. Request-too-large
    errors (413) are deliberately excluded here - see
    _is_request_too_large_error - because switching keys cannot fix them.
    Callers must check _is_request_too_large_error FIRST.
    """
    status_code = _extract_status_code(e)
    if status_code == 429:
        return True
    if status_code == 413:
        # explicit non-429 status code that isn't 429 - not a rate limit,
        # regardless of message text
        return False

    error_code = _extract_error_code(e)
    if error_code in _RATE_LIMITED_ERROR_CODES:
        return True
    if error_code in _TOO_LARGE_ERROR_CODES:
        return False

    if status_code is not None:
        # any other explicit status code is NOT a 429, regardless of message
        return False

    message = str(e).lower()
    return (
        "rate limit" in message
        or "rate_limit_exceeded" in message
        or "quota exceeded" in message
        or "quota_exceeded" in message
        or "too many requests" in message
        or "too_many_requests" in message
        or "429" in message
    )


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
          2. Genuine per-key 429s (or known rate-limit/quota error codes)
             rotate to the next key and retry.
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