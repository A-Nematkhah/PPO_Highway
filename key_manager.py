"""
key_manager.py

Manages a pool of Groq API keys loaded from groq_keys.json. If a call hits
a rate limit (HTTP 429), the manager marks that key as "cooling down",
switches to the next key, and retries - instead of the whole training/
generation loop dying because one key ran out of quota.

Usage (drop-in replacement for client.chat.completions.create(**kwargs)):

    from key_manager import get_key_manager

    manager = get_key_manager()
    response = manager.chat_completion(
        model="llama-3.1-8b-instant",
        messages=[...],
        max_tokens=2,
        temperature=0,
    )
"""

import json
import os
import time

GROQ_KEYS_PATH = "groq_keys.json"
COOLDOWN_SECONDS = 60  # how long to avoid a rate-limited key before trying it again


def _is_rate_limit_error(e: Exception) -> bool:
    status_code = getattr(e, "status_code", None)
    if status_code == 429:
        return True
    return "rate limit" in str(e).lower() or "429" in str(e)


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
        returns a rate-limit error. Non-rate-limit errors are raised
        immediately (they won't be fixed by switching keys).
        """
        last_exception = None

        for _ in range(len(self.keys)):
            client, key = self.get_client()
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as e:
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
