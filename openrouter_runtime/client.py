from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_RPM = 20
DEFAULT_MAX_CONCURRENCY = 4


def load_runtime_env(project_root: Path) -> None:
    """Load env files in order, preferring the OpenRouter-specific file."""
    load_dotenv(project_root / ".env", override=False)
    load_dotenv(project_root / ".env.openrouter", override=True)


@dataclass(slots=True)
class ChatResult:
    text: str
    provider: str | None
    model: str
    latency_seconds: float
    usage: dict[str, Any]
    raw_id: str | None


class SlidingWindowRateLimiter:
    """Thread-safe rate limiter for API calls."""

    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = max(1, requests_per_minute)
        self._timestamps: deque[float] = deque()
        self._condition = threading.Condition()

    def acquire(self) -> None:
        while True:
            with self._condition:
                now = time.monotonic()
                cutoff = now - 60.0
                while self._timestamps and self._timestamps[0] < cutoff:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    self._condition.notify_all()
                    return

                sleep_for = max(0.05, 60.0 - (now - self._timestamps[0]))

            time.sleep(min(sleep_for, 3.0))


class OpenRouterClient:
    """Thin OpenRouter client with retries and shared rate limiting."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        site_url: str | None = None,
        app_name: str | None = None,
        requests_per_minute: int = DEFAULT_RPM,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        timeout_seconds: int = 120,
        max_retries: int = 8,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.site_url = site_url
        self.app_name = app_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.rate_limiter = SlidingWindowRateLimiter(requests_per_minute)
        self.max_concurrency = max(1, max_concurrency)
        self._concurrency_limiter = threading.BoundedSemaphore(
            self.max_concurrency
        )

    @classmethod
    def from_env(
        cls,
        project_root: Path,
        *,
        requests_per_minute: int = DEFAULT_RPM,
        max_concurrency: int | None = None,
        timeout_seconds: int = 120,
        max_retries: int = 8,
    ) -> "OpenRouterClient":
        load_runtime_env(project_root)

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        model = os.getenv(
            "OPENROUTER_MODEL",
            "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        )
        base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
        site_url = os.getenv("OPENROUTER_SITE_URL")
        app_name = os.getenv("OPENROUTER_APP_NAME", project_root.name)
        if max_concurrency is None:
            max_concurrency = int(
                os.getenv(
                    "OPENROUTER_MAX_CONCURRENCY",
                    str(DEFAULT_MAX_CONCURRENCY),
                )
            )

        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            site_url=site_url,
            app_name=app_name,
            requests_per_minute=requests_per_minute,
            max_concurrency=max_concurrency,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    def key_status(self) -> dict[str, Any]:
        resp = self.session.get(
            f"{self.base_url}/key",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json().get("data", {})

    def list_models(self) -> list[dict[str, Any]]:
        resp = self.session.get(
            f"{self.base_url}/models",
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])

    def get_model_info(self, model_id: str | None = None) -> dict[str, Any] | None:
        target = model_id or self.model
        for model in self.list_models():
            if model.get("id") == target:
                return model
        return None

    def ensure_model_available(self, model_id: str | None = None) -> dict[str, Any]:
        target = model_id or self.model
        info = self.get_model_info(target)
        if info is not None:
            return info

        candidates = []
        target_tokens = {
            token
            for token in re_split_model_id(target.lower())
            if token and token not in {"free", "instruct"}
        }
        for model in self.list_models():
            mid = model.get("id", "").lower()
            if any(token in mid for token in target_tokens):
                candidates.append(model.get("id"))

        hint = ""
        if candidates:
            hint = f" Similar available models: {', '.join(candidates[:5])}."
        raise ValueError(f"Model '{target}' is not in the OpenRouter catalog.{hint}")

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        attempt = 0
        while True:
            attempt += 1
            self.rate_limiter.acquire()
            with self._concurrency_limiter:
                started = time.monotonic()
                resp = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            latency = time.monotonic() - started

            if resp.ok:
                body = resp.json()
                choice = body["choices"][0]["message"]["content"].strip()
                provider = None
                if isinstance(body.get("provider"), dict):
                    provider = body["provider"].get("name")
                return ChatResult(
                    text=choice,
                    provider=provider,
                    model=body.get("model", self.model),
                    latency_seconds=round(latency, 3),
                    usage=body.get("usage", {}),
                    raw_id=body.get("id"),
                )

            should_retry = resp.status_code in {408, 429, 500, 502, 503, 504}
            if not should_retry or attempt >= self.max_retries:
                raise RuntimeError(
                    f"OpenRouter chat failed: status={resp.status_code} "
                    f"body={resp.text[:500]}"
                )

            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                sleep_for = float(retry_after)
            else:
                sleep_for = min(90.0, 5.0 * attempt)
                if "temporarily rate-limited upstream" in resp.text:
                    sleep_for = min(120.0, 15.0 * attempt)
                sleep_for += random.uniform(0.0, 1.5)
            print(
                f"Retrying OpenRouter request after {sleep_for:.1f}s "
                f"(attempt {attempt}/{self.max_retries}, status {resp.status_code})"
            )
            time.sleep(sleep_for)

    def save_key_status(self, output_path: Path) -> None:
        output_path.write_text(
            json.dumps(self.key_status(), indent=2),
            encoding="utf-8",
        )


def re_split_model_id(model_id: str) -> list[str]:
    return [part for part in model_id.replace("/", "-").replace(":", "-").split("-") if part]
