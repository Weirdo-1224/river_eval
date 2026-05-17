"""OpenAI GPT-4o model adapter with caching and cost tracking."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import openai

from benchmark.models.base import BaseModel
from benchmark.models.registry import register_model
from benchmark.storage.cache import RequestCache
from benchmark.storage.cost_tracker import CostTracker


@register_model("openai_api")
@register_model("openai")
class OpenAIModel(BaseModel):
    """Adapter for OpenAI vision models (e.g. gpt-4o)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 16,
        api_key: str | None = None,
        base_url: str | None = None,
        cache: RequestCache | None = None,
        cost_tracker: CostTracker | None = None,
        cache_errors: bool = False,
        reuse_failed_cache: bool = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.cost_tracker = cost_tracker
        self.cache_errors = cache_errors
        self.reuse_failed_cache = reuse_failed_cache
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key."
            )
        self.client = openai.OpenAI(api_key=key, base_url=base_url)
        self._last_latency = 0.0
        self._last_cached = False
        self._last_usage: dict[str, int] = {}
        self._last_status = "init"
        self._last_error: str | None = None

    def generate(
        self,
        messages: list[dict[str, Any]],
        image_paths: list[Path],
        memory: Any = None,
        sample_id: str = "",
    ) -> str:
        """Send a vision request with image frames.

        Args:
            messages: Chat messages in OpenAI format.
            image_paths: Paths to image frames to include in the prompt.
            memory: Optional memory context (reserved for Phase 4).
            sample_id: Sample identifier for cache key.
        """
        # Build content list: text messages first, then images.
        content: list[dict[str, Any]] = []

        # Add system/user text from messages.
        for msg in messages:
            if msg.get("role") == "system":
                content.append({"type": "text", "text": msg["content"]})

        # Add images.
        for img_path in image_paths:
            content.append(self._encode_image(img_path))

        # Add user text after images (GPT-4o typically expects images before the question).
        for msg in messages:
            if msg.get("role") == "user":
                content.append({"type": "text", "text": msg["content"]})

        request_messages = [{"role": "user", "content": content}]

        # Check cache.
        if self.cache is not None:
            cache_key = RequestCache.make_key(self.model, sample_id, messages, image_paths)
            cached = self.cache.get(cache_key)
            cached_status = cached.get("status") if cached is not None else None
            if cached is not None and cached_status is None:
                cached_status = "failed" if str(cached.get("raw_output", "")).startswith("<ERROR:") else "success"
            if cached is not None and (cached_status == "success" or self.reuse_failed_cache):
                self._last_latency = 0.0
                self._last_cached = True
                self._last_usage = cached.get("usage", {})
                self._last_status = cached_status
                self._last_error = cached.get("error")
                if self.cost_tracker is not None:
                    self.cost_tracker.add(
                        self.model,
                        self._last_usage.get("prompt_tokens", 0),
                        self._last_usage.get("completion_tokens", 0),
                    )
                return cached["raw_output"]

        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=request_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            latency = time.time() - start
            self._last_latency = latency
            self._last_cached = False
            self._last_usage = {}
            self._last_status = "failed"
            self._last_error = f"{type(exc).__name__}: {exc}"
            error_msg = f"<ERROR: {type(exc).__name__}: {exc}>"
            if self.cache is not None and self.cache_errors:
                self.cache.set(
                    cache_key,
                    {
                        "status": "failed",
                        "raw_output": error_msg,
                        "usage": {},
                        "error": self._last_error,
                    },
                )
            return error_msg

        latency = time.time() - start

        raw = response.choices[0].message.content or ""
        self._last_latency = latency
        self._last_cached = False
        self._last_status = "success"
        self._last_error = None

        # Extract usage.
        usage = {}
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        self._last_usage = usage

        # Track cost.
        if self.cost_tracker is not None:
            self.cost_tracker.add(
                self.model,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )

        # Save to cache.
        if self.cache is not None:
            self.cache.set(cache_key, {"status": "success", "raw_output": raw, "usage": usage})

        return raw

    def _encode_image(self, path: Path) -> dict[str, Any]:
        """Encode an image file as a base64 data URL."""
        ext = path.suffix.lower().replace(".", "")
        if ext == "jpg":
            ext = "jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/{ext};base64,{b64}"},
        }
