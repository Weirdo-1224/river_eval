"""Local Qwen3-VL model adapter using Transformers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from benchmark.models.base import BaseModel
from benchmark.models.registry import register_model
from benchmark.storage.cache import RequestCache


@register_model("qwen3vl_local")
class QwenLocalModel(BaseModel):
    """Adapter for local Qwen3-VL checkpoints.

    The first local baseline treats the sampled RIVER frames as multiple image
    inputs. It does not implement the original RIVER model-internal long-short
    memory compression.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model: str | Path | None = None,
        temperature: float = 0.0,
        max_tokens: int = 16,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        attn_implementation: str | None = None,
        cache: RequestCache | None = None,
        cache_errors: bool = False,
        reuse_failed_cache: bool = False,
        **_: Any,
    ) -> None:
        checkpoint = model_path or model
        if checkpoint is None:
            raise ValueError("Qwen3-VL local model requires model_path.")

        self.model_path = Path(checkpoint)
        self.model_id = str(self.model_path)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.cache = cache
        self.cache_errors = cache_errors
        self.reuse_failed_cache = reuse_failed_cache
        self._last_latency = 0.0
        self._last_cached = False
        self._last_usage: dict[str, int] = {}
        self._last_status = "init"
        self._last_error: str | None = None

        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL local inference requires torch and transformers in the active "
                "conda environment. Install them in benchmark_new_qwen8b before running."
            ) from exc

        dtype = self._resolve_dtype(torch, torch_dtype)
        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            **load_kwargs,
        )
        self.model.eval()

    @staticmethod
    def _resolve_dtype(torch: Any, value: str) -> str | Any:
        if value in (None, "auto"):
            return "auto"
        normalized = str(value).lower()
        if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
            return torch.bfloat16
        if normalized in {"fp16", "float16", "torch.float16", "half"}:
            return torch.float16
        if normalized in {"fp32", "float32", "torch.float32"}:
            return torch.float32
        raise ValueError(f"Unsupported torch_dtype for Qwen3-VL local model: {value}")

    def generate(
        self,
        messages: list[dict[str, Any]],
        image_paths: list[Path],
        memory: Any = None,
        sample_id: str = "",
    ) -> str:
        del memory
        if self.cache is not None:
            cache_key = RequestCache.make_key(self.model_id, sample_id, messages, image_paths)
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
                return cached["raw_output"]

        request_messages = self._build_qwen_messages(messages, image_paths)
        start = time.time()
        try:
            inputs = self.processor.apply_chat_template(
                request_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._input_device())
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_tokens,
                "do_sample": self.temperature > 0,
            }
            if self.temperature > 0:
                generation_kwargs["temperature"] = self.temperature

            with self.torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            raw = decoded[0].strip() if decoded else ""
        except Exception as exc:
            self._last_latency = time.time() - start
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

        self._last_latency = time.time() - start
        self._last_cached = False
        self._last_usage = {}
        self._last_status = "success"
        self._last_error = None

        if self.cache is not None:
            self.cache.set(cache_key, {"status": "success", "raw_output": raw, "usage": {}})

        return raw

    def _input_device(self) -> Any:
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    def _build_qwen_messages(
        self,
        messages: list[dict[str, Any]],
        image_paths: list[Path],
    ) -> list[dict[str, Any]]:
        qwen_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                qwen_messages.append(
                    {"role": "system", "content": [{"type": "text", "text": msg["content"]}]}
                )

        user_content: list[dict[str, Any]] = []
        for image_path in image_paths:
            user_content.append({"type": "image", "image": str(image_path)})
        for msg in messages:
            if msg.get("role") == "user":
                user_content.append({"type": "text", "text": msg["content"]})
        qwen_messages.append({"role": "user", "content": user_content})
        return qwen_messages
