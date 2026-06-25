from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from oracle_report.config import LlmConfig


class LlamaCppChatClient:
    def __init__(self, config: LlmConfig) -> None:
        self._config = config

    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        import requests
        import time

        payload = self._build_payload(prompt, image_path)
        t0 = time.perf_counter()
        response = requests.post(
            self._chat_completions_url(),
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=self._config.timeout_seconds,
        )
        t1 = time.perf_counter()
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(
                f"local llama.cpp request failed: HTTP {response.status_code}",
            )
        root = response.json()
        result = _extract_output_text(root)

        elapsed = t1 - t0
        usage = root.get("usage", {}) if isinstance(root.get("usage"), dict) else {}
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        
        speed_str = ""
        if completion_tokens > 0 and elapsed > 0:
            speed = completion_tokens / elapsed
            speed_str = f" ({speed:.2f} tokens/sec)"
        
        print(
            f"[LLM] Inference complete: prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}, elapsed={elapsed:.2f}s{speed_str}"
        )
        return result

    def _build_payload(self, prompt: str, image_path: Path | None) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path is not None and self._config.send_image:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _encode_image_data_url(image_path),
                        "detail": "low",
                    },
                },
            )
        result = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self._config.max_output_tokens,
            "temperature": self._config.temperature,
            "stream": False,
        }
        return result

    def _chat_completions_url(self) -> str:
        result = f"{self._config.base_url.rstrip('/')}/chat/completions"
        return result


def _encode_image_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    result = f"data:image/jpeg;base64,{encoded}"
    return result


def _extract_output_text(root: dict[str, Any]) -> str:
    result = ""
    choices = root.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                result = _message_content_to_text(message.get("content"))
    if result.strip() == "":
        raise RuntimeError("empty LLM response")
    return result


def _message_content_to_text(content: Any) -> str:
    result = ""
    if isinstance(content, str):
        result = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        result = "".join(parts)
    return result
