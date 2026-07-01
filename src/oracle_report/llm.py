from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oracle_report.config import LlmConfig
from oracle_report.prompt_templates import RenderedPrompt


import threading

_INCOMPLETE_FINISH_REASONS = frozenset(("length",))
_ACTIVE_GENERATIONS_LOCK = threading.Lock()
_ACTIVE_GENERATIONS_COUNT = 0


def is_local_llm_running() -> bool:
    global _ACTIVE_GENERATIONS_COUNT
    with _ACTIVE_GENERATIONS_LOCK:
        result = _ACTIVE_GENERATIONS_COUNT > 0
    return result


class LlamaCppChatClient:
    _measured_tps: float | None = None
    _tps_lock = threading.Lock()

    def __init__(self, config: LlmConfig) -> None:
        self._config = config

    def get_or_measure_tps(self) -> float:
        """
        이 디바이스의 LLM 추론 속도(TPS)를 안전하게 실측하거나 캐싱된 값을 반환합니다.
        벤치마크 시 시스템 프롬프트 캐시(KV cache) 슬롯을 Evict/오염시키지 않기 위해
        'cache_prompt: False' 와 콤팩트한 더미 메시지 세트를 사용합니다.
        """
        with LlamaCppChatClient._tps_lock:
            if LlamaCppChatClient._measured_tps is not None:
                return LlamaCppChatClient._measured_tps

        import requests
        import time

        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "user", "content": "1"}
            ],
            "max_tokens": 5,
            "temperature": 0.1,
            "stream": False,
            "cache_prompt": False  # 프롬프트 캐시 기능 절대 미사용 (캐시 오염 방지)
        }

        try:
            t0 = time.perf_counter()
            response = requests.post(
                self._chat_completions_url(),
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=10.0
            )
            t1 = time.perf_counter()
            if response.status_code == 200:
                root = response.json()
                usage = root.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                elapsed = t1 - t0
                if completion_tokens > 0 and elapsed > 0:
                    tps = completion_tokens / elapsed
                    with LlamaCppChatClient._tps_lock:
                        LlamaCppChatClient._measured_tps = tps
                    print(f"[LLM][Benchmark] Auto-profiling complete: {tps:.2f} tokens/sec (Cache bypass enabled)")
                    return tps
        except Exception as e:
            print(f"[LLM][Benchmark] Auto-profiling failed: {e}. Defaulting to 1.0 TPS.")
        
        return 1.0

    def get_model_parameter_size(self) -> float:
        """
        모델 파일명/이름을 기반으로 대략적인 파라미터 크기(Billion 단위)를 유추합니다.
        """
        model_name = self._config.model.lower()
        if "1b" in model_name:
            return 1.0
        elif "2b" in model_name:
            return 2.0
        elif "7b" in model_name:
            return 7.0
        elif "8b" in model_name:
            return 8.0
        elif "9b" in model_name:
            return 9.0
        elif "27b" in model_name:
            return 27.0
        elif "e2b" in model_name or "gemma-4" in model_name:
            return 9.0
        return 2.0

    def get_compute_score(self) -> float:
        env_score = os.getenv("ORACLE_COMPUTE_SCORE")
        if env_score is not None:
            try:
                return float(env_score)
            except ValueError:
                pass
        tps = self.get_or_measure_tps()
        param_size = self.get_model_parameter_size()
        score = tps * param_size
        return score

    def generate(
        self,
        prompt: str | RenderedPrompt,
        image_path: Path | None = None,
    ) -> str:
        import requests
        import time

        global _ACTIVE_GENERATIONS_COUNT
        with _ACTIVE_GENERATIONS_LOCK:
            _ACTIVE_GENERATIONS_COUNT += 1

        try:
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
                error_preview = response.text.strip()
                if len(error_preview) > 300:
                    error_preview = error_preview[:300] + "..."
                prompt_chars = 0
                prefix_chars = 0
                user_chars = 0
                messages = payload.get("messages")
                if isinstance(messages, list):
                    for message in messages:
                        if not isinstance(message, dict):
                            continue
                        role = message.get("role")
                        content = message.get("content")
                        if isinstance(content, str):
                            prompt_chars += len(content)
                            if role == "system":
                                prefix_chars += len(content)
                        elif isinstance(content, list):
                            for item in content:
                                if not isinstance(item, dict):
                                    continue
                                text = item.get("text")
                                if isinstance(text, str):
                                    prompt_chars += len(text)
                                    user_chars += len(text)
                raise RuntimeError(
                    "local llama.cpp request failed: "
                    f"HTTP {response.status_code}; "
                    f"prefix_chars={prefix_chars}; "
                    f"user_chars={user_chars}; "
                    f"prompt_chars={prompt_chars}; "
                    f"response={error_preview or '<empty>'}",
                )
            elapsed = t1 - t0
            root = response.json()
            usage = root.get("usage", {}) if isinstance(root.get("usage"), dict) else {}
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            cached_tokens = _extract_cached_tokens(root)
            finish_reason = _extract_finish_reason(root)
            result = _extract_output_text(root)

            timings = root.get("timings", {}) if isinstance(root.get("timings"), dict) else {}
            predicted_per_sec = timings.get("predicted_per_second")
            predicted_ms = timings.get("predicted_ms")
            prompt_ms = timings.get("prompt_ms")

            prompt_eval_val = 0.0
            prompt_eval_str = ""
            if isinstance(prompt_ms, (int, float)) and prompt_ms > 0:
                prompt_eval_val = prompt_ms / 1000.0
                prompt_eval_str = f", prompt_eval={prompt_eval_val:.2f}s"

            generation_val = 0.0
            generation_str = ""
            if isinstance(predicted_ms, (int, float)) and predicted_ms > 0:
                generation_val = predicted_ms / 1000.0
                generation_str = f", generation={generation_val:.2f}s"

            speed_str = ""
            if isinstance(predicted_per_sec, (int, float)) and predicted_per_sec > 0:
                speed = predicted_per_sec
                speed_str = f", speed={speed:.2f} tokens/sec"
            elif generation_val > 0 and completion_tokens > 0:
                speed = completion_tokens / generation_val
                speed_str = f", speed={speed:.2f} tokens/sec"
            elif completion_tokens > 0 and elapsed > 0:
                speed = completion_tokens / elapsed
                speed_str = f", speed={speed:.2f} tokens/sec"

            print(
                f"[LLM] Inference complete: prompt_tokens={prompt_tokens}, "
                f"cached_tokens={cached_tokens}, "
                f"completion_tokens={completion_tokens}, "
                f"finish_reason={finish_reason or 'unknown'}, "
                f"elapsed={elapsed:.2f}s"
                f"{prompt_eval_str}"
                f"{generation_str}"
                f"{speed_str}"
            )
            if 'speed' in locals() and speed is not None and speed > 0:
                with LlamaCppChatClient._tps_lock:
                    LlamaCppChatClient._measured_tps = speed
            if finish_reason in _INCOMPLETE_FINISH_REASONS:
                raise RuntimeError(
                    "incomplete LLM response: "
                    f"finish_reason={finish_reason}, "
                    f"completion_tokens={completion_tokens}, "
                    f"max_output_tokens={self._config.max_output_tokens}",
                )
            return result
        finally:
            with _ACTIVE_GENERATIONS_LOCK:
                _ACTIVE_GENERATIONS_COUNT -= 1

    def _build_payload(
        self,
        prompt: str | RenderedPrompt,
        image_path: Path | None,
    ) -> dict[str, Any]:
        del image_path
        rendered_prompt = _coerce_rendered_prompt(prompt)
        if not self._config.prompt_cache:
            rendered_prompt = RenderedPrompt(
                name=rendered_prompt.name,
                prefix="",
                body=rendered_prompt.text,
                slot_id=None,
            )
        content: list[dict[str, Any]] = [{"type": "text", "text": rendered_prompt.body}]
        messages: list[dict[str, Any]] = []
        if rendered_prompt.prefix.strip() != "":
            messages.append({"role": "system", "content": rendered_prompt.prefix})
        messages.append({"role": "user", "content": content})
        max_tokens = self._config.max_output_tokens
        if self._config.reasoning:
            max_tokens = max(max_tokens, 8192)

        result = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self._config.temperature,
            "stream": False,
        }
        if rendered_prompt.slot_id is not None:
            result["id_slot"] = rendered_prompt.slot_id
        if self._config.prompt_cache:
            result["cache_prompt"] = True
        return result

    def _chat_completions_url(self) -> str:
        result = f"{self._config.base_url.rstrip('/')}/chat/completions"
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


def _coerce_rendered_prompt(prompt: str | RenderedPrompt) -> RenderedPrompt:
    if isinstance(prompt, RenderedPrompt):
        result = prompt
    else:
        result = RenderedPrompt(
            name="raw",
            prefix="",
            body=prompt,
            slot_id=None,
        )
    return result


def _extract_finish_reason(root: dict[str, Any]) -> str:
    result = ""
    choices = root.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            finish_reason = first_choice.get("finish_reason")
            if isinstance(finish_reason, str):
                result = finish_reason
    return result


def _extract_cached_tokens(root: dict[str, Any]) -> int:
    result = 0
    usage = root.get("usage")
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached_tokens = details.get("cached_tokens", 0)
            if isinstance(cached_tokens, int):
                result = cached_tokens
    if result == 0:
        timings = root.get("timings")
        if isinstance(timings, dict):
            cached_tokens = timings.get("cached_n", timings.get("prompt_cached_n", 0))
            if isinstance(cached_tokens, int):
                result = cached_tokens
    if result == 0:
        cached_tokens = root.get("tokens_cached", 0)
        if isinstance(cached_tokens, int):
            result = cached_tokens
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
