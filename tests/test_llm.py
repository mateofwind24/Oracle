from __future__ import annotations

import pytest

from oracle_report.config import LlmConfig, load_face_llm_config, load_llm_config
from oracle_report.llm import (
    LlamaCppChatClient,
    _extract_finish_reason,
    _extract_output_text,
)
from oracle_report.prompt_templates import RenderedPrompt


def test_default_llm_config_uses_llama_cpp(monkeypatch) -> None:
    monkeypatch.delenv("ORACLE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ORACLE_LLM_MODEL", raising=False)
    monkeypatch.delenv("ORACLE_LLM_PROMPT_CACHE", raising=False)
    monkeypatch.delenv("ORACLE_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("ORACLE_REPORT_LLM_MAX_OUTPUT_TOKENS", raising=False)

    config = load_llm_config()

    assert config.base_url == "http://127.0.0.1:8080/v1"
    assert config.model == "local-model"
    assert config.prompt_cache is False
    assert config.max_output_tokens == 4096


def test_llm_config_enables_prompt_cache(monkeypatch) -> None:
    monkeypatch.setenv("ORACLE_LLM_PROMPT_CACHE", "1")

    config = load_llm_config()

    assert config.prompt_cache is True


def test_default_face_llm_config_uses_text_mode(monkeypatch) -> None:
    monkeypatch.setenv("ORACLE_LLM_SEND_IMAGE", "")
    monkeypatch.setenv("ORACLE_FACE_LLM_SEND_IMAGE", "")

    config = load_face_llm_config()

    assert config.send_image is False


@pytest.mark.parametrize(
    "env_name",
    (
        "ORACLE_LLM_BASE_URL",
        "ORACLE_FACE_LLM_BASE_URL",
        "ORACLE_REPORT_LLM_BASE_URL",
    ),
)
def test_rejects_non_local_llm_url(monkeypatch, env_name: str) -> None:
    monkeypatch.setenv(env_name, "https://example.com/v1")

    with pytest.raises(ValueError, match="local llama.cpp"):
        load_llm_config()


def test_llama_cpp_payload_uses_chat_completions_shape() -> None:
    config = LlmConfig(
        model="local-model",
        base_url="http://127.0.0.1:8080/v1",
        timeout_seconds=60.0,
        max_output_tokens=512,
        temperature=0.7,
        send_image=False,
    )
    client = LlamaCppChatClient(config)

    payload = client._build_payload("hello", None)

    assert payload["model"] == "local-model"
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"][0]["type"] == "text"
    assert payload["max_tokens"] == 512


def test_llama_cpp_payload_omits_prompt_cache_by_default() -> None:
    config = LlmConfig(
        model="local-model",
        base_url="http://127.0.0.1:8080/v1",
        timeout_seconds=60.0,
        max_output_tokens=512,
        temperature=0.7,
        send_image=False,
    )
    client = LlamaCppChatClient(config)
    prompt = RenderedPrompt(
        name="saju_reading",
        prefix="STATIC PREFIX",
        body="DYNAMIC INPUT",
        slot_id=1,
    )

    payload = client._build_payload(prompt, None)

    assert "id_slot" not in payload
    assert "cache_prompt" not in payload
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"][0]["text"] == (
        "STATIC PREFIX\n\nDYNAMIC INPUT"
    )


def test_llama_cpp_payload_uses_prompt_cache_slot_when_enabled() -> None:
    config = LlmConfig(
        model="local-model",
        base_url="http://127.0.0.1:8080/v1",
        timeout_seconds=60.0,
        max_output_tokens=512,
        temperature=0.7,
        send_image=False,
        prompt_cache=True,
    )
    client = LlamaCppChatClient(config)
    prompt = RenderedPrompt(
        name="saju_reading",
        prefix="STATIC PREFIX",
        body="DYNAMIC INPUT",
        slot_id=1,
    )

    payload = client._build_payload(prompt, None)

    assert payload["id_slot"] == 1
    assert payload["cache_prompt"] is True
    assert payload["messages"][0] == {"role": "system", "content": "STATIC PREFIX"}
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"][0]["text"] == "DYNAMIC INPUT"


def test_extracts_chat_completion_text() -> None:
    text = _extract_output_text(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "리포트 본문",
                    },
                },
            ],
        },
    )

    assert text == "리포트 본문"


def test_extracts_chat_completion_finish_reason() -> None:
    finish_reason = _extract_finish_reason(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": "잘린 리포트",
                    },
                },
            ],
        },
    )

    assert finish_reason == "length"
