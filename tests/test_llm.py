from __future__ import annotations

import pytest

from oracle_report.config import LlmConfig, load_face_llm_config, load_llm_config
from oracle_report.llm import LlamaCppChatClient, _extract_output_text


def test_default_llm_config_uses_llama_cpp(monkeypatch) -> None:
    monkeypatch.delenv("ORACLE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ORACLE_LLM_MODEL", raising=False)

    config = load_llm_config()

    assert config.base_url == "http://127.0.0.1:8080/v1"
    assert config.model == "local-model"


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
