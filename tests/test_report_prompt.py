from __future__ import annotations

import json
from pathlib import Path

from oracle_report import prompt_templates


_PROMPT_TEMPLATE_NAMES = (
    "saju_reading",
    "saju_reading_couple",
)


def test_runtime_prompts_define_explicit_cache_prefixes() -> None:
    prompt_path = Path("configs/prompts.json")
    root = json.loads(prompt_path.read_text(encoding="utf-8"))
    assert root == {
        "include": [
            "prompts_personal.json",
            "prompts_compatibility.json",
        ],
    }

    template_info = {
        info.name: info for info in prompt_templates.list_prompt_template_info()
    }
    for prompt_name in _PROMPT_TEMPLATE_NAMES:
        prompt_config = json.loads(
            Path(
                "configs/prompts_personal.json"
                if prompt_name == "saju_reading"
                else "configs/prompts_compatibility.json",
            ).read_text(encoding="utf-8"),
        )[prompt_name]

        assert isinstance(prompt_config, dict)
        assert isinstance(prompt_config["id_slot"], int)
        assert isinstance(prompt_config["prefix"], list)
        assert isinstance(prompt_config["body"], list)
        assert prompt_config["prefix"]
        assert prompt_config["body"]
        prefix_str = "\n".join(prompt_config["prefix"])
        clean_prefix_str = prefix_str.replace("${saju_rules}", "")
        assert "${" not in clean_prefix_str
        assert "${" in "\n".join(prompt_config["body"])
        assert prompt_name in template_info
        assert template_info[prompt_name].slot_id == prompt_config["id_slot"]
