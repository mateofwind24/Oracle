from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_build_script_defaults_to_cpu_llama_cpp() -> None:
    script_text = (ROOT_DIR / "build.sh").read_text(encoding="utf-8")

    assert 'FORCE_CUDA="0"' in script_text
    assert "--cpu                    Force build llama.cpp in CPU-only mode (default)" in script_text
    assert "-DGGML_CUDA=OFF" in script_text
    assert "llama_cuda_cache_enabled" in script_text
    assert "reconfiguring CPU-only build" in script_text


def test_run_script_keeps_llama_cpu_by_default() -> None:
    script_text = (ROOT_DIR / "run.sh").read_text(encoding="utf-8")

    assert "running llama.cpp on CPU by default" in script_text
    assert "automatically setting --n-gpu-layers 99" not in script_text


def test_run_script_accepts_windows_virtualenv() -> None:
    script_text = (ROOT_DIR / "run.sh").read_text(encoding="utf-8")

    assert "activate_repo_venv" in script_text
    assert "Scripts/activate" in script_text


def test_run_script_consumes_legacy_distributed_options() -> None:
    script_text = (ROOT_DIR / "run.sh").read_text(encoding="utf-8")

    assert "--distributed-role|--master-addr|--slave-addrs)" in script_text
    assert "--distributed-role=*|--master-addr=*|--slave-addrs=*)" in script_text
    assert "--distributed-split|--distributed-warmup)" in script_text


def test_scripts_find_windows_llama_server_binary() -> None:
    build_script = (ROOT_DIR / "build.sh").read_text(encoding="utf-8")
    run_script = (ROOT_DIR / "run.sh").read_text(encoding="utf-8")
    service_script = (ROOT_DIR / "scripts" / "run_llama_server.sh").read_text(
        encoding="utf-8",
    )

    assert "llama-server.exe" in build_script
    assert "llama-server.exe" in run_script
    assert "llama-server.exe" in service_script
