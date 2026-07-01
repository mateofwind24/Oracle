# QLoRA Fine-Tuning

이 브랜치는 현재 런타임 GGUF 모델과 같은 base 계열의 Hugging Face 모델에
QLoRA adapter를 학습하는 흐름을 제공합니다.

## 모델 기준

- 현재 런타임 GGUF: `unsloth/gemma-4-E2B-it-GGUF`
- HF base model: `google/gemma-4-E2B-it`
- 기본 QLoRA 입력 모델: `unsloth/gemma-4-E2B-it` with `load_in_4bit=True`

GGUF 파일은 llama.cpp 추론용 산출물이므로 직접 QLoRA 학습하지 않습니다.
학습은 4-bit Hugging Face 모델에서 adapter를 만들고, 필요하면 병합 또는 GGUF
export를 별도로 수행합니다.

## 데이터 생성

기본 데이터는 저작권 있는 애니메이션 대사나 특정 캐릭터 고유 말투를 복사하지
않습니다. 네트워크 옵션을 켜면 한국어 Wikipedia, Wiktionary, Wikiquote,
Wikibooks, Wikinews, Wikidata API에서 제목/라벨 seed만 수집하고, 본문이나 실제
대사는 학습 데이터에 저장하지 않습니다.

```powershell
python scripts/finetune/build_korean_cute_dataset.py --allow-network --target-count 6000
```

산출물:

- `data/finetune/korean_cute_style_train.jsonl`
- `data/finetune/korean_cute_style_sources.json`

## QLoRA 학습

노트북/Linux 환경에서는 아래 한 줄을 기본 진입점으로 사용합니다. 이미 설치된
venv, CUDA torch, fine-tune 패키지는 스킵하고, 데이터가 충분하면 크롤링도
스킵합니다.

```bash
bash train.sh
```

환경 변수로 기본값을 바꿀 수 있습니다.

```bash
ORACLE_FINETUNE_TARGET_COUNT=8000 \
ORACLE_FINETUNE_NUM_TRAIN_EPOCHS=1 \
ORACLE_FINETUNE_MAX_SEQ_LENGTH=1024 \
bash train.sh
```

수동 실행은 아래와 같습니다.

```powershell
python -m venv runs/finetune-venv
.\runs\finetune-venv\Scripts\python.exe -m pip install --upgrade pip "setuptools<82" wheel
.\runs\finetune-venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
.\runs\finetune-venv\Scripts\python.exe -m pip install -r requirements-finetune.txt
.\runs\finetune-venv\Scripts\python.exe scripts/finetune/train_qlora.py `
  --dataset data/finetune/korean_cute_style_train.jsonl `
  --output-dir runs/finetune/korean-cute-lora `
  --max-seq-length 1024 `
  --max-steps 60 `
  --batch-size 1 `
  --grad-accumulation 8 `
  --offload-embedding
```

`requirements-finetune.txt`는 Gemma 4 지원을 위해 PyPI 구버전 Unsloth가 아니라
GitHub의 최신 Unsloth/Unsloth Zoo를 설치합니다.

현재 RTX 4050 Laptop 6GB VRAM에서는 E2B QLoRA가 빠듯할 수 있습니다. Unsloth
문서 기준 E2B LoRA는 8GB VRAM부터가 권장 범위입니다. OOM이
나면 `--max-seq-length 512`, `--lora-rank 8`, `--grad-accumulation 16` 순서로
줄여 재시도합니다.

기본 학습 스크립트는 말투 SFT에 맞춰 `text_only=True`로 로드합니다.
`--include-multimodal`을 주면 이미지/오디오 encoder까지 로드하지만, 이 말투
학습에는 권장하지 않습니다.

로컬 smoke run 결과, 이 저장소가 실행된 RTX 4050 Laptop 6GB VRAM에서는 최초
멀티모달 로딩 시 GPU 메모리 부족이 발생했습니다. `--offload-embedding`을 켜도
OOM이 나면 실제 E2B 학습은 8GB 이상 GPU 또는 Colab/외부 GPU에서 진행하는 것이
현실적입니다.

스크립트는 기본적으로 8GB 미만 GPU에서 바로 중단합니다. 6GB에서 로딩 가능성을
확인하고 싶다면 `--allow-low-vram --max-seq-length 512 --lora-rank 8`을 함께
전달합니다.

## 검증

학습 후 아래 명령을 실행합니다. 데이터셋 구조, source metadata, LoRA adapter
파일(`adapter_config.json`, `adapter_model.safetensors` 또는 `.bin`)을 검증하고
JSON 리포트를 출력합니다.

```bash
bash validation.sh
```

학습 전 데이터셋만 확인할 때:

```bash
python scripts/finetune/validate_finetune.py --allow-missing-adapter --min-examples 6000
```

## 선택적 export

16-bit 병합 모델:

```powershell
.\runs\finetune-venv\Scripts\python.exe scripts/finetune/train_qlora.py `
  --merge-16bit-output-dir runs/finetune/korean-cute-merged
```

Unsloth GGUF export:

```powershell
.\runs\finetune-venv\Scripts\python.exe scripts/finetune/train_qlora.py `
  --gguf-output-dir models/finetune/korean-cute-gguf `
  --gguf-quantization q4_k_m
```

GGUF export는 로컬 디스크와 메모리를 더 많이 사용합니다. 생성된 GGUF를 운영에
쓰려면 `.env` 또는 실행 옵션의 `ORACLE_LLAMA_MODEL_PATH`를 새 파일로 지정합니다.
