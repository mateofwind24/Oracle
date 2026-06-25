# Oracle Local Model

The GGUF models are downloaded by `./build.sh` only when no usable `*.gguf`
model already exists under this `models/` directory. `./run.sh` also reuses an
existing repo model before attempting any download.

## Default Runtime Model

- Source: `unsloth/gemma-4-E2B-it-GGUF`
- File: `gemma-4-E2B-it-UD-Q2_K_XL.gguf`
- URL: `https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-UD-Q2_K_XL.gguf`
- Size: `2,403,612,800` bytes
- SHA256: `dd279a54c0c0dc9724ed11d7f73ad7fb4489a45f58fefe9447da2429a727de0c`
- License: Gemma
- Path: `models/gemma-4-E2B-it-UD-Q2_K_XL.gguf`

The model files are not committed to this repository. `*.tmp` files are used for
resumable downloads, and completed files are moved into place after SHA256
verification. If a matching GGUF file is already present, the scripts verify the
known packaged hashes and skip downloading. The default app configuration treats
these as text-first models and sends capture metadata instead of image bytes.
Use a compatible multimodal GGUF plus projector before setting
`ORACLE_FACE_LLM_SEND_IMAGE=1`.
