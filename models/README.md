# Oracle Local Model

`model.gguf` is assembled from Git LFS part files.

- Source: `unsloth/gemma-4-E2B-it-GGUF`
- File: `gemma-4-E2B-it-UD-IQ2_M.gguf`
- Size: `2,290,858,112` bytes
- SHA256: `60f84cb5b9512175f219506da4a5d98d30b112855c474a3a6f06f6596dc7fd9b`
- License: Apache-2.0
- Default runtime path: `models/model.gguf`

GitHub LFS rejects files above 2 GiB, so the model is stored as:

- `model.gguf.part01`
- `model.gguf.part02`

`./build.sh` and `./run.sh` assemble those parts into `model.gguf` and verify the
SHA256 hash. This is the smallest Gemma 4 E2B GGUF variant found for the
Raspberry Pi deployment target. The default app configuration treats it as a
text-first model and sends capture metadata instead of image bytes. Use a
compatible multimodal GGUF plus projector before setting
`ORACLE_FACE_LLM_SEND_IMAGE=1`.
