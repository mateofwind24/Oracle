# Fine-Tune Data

`korean_cute_style_train.jsonl`은 특정 저작물의 대사를 복사하지 않는 합성
chat 데이터입니다. `--allow-network`로 생성하면 Wikimedia 계열 API와 Wikidata에서
제목/라벨 seed만 수집하고 본문이나 실제 대사는 저장하지 않습니다.

재생성:

```powershell
python scripts/finetune/build_korean_cute_dataset.py --allow-network --target-count 6000
```
