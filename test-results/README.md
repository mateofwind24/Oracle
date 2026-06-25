# Test Results

테스트 실행 결과 로그와 리포트를 저장하는 디렉터리입니다.

최신 pytest 결과는 `pytest-latest.txt`에 저장합니다.

```bash
python -m pytest 2>&1 | tee test-results/pytest-latest.txt
```
