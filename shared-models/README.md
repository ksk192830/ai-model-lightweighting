# Shared ONNX Models

1차 정적평가를 통과해 노트북 2차 평가 대상이 된 portable ONNX를 이 디렉터리에
둔다. 22개 engine recipe가 공유하는 중복 제거 ONNX 17개를 Git LFS로 관리한다.
TensorRT engine은 평가 GPU에서 이 ONNX와 저장소의 실험 설정을 사용해 생성한다.

```bash
git lfs install
git lfs pull
```

새 ONNX를 추가할 때 모델 의미, 원본 experiment, 바이트 크기와 SHA-256을
`manifest.yaml`에 기록한다. 동일 ONNX를 사용하는 precision/tactic 실험은 파일을
중복 저장하지 않는다. 현재 manifest는 1차 통과 22개 후보의 source ID 17개를
전부 포함한다. PTH, TensorRT engine과 데이터셋은 공유하지 않는다.

새 clone에서 전체 노트북 묶음 준비 상태를 확인한다.

```bash
git lfs pull
.venv/bin/python scripts/experiments/package_notebook_bundle.py \
  --suite stage1 --output-dir delivery/notebook-front-stage1 --dry-run
```

TensorRT 생성 예:

```bash
.venv/bin/python scripts/experiments/build_candidate.py \
  B02 --camera front --target engine --force
```
