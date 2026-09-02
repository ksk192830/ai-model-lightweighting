# Shared ONNX Models

신규 baseline에서 검증을 마친 portable ONNX만 이 디렉터리에 둔다. 과거 PTH와
ONNX는 모두 제거했으며 신규 B01 baseline ONNX부터 다시 등록한다. TensorRT
engine은 평가 또는 배포 GPU에서 ONNX와 저장소의 실험 설정을 사용해 생성한다.

```bash
git lfs install
git lfs pull
```

새 ONNX를 추가할 때 모델 의미, 원본 experiment와 SHA-256을 `manifest.yaml`에
기록한다.
동일 ONNX를 사용하는 precision/tactic 실험은 파일을 중복 저장하지 않는다.
현재 검증·등록된 입력은 B01 504×504, R01 432×432, 6 epoch 복구 학습을
마친 S01 504×504, 8 epoch 복구 학습을 마친 S02 504×504이다. S01 ONNX는
S01 FP32, C01 FP16, C02 INT8 engine의 공통 입력이다. S02는 정확도 gate에서
탈락했지만 구조 축소의 실제 속도 효과를 측정하는 대조군으로 유지한다. PTH와
TensorRT engine은 공유하지 않으며, 아직 복구 중인 M01은 완료 후 검증된
ONNX만 추가한다.

TensorRT 생성 예:

```bash
.venv/bin/python scripts/experiments/build_candidate.py \
  B02 --camera front --target engine --force
```
