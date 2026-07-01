# Shared ONNX Models

이 디렉터리의 ONNX 파일은 Git LFS로 공유한다. TensorRT engine은 평가 또는
배포 GPU에서 이 ONNX와 저장소의 실험 설정을 사용해 다시 생성한다.

```bash
git lfs install
git lfs pull
```

모델별 의미, 원본 experiment와 SHA-256은 `manifest.yaml`에서 확인한다.
동일 ONNX를 사용하는 precision/tactic 실험은 파일을 중복 저장하지 않는다.

TensorRT 생성 예:

```bash
.venv/bin/python scripts/experiments/build_candidate.py \
  M02 --camera front --target engine --force
```
