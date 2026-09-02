# Lightweighting result evaluation

현재 기준 데이터와 모델:

```text
checkpoint: artifacts/training/front-rfdetr-seg-large-v1/checkpoint_best_total.pth
test:       data/training/front_session_split_v1/test
```

## PTH 기준 평가

```bash
.venv/bin/python scripts/evaluation/evaluate_coco_rfdetr_pth.py \
  --checkpoint artifacts/training/front-rfdetr-seg-large-v1/checkpoint_best_total.pth \
  --dataset-dir data/training/front_session_split_v1/test \
  --name front-rfdetr-seg-large-v1-test \
  --device cuda
```

## TensorRT 후보 평가

- `benchmark.py`: latency, FPS, GPU memory와 정확도 측정
- `evaluate_coco_tensorrt.py`: COCO bbox/mask AP 및 mask mIoU 측정
- `benchmark_baseline.py`: PyTorch checkpoint 또는 TensorRT engine 벤치마크
- `visualize_inference_stream.py`: 후보 예측의 시각 비교

모든 engine은 동일한 test 437장, batch 1, confidence와 IoU 기준, warm-up과
반복 횟수를 사용한다. 432 입력 실험은 504 입력 후보와 분리해 보고한다.

원시 결과는 `results/coco-evaluation/`, 표 형식 결과는 `results/`에 저장한다.
