# 모델 경량화 평가 작업 인수인계

## 1. 목적과 현재 상태

전방 주차 객체 탐지 모델의 경량화 후보를 노트북 환경에서 동일한 절차로
평가하고, 논문 자동화 담당자가 표·그래프·Results 초안을 만들 수 있는 원본
CSV를 생성했다.

- 평가 완료일: 2026-07-04 (KST)
- 평가 대상: 전방 모델 9개
- 평가 성공: 9/9
- 평가 코드와 결과 반영 커밋: `9ef9601`
- 원격 브랜치: `main`

후방 카메라 모델은 이번 논문용 평가 범위에 포함하지 않았다. 논문이 전·후방
모델을 모두 다룬다면 후방 엔진 생성과 동일 조건 재평가가 추가로 필요하다.

## 2. 주요 파일

| 경로 | 용도 |
|---|---|
| `scripts/benchmark.py` | 속도, 지연시간, 정확도, 크기, 메모리 통합 측정 |
| `scripts/evaluate.py` | 데이터 로딩, 추론 백엔드, COCO 정확도 평가 |
| `results/paper_metrics.csv` | 논문용 조건으로 재측정한 최종 원본 결과 |
| `results/results_raw.csv` | 초기 단기 측정 결과 |
| `tests/test_evaluate_metrics.py` | COCOeval 및 confidence 분리 검증 |
| `models/parking_front.pt` | 기존 Ultralytics 전방 모델 |
| `artifacts/experiments/*/front/model.engine` | TensorRT 경량화 엔진 |

논문 자동화 담당자는 `results/paper_metrics.csv`를 우선 사용해야 한다.
`results/results_raw.csv`는 측정 횟수와 평가 방식이 이전 버전이므로 참고용이다.

## 3. 평가 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU 메모리 | 6 GB |
| OS | Ubuntu 22.04 / Linux 6.8 |
| Python | 3.10.12 |
| PyTorch | 2.12.1+cu130 |
| TensorRT | 10.16.1.11 |
| 데이터셋 | `data/labeled_test/front` |
| 평가 이미지 | 296장 |
| 정답 객체 | 423개 |

노트북은 AC 전원을 연결하고 동일 성능 모드를 유지해야 한다. 다른 GPU 작업을
동시에 실행하면 FPS와 P95 지연시간 비교가 무효화될 수 있다.

## 4. 논문용 평가 조건

- 모델은 병렬이 아닌 순차 실행
- batch size: 1
- 운영 confidence: 0.25
- AP 평가 confidence floor: 0.001
- NMS IoU: 0.7
- metric backend: 공식 `pycocotools COCOeval`
- warm-up: 10회
- timed runs: 100회
- timing repetition blocks: 3회
- FPS 표본: 앞 32장
- 정확도: 전체 296장
- TensorRT 엔진 입력: 504×504
- R01 입력: 432×432
- Ultralytics `parking_front.pt` 입력: 512×512

R01은 입력 해상도 감소 효과를 보는 별도 실험이다. `parking_front.pt`는
Ultralytics stride 제약 때문에 504가 아닌 가장 가까운 유효 크기 512를
사용했다. 두 모델은 504×504 TensorRT 후보와 완전히 동일한 입력 조건이
아니므로 표와 그래프에 입력 크기를 반드시 표시한다.

## 5. 재현 명령

504×504 TensorRT 후보:

```bash
.venv/bin/python scripts/benchmark.py \
  --model artifacts/experiments/B01/front/model.engine \
  --model artifacts/experiments/B02/front/model.engine \
  --model artifacts/experiments/B03/front/model.engine \
  --model artifacts/experiments/C01/front/model.engine \
  --model artifacts/experiments/M01/front/model.engine \
  --model artifacts/experiments/M02/front/model.engine \
  --model artifacts/experiments/S01/front/model.engine \
  --baseline-model models/parking_front.pth \
  --data data/labeled_test/front \
  --imgsz 504 \
  --batch-size 1 \
  --conf 0.25 \
  --eval-conf 0.001 \
  --iou 0.7 \
  --metric-backend coco \
  --device cuda:0 \
  --backend rfdetr_engine \
  --warmup-runs 10 \
  --timed-runs 100 \
  --repetitions 3 \
  --speed-samples 32 \
  --output results/paper_metrics.csv
```

R01은 `--model artifacts/experiments/R01/front/model.engine --imgsz 432`,
기존 Ultralytics 모델은
`--model models/parking_front.pt --backend ultralytics --imgsz 512`로 실행한다.

기존 CSV에 재실행 결과를 추가하면 중복 행이 생긴다. 완전 재평가 시 기존
파일을 보관하거나 새 출력 경로를 사용한다.

## 6. 최종 측정 결과

| 모델 | 방식 | 입력 | 크기(MB) | FPS | FPS 표준편차 | P95(ms) | mAP50 | mAP50-95 | GPU Peak(MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B01 | TensorRT FP32 | 504 | 129.28 | 18.91 | 0.02 | 56.86 | 0.9836 | 0.8310 | 486 |
| B02 | TensorRT FP16 | 504 | 67.90 | 32.65 | 0.02 | 31.60 | 0.9857 | 0.8333 | 252 |
| B03 | TensorRT INT8 PTQ | 504 | 67.98 | 31.86 | 0.03 | 32.88 | 0.9853 | 0.8352 | 252 |
| C01 | 구조적 경량화 FP16 | 504 | 64.70 | 32.33 | 0.05 | 32.14 | 0.9886 | 0.8378 | 250 |
| M01 | 2:4 희소화 대조군 | 504 | 67.93 | 31.64 | 0.02 | 33.01 | 0.9700 | 0.7961 | 252 |
| M02 | 2:4 희소화 | 504 | 62.40 | 31.91 | 0.06 | 32.87 | 0.9734 | 0.8005 | 252 |
| S01 | Decoder layer 축소 | 504 | 123.28 | 19.78 | 0.01 | 52.11 | 0.9877 | 0.8390 | 486 |
| R01 | 해상도 축소 FP16 | 432 | 68.05 | 36.10 | 0.02 | 29.52 | 0.9792 | 0.8239 | 238 |
| parking_front.pt | Ultralytics PyTorch | 512 | 5.70 | 36.87 | 0.04 | 29.65 | 0.0084 | 0.0017 | 216 |

해석:

- 동일 504 조건의 종합 균형 후보는 C01이다.
- 동일 504 조건에서 최고 FPS는 B02, 최고 mAP50-95는 S01이다.
- R01은 가장 빠른 TensorRT 후보지만 해상도 감소에 따른 정확도 손실이 있다.
- M01과 M02는 속도·크기 이점 대비 정확도 손실이 크다.
- `parking_front.pt`는 클래스 이름은 일치하지만 현재 테스트셋 정확도가
  비정상적으로 낮다. 원인 규명 전에는 속도 참고용으로만 사용한다.

## 7. 논문 자동화 담당 후속 작업

Notion의 `3. 논문 자동화 담당` 페이지 기준으로 다음 작업이 남아 있다.

1. `paper_metrics.csv`에서 논문용 열을 추출해 `results_final.csv` 생성
2. 모델 이름과 입력 크기 표기 통일
3. B01 대비 크기 감소율, FPS 증가율, latency 감소율, mAP 변화량 계산
4. 성능 비교표와 경량화 효과표 작성
5. FPS, latency, size, mAP 그래프 작성
6. mAP-FPS, mAP-size, mAP-latency Pareto 그래프 작성
7. `4. Experimental Results` 초안 작성

그래프와 표에서는 R01과 `parking_front.pt`의 입력 크기 및 백엔드 차이를
명시한다. `parking_front.pt`의 낮은 정확도를 이상치로 숨기지 말고 별도
분석 대상으로 표시한다.

## 8. 검증

```bash
env PYTHONPATH=. .venv/bin/pytest -q
```

결과: `9 passed`.

실제 RTX 4050에서 TensorRT/COCO 통합 smoke test도 통과했다.
