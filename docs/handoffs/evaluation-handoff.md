# 모델 경량화 평가 작업 인수인계

이 문서는 다음 작업자가 저장소의 현재 상태를 빠르게 파악하고, 평가를
재현하거나 논문 산출물을 갱신할 수 있도록 만든 운영 문서다. 평가 지표의
정의와 해석은 [평가 개념 정리](../concepts/evaluation-metrics.md)를 참고한다.

## 1. 현재 상태

- 완료일: 2026-07-04 (KST)
- 반영 커밋: `c7c9bec`
- 원격 브랜치: `main`
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU 6 GB
- 데이터셋: `data/labeled_test/front`
- 평가 이미지: 296장
- 정답 객체: 423개
- 평가 대상: RF-DETR TensorRT 8개 + Ultralytics PT 1개
- bbox, segmentation, 속도, 지연시간, 크기, GPU 메모리 평가 완료
- 후방 카메라는 이번 범위에 포함되지 않음

## 2. 파일별 역할

| 경로 | 역할 |
|---|---|
| `results/paper_metrics.csv` | 속도와 정확도를 모두 보존한 원본 결과 |
| `results/results_final.csv` | 논문용 RF-DETR 후보 8개 정리본 |
| `results/coco-evaluation/*-front.json` | 모델별 segmentation 상세 결과 |
| `scripts/evaluation/benchmark.py` | 속도·지연시간·메모리 통합 benchmark |
| `scripts/evaluation/evaluate.py` | bbox 정확도 및 클래스 이름 매핑 평가 |
| `scripts/evaluation/evaluate_coco_tensorrt.py` | TensorRT mask 평가 |
| `scripts/evaluation/evaluate_coco_ultralytics.py` | Ultralytics mask 평가 |
| `scripts/reporting/paper_results.py` | 최종 CSV와 `figures/` 재생성 |
| `docs/handoffs/gpu-evaluation-prompt.md` | GPU에서 평가를 다시 수행하는 상세 절차 |
| `docs/reports/results-draft.md` | 논문 Experimental Results 초안 |

## 3. 평가 조건

| 항목 | 값 |
|---|---|
| batch size | 1 |
| 운영 confidence | 0.25 |
| AP confidence floor | 0.001 |
| NMS IoU | 0.7 |
| 평가 API | `pycocotools COCOeval` |
| 속도 warm-up | 10회 |
| timed runs | 100회 × 3 blocks |
| 정확도 이미지 | 전체 296장 |
| B01~S01 입력 | 504×504 |
| R01 입력 | 432×432 |
| `parking_front.pt` 입력 | 512×512 |

속도 결과를 다시 측정할 때는 AC 전원과 GPU 성능 모드를 고정하고 다른 GPU
프로세스를 최소화한다. 정확도만 갱신할 때는 기존 FPS, latency, 메모리,
모델 크기 열을 변경하지 않는다.

## 4. 중요한 데이터 규칙

COCO 정답에는 배경 범주 `0: front`가 있고 실제 객체 범주는
`1: out_line`, `2: parking_lot`, `3: parking_space`다.
`parking_front.pt` 출력은 같은 이름을 `0, 1, 2`로 사용한다.

평가기에서 반드시 다음 이름 기반 매핑이 적용되어야 한다.

```text
{0: 1, 1: 2, 2: 3}
```

이 매핑 전의 `parking_front.pt` mAP50 `0.0084`, mAP50-95 `0.0017`은
무효 수치다. 수정 후 값은 mAP50 `0.9070`, mAP50-95 `0.7130`이다.

## 5. 최종 결과

| ID | bbox mAP50-95 | Mask AP | Mask mIoU | FPS | 크기(MB) |
|---|---:|---:|---:|---:|---:|
| B01 | 0.8310 | 0.6309 | 0.7931 | 18.91 | 129.28 |
| B02 | 0.8333 | 0.6313 | 0.7994 | 32.65 | 67.90 |
| B03 | 0.8352 | 0.6306 | 0.7992 | 31.86 | 67.98 |
| C01 | 0.8378 | 0.6441 | 0.8044 | 32.33 | 64.70 |
| M01 | 0.7961 | 0.6160 | 0.7997 | 31.64 | 67.93 |
| M02 | 0.8005 | 0.6171 | 0.7962 | 31.91 | 62.40 |
| S01 | 0.8390 | 0.6460 | 0.8042 | 19.78 | 123.28 |
| R01 | 0.8239 | 0.6237 | 0.7869 | 36.10 | 68.05 |
| parking_front | 0.7130 | 0.3065 | 0.6045 | 36.87 | 5.70 |

결론은 다음과 같다.

- 일반 배포 후보: C01
- 단순하고 안정적인 경량화: B02
- 정확도 최우선: S01
- RF-DETR 속도 최우선: R01
- 크기 최우선 참고 모델: `parking_front.pt`
- B03은 B02 대비 INT8 추가 이득이 작음
- M01/M02는 크기·속도 대비 정확도 손실이 큼

`parking_front.pt`는 그래프에 참고 모델로 표시하지만 RF-DETR 파생 모델이
아니므로 `results/results_final.csv`의 메인 경량화 비교에는 포함하지 않는다.

## 6. 재생성 및 검증

결과 CSV와 모든 그래프 재생성:

```bash
.venv/bin/python scripts/reporting/paper_results.py
```

정확도 평가 관련 테스트:

```bash
.venv/bin/python tests/test_segmentation_metrics.py
.venv/bin/python -m pytest -q tests/test_evaluate_metrics.py
```

GPU에서 정확도를 완전히 다시 평가해야 한다면
[gpu-evaluation-prompt.md](gpu-evaluation-prompt.md)의 명령을 따른다.

## 7. 산출물 관리 규칙

- 커밋 가능: 평가 코드, 테스트, CSV, JSON, 문서, `figures/*.png`
- 커밋 금지: `.pth`, `.pt`, `.engine` 등 모델·엔진 바이너리
- `results/`는 기본적으로 ignore되므로 새 JSON/CSV를 커밋할 때는 대상 파일을
  명시적으로 검토한 뒤 `git add -f <path>`를 사용한다.
- bbox나 속도 측정을 다시 하지 않았다면 기존 열이 바뀌지 않았는지 비교한다.
- JSON의 `image_count`가 모델별로 296인지 확인한다.

## 8. 남은 작업

- 후방 카메라 모델을 논문 범위에 포함할 경우 동일 조건으로 추가 평가
- 실제 배포 하드웨어에서 latency와 메모리 재검증
- ~~필요하면 mask AP와 FPS/크기의 Pareto 그래프 추가~~ — 2026-07-05 완료
  (`figures/pareto_mask_fps.png`, `figures/pareto_mask_size.png`; bbox 기준
  front와 구성 동일함을 확인, `0705.md` §4 참고)
- 논문 본문에는 입력 크기와 백엔드가 다른 YOLO를 참고 모델로 명시
