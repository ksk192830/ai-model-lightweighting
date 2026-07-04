# GPU 노트북 AI 전달 프롬프트

아래 저장소의 최신 원격 브랜치를 받아, 기존 bbox·속도 측정을 반복하지 않고
누락된 segmentation 성능과 잘못 평가된 Ultralytics 정확도만 재측정해 주세요.

## 목표

1. RF-DETR TensorRT 후보 8개의 `Mask AP`, `Mask AP50`, `Mask AP75`,
   `Mask mIoU`를 측정해 기존 `results/paper_metrics.csv` 행에 추가합니다.
2. `models/parking_front.pt`는 기존 속도 결과를 보존하고, class-ID 매핑을
   수정한 bbox·segmentation 정확도를 다시 측정해 동일 CSV 행을 갱신합니다.
3. 논문용 CSV와 그래프를 재생성하고 결과의 타당성을 점검합니다.

## 중요한 배경

- COCO 정답에는 빈 배경 범주 `0: front`가 있고 실제 객체 ID는
  `1: out_line`, `2: parking_lot`, `3: parking_space`입니다.
- `parking_front.pt`의 출력 ID는 `0, 1, 2`입니다.
- 수정된 평가기는 클래스 이름을 사용해 `{0: 1, 1: 2, 2: 3}`을 자동
  적용합니다. 실행 로그에 이 매핑이 나타나는지 확인하세요.
- `parking_front.pt`는 RF-DETR 경량화 후보가 아니므로 재평가 결과와
  관계없이 메인 표·Pareto 분석에서는 제외하고 참고 결과로만 둡니다.

## 실행 전 확인

```bash
git status --short
git pull --ff-only
python --version
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import tensorrt, pycocotools; print(tensorrt.__version__)"
```

의존성이 없다면 저장소의 `requirements-tensorrt.txt`에 맞춰 설치하세요.
다음 파일이 모두 있는지 확인하세요.

```text
data/labeled_test/front/_annotations.coco.json
models/parking_front.pt
artifacts/experiments/{B01,B02,B03,C01,M01,M02,S01,R01}/front/model.engine
```

## 1. Segmentation 지표만 추가

아래 작업은 mask 추론과 COCO `segm` 평가만 수행합니다. bbox AP 및 latency
benchmark는 다시 실행하지 않습니다.

```bash
for experiment in B01 B02 B03 C01 M01 M02 S01 R01; do
  python scripts/evaluation/evaluate_coco_tensorrt.py \
    --experiment "$experiment" \
    --camera front \
    --csv results/paper_metrics.csv
done
```

각 실행 후 다음을 확인하세요.

- `results/coco-evaluation/<ID>-front.json` 생성
- CSV에 `Mask AP`, `Mask AP50`, `Mask AP75`, `Mask mIoU` 열 생성
- 해당 모델 행에 네 값이 채워짐
- 기존 FPS, latency, bbox mAP 열은 변경되지 않음
- AP 값이 `-1`, NaN 또는 빈 값이 아님

## 2. parking_front.pt bbox·segmentation 정확도 재평가

```bash
python scripts/evaluate.py \
  --model models/parking_front.pt \
  --backend ultralytics \
  --data data/labeled_test/front \
  --imgsz 512 \
  --batch-size 1 \
  --conf 0.25 \
  --eval-conf 0.001 \
  --iou 0.7 \
  --device cuda:0 \
  --metric-backend coco \
  --output results/paper_metrics.csv \
  --update-existing

python scripts/evaluation/evaluate_coco_ultralytics.py \
  --model models/parking_front.pt \
  --dataset-dir data/labeled_test/front \
  --imgsz 512 \
  --threshold 0.001 \
  --miou-threshold 0.25 \
  --iou 0.7 \
  --device cuda:0 \
  --csv results/paper_metrics.csv
```

다음을 확인하세요.

- 적용 매핑이 `{0: 1, 1: 2, 2: 3}`임
- `parking_front.pt` 행에서 Precision, Recall, mAP50, mAP50-95,
  Predictions와 Mask AP, Mask AP50, Mask AP75, Mask mIoU가 갱신됨
- 기존 FPS, latency, 메모리, 모델 크기는 그대로임
- 기존 무효 수치인 mAP50 `0.0084`, mAP50-95 `0.0017`이 교체됨

## 3. 논문 산출물 재생성 및 검증

```bash
python scripts/paper_results.py
python -m unittest tests/test_segmentation_metrics.py
python -m pytest -q tests/test_evaluate_metrics.py
```

검증 항목:

- `results/results_final.csv`에는 RF-DETR TensorRT 후보 8개만 존재
- `parking_front`는 메인 CSV, 정확도 그래프, Pareto 그래프에 없음
- `results/paper_metrics.csv`에는 참고용 원본 행이 유지됨
- 표와 그래프에 사용된 bbox 수치는 기존 값과 동일함
- segmentation JSON 8개의 이미지 수가 각각 296장임

## 4. 결과 보고

최종 답변에 아래를 표로 정리하세요.

- 모델 ID
- Mask AP / AP50 / AP75 / mIoU
- 기존 bbox mAP50-95
- 평가 이미지 수

그리고 `parking_front.pt`의 수정 전·후 Precision, Recall, mAP50,
mAP50-95를 별도 표로 비교하세요. 오류나 누락이 있으면 추측으로 값을
채우지 말고, 실패한 명령과 오류 메시지 및 필요한 조치를 보고하세요.

검증이 모두 통과하면 변경된 결과 CSV, JSON, 논문 산출물, 그래프만
커밋하고 모델·엔진 바이너리는 커밋하지 마세요.
