# Data preparation utilities

현재 사용하는 데이터는 Roboflow Version 9 무증강 export와 이를 촬영 세션
단위로 다시 나눈 최종 split이다.

```text
data/training/front_unaugmented/
data/training/front_session_split_v1/
├── train/  # 3,625장
├── valid/  #   404장
└── test/   #   437장
```

유지하는 실행기:

- `create_unaugmented_roboflow_version.py`: 무증강 Roboflow 버전 생성
- `create_leakage_safe_split.py`: 촬영 세션 단위 train/valid/test 생성
- `audit_dataset_split.py`: 증강·중복·세션·COCO 참조 누수 검사
- `fingerprint_coco_dataset.py`: 원본 데이터를 공개하지 않고 COCO 의미·이미지
  바이트가 같은지 비교할 fingerprint 생성

과거 Google Drive calibration/labeled-test 파이프라인과 rear split 목록은
현재 실험에 사용하지 않아 제거했다. 양자화 calibration 표본은 최종 `train`
안에서만 선택한다.

## 노트북·데스크탑 benchmark 동일성 확인

원본 이미지와 annotation 내용을 원격에 올리지 않고도 byte hash가 다른 두 COCO
export가 평가 관점에서 동일한지 확인할 수 있다. 각 컴퓨터에서 다음처럼 보고서를
생성한다.

```bash
python scripts/data_preparation/fingerprint_coco_dataset.py generate \
  --dataset data/training/front_session_split_v1/test \
  --label notebook \
  --output docs/reports/metrics/dataset-fingerprint-notebook.json
```

데스크탑에서는 `--label desktop`과
`docs/reports/metrics/dataset-fingerprint-desktop.json`을 사용한다.
두 보고서가 한 컴퓨터에 모이면 다음 명령으로 비교한다.

```bash
python scripts/data_preparation/fingerprint_coco_dataset.py compare \
  docs/reports/metrics/dataset-fingerprint-desktop.json \
  docs/reports/metrics/dataset-fingerprint-notebook.json \
  --output docs/reports/metrics/dataset-fingerprint-comparison.json
```

`equivalent: true`인 경우 raw annotation SHA-256이 달라도 JSON 순서·직렬화 차이로
설명할 수 있다. `false`이면 기존 정확도 결과를 재사용하지 않는다.
