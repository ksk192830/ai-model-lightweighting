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

과거 Google Drive calibration/labeled-test 파이프라인과 rear split 목록은
현재 실험에 사용하지 않아 제거했다. 양자화 calibration 표본은 최종 `train`
안에서만 선택한다.
