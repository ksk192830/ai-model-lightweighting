# Data preparation utilities

이 디렉터리의 스크립트는 확정된 calibration 및 labeled test 데이터를
다시 생성하거나 생성 과정을 재현할 때만 사용한다.

일반적인 경량화 작업에서는 Google Drive에서 정리된 다음 폴더를
`data/` 아래에 내려받으면 되며, 이 스크립트들을 실행할 필요가 없다.

```bash
.venv/bin/python scripts/data_preparation/download_data.py
```

```text
data/
├── calibration/
│   ├── front/
│   └── rear/
└── labeled_test/
    ├── front/
    └── rear/
```

## Utilities

- `create_calibration_splits.py`: 원본 이미지에서 calibration/test 목록 생성
- `create_labeled_test_splits.py`: 라벨 보유 여부에 따라 test 목록 보정
- `extract_coco_subset.py`: Roboflow COCO export에서 평가 subset 추출
- `materialize_split.py`: split 목록의 이미지를 휴대 가능한 폴더로 복사

모든 명령은 저장소 루트에서 실행한다.

```bash
python3 scripts/data_preparation/create_calibration_splits.py
```
