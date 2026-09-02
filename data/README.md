# 로컬 데이터

이미지와 COCO annotation은 용량·접근권한·재현성 문제 때문에 Git에 포함하지
않는다. 이 README만 저장소 구조 설명을 위해 추적한다.

```text
data/training/front_unaugmented/       # Roboflow v9 무증강 원본 export
data/training/front_session_split_v1/
├── train/                             # 3,625장
├── valid/                             #   404장
└── test/                              #   437장 비교용 benchmark
```

재구성은 `scripts/data_preparation/`을, 분할 근거와 검증 수치는
`docs/reports/dataset-split-validity.md`를 따른다. 양자화 calibration은 test가
아닌 train에서 seed 42로 선택한 128장만 사용한다.
