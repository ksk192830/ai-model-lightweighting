# 학습 데이터 구조

`front_unaugmented/`는 Roboflow 무증강 원본이고, `front_session_split_v1/`은
이를 촬영 세션 단위로 다시 배정한 논문 실험용 split이다. 이미지와 annotation
파일 자체는 Git에 포함하지 않는다.

새 버전을 만들면 기존 폴더를 덮어쓰지 말고 별도 이름으로 생성한 뒤
`scripts/data_preparation/audit_dataset_split.py`를 통과한 경우에만
`configs/dataset.yaml`의 기준 경로를 바꾼다.
