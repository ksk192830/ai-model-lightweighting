# 보고서 근거 지표

데이터 분할 및 보고서가 참조하는 작은 CSV/JSON 근거를 둔다. 각 파일은
원본 데이터 대신 이미지 수, class 분포, 세션 배정, 중복·참조 감사 결과만
포함한다.

`dataset-split-audit.json`이 `paper_ready: true`인지 확인한 뒤 학습을 시작한다.
데이터를 다시 분할하면 이 폴더의 지표와 `dataset-split-validity.md`, 관련
그림을 함께 재생성한다.

`dataset-fingerprint-desktop.json`과 `dataset-fingerprint-notebook.json`은
원본 데이터 대신 평가 의미·이미지 byte collection의 hash만 저장한다. 두 환경의
보고서는 `fingerprint_coco_dataset.py compare`로 비교하고 결과를
`dataset-fingerprint-comparison.json`에 보존한다.
