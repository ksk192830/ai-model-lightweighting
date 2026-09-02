# 학습 설정

`front_rfdetr_seg_large.yaml`은 논문 baseline 재학습에 사용한 설정이다. 데이터
경로·class 순서·resolution과 optimizer, effective batch, early stopping,
seed를 함께 고정한다.

학습 진입점은 `scripts/training/train_rfdetr_front.py`이며 결과는
`artifacts/training/<run-name>/`에 생성된다. checkpoint 선택에는 valid만
사용하고, test 평가는 학습 종료 후 별도 실행한다.
