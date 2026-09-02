# COCO 및 segmentation 평가 결과

파일 하나가 한 모델·backend의 고정 benchmark 평가를 나타낸다. bbox/mask
COCO AP와 semantic mIoU, 이미지 수, threshold, 모델·annotation provenance를
함께 기록한다.

현재 논문 비교의 기준 파일은 `front-rfdetr-seg-large-v1-test.json`이다.
`*-invalid-*`, `*-before-recovery.*` 파일은 진단용이며 최종 결과표의 근거로
사용하지 않는다.
