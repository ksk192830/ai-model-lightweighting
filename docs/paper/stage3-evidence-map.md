# Stage 3 논문 근거 연결표

> 논문 작성자는 아래 근거가 있는 범위에서만 수치와 결론을 사용한다. 저장소에
> 근거가 없는 항목은 추정하지 않고 자료 필요 상태로 남긴다.

| 논문 내용 | 공식 근거 | 사용 가능한 주장 | 주의사항 |
|---|---|---|---|
| 데이터 분할 | `docs/reports/dataset-split-validity.md`, `docs/reports/metrics/dataset-split-audit.json` | train 3,625 / valid 404 / benchmark 437, 이미지·세션 중복 0 | benchmark는 반복 비교용 |
| 노트북 데이터 동일성 | `docs/reports/metrics/dataset-fingerprint-comparison.json` | 의미적·이미지·최종 fingerprint 일치 | raw JSON byte hash 차이는 직렬화 순서 차이 |
| 기준 모델 학습 | `configs/training/front_rfdetr_seg_large.yaml`, `docs/guides/front-baseline-training.md` | 설정과 18 epoch 조기 종료 | 단일 seed |
| 후보 구성 | `configs/experiments/registry.yaml`, `results/round2-candidate-summary.md` | 전체 26개 후보와 경량화 방식 | W 계열은 연구 범위에서 제거됨 |
| 1차 정적평가 | `results/stage1-static-evaluation.json`, `results/stage1-static-evaluation.md` | 26개 중 22개 통과, 4개 불통 | MAC/FLOP는 해석 가능한 연산의 dense 하한 |
| 평가 기준 | `configs/experiments/defaults.yaml`, `docs/guides/three-stage-candidate-evaluation.md` | 정확도 gate, 3회 지연시간, CV, 30 FPS, P95 정책 | engineering gate이며 통계적 유의성 기준 아님 |
| 측정 부하 통제 | `docs/reports/measurement-load-control.md`, `results/measurement-environment/20260904_104755/` | AC·performance 조건, 63개 부하 gate 통과 | 부하 스트레스 실험과 구분 |
| 2차 후보 결과 | `results/stage2-notebook-summary.json`, `docs/reports/notebook-stage2-results.md` | 21개 완료, Q03 실패, 모든 정확도·속도·크기 수치 | 공식 실행 외 기존 latency와 혼합 금지 |
| Pareto 분석 | `results/stage3-pareto.json`, `docs/reports/stage3-final-analysis.md` | C01·R01 비지배해와 목적별 추천 | 임의 가중합 단일 우승자 없음 |
| 최종 엔진 보존 | `docs/reports/metrics/notebook-final-engine-verification.json` | C01·R01 engine hash 일치 | engine은 노트북에만 보존되는 장비 종속 파일 |
| 종합 표 | `results/stage3-paper-candidate-decisions.csv`, `results/stage2-evaluation-report.xlsx` | 26개 전체 단계별 판정 추적 | Q03에는 성능 수치 없음 |

## 저장소에 아직 없는 자료

- 원본 카메라 모델, 영상 해상도, 프레임 추출 간격, 장소·조명·날씨와 데이터 사용 권한
- 클래스 정의, polygon 경계·가림·잘림 처리 규칙, 라벨링 도구와 검수 이력
- 기존 ASK 2026 논문의 정확한 서지정보와 본 연구와의 연결
- 지원기관, 과제번호와 공식 사사 문구

이 자료가 제공되기 전에는 논문에서 사실처럼 서술하지 않는다.
