# 노트북 원클릭 실행 안내

이 폴더를 노트북의 로컬 SSD에 복사한 뒤 폴더 안에서 다음 한 줄만 실행한다.

```bash
bash run_notebook_pipeline.sh
```

최초 실행은 Python 3.10 가상환경과 고정 의존성을 설치한다. 이어서 데이터와
ONNX checksum, test 437장, train/test 누수 점검 기록과 디스크 여유 공간을 확인하고 1차 평가 통과
후보 22개를 다음 순서로 자동 처리한다.

```text
TensorRT engine 생성
→ engine 구조 검사
→ 동일 32장으로 warm-up 20회 + 200회 × 3반복 latency 측정
→ 고정 test 437장 COCO bbox/mask 및 semantic mask 평가
→ B01 대비 정확도·속도·메모리·크기 변화 계산
→ 정확도 보존 gate 이후 Pareto 분석
→ 단일 Excel 보고서 생성
```

중단 뒤 같은 명령을 다시 실행하면 완료된 후보는 재사용하고 실패·미완료 후보부터
계속한다. 기존 engine까지 모두 다시 만들려면 다음을 사용한다.

```bash
bash run_notebook_pipeline.sh --force-rebuild
```

설치와 사전 점검만 수행하려면 다음을 사용한다.

```bash
bash run_notebook_pipeline.sh --setup-only
```

다른 터미널에서 진행 상황을 확인한다.

```bash
watch -n 5 '.venv/bin/python scripts/reporting/show_project_status.py'
```

기존 엔진과 437장 정확도 결과는 그대로 검증·재사용하고, 모든 성공 후보의
지연시간만 통제된 환경에서 다시 측정하려면 다음을 실행한다. 각 반복 직전에
CPU/GPU 부하와 GPU 온도를 1초 간격으로 5회 확인한다. 설정된 유휴 범위 및 첫
안정 상태의 온도 범위에 들 때만 측정을 시작하며, 5분 안에 안정화되지 않으면
측정을 강행하지 않고 일시 중단한다.

```bash
set -o pipefail
bash run_notebook_pipeline.sh --remeasure-latency 2>&1 | tee results/controlled-rerun-console.log
```

중단되거나 부하 안정화 대기로 멈춘 뒤에는 위 명령을 그대로 다시 실행한다.
완료 후보는 재사용하고 남은 후보부터 이어서 측정한다. 진행률은 총 63회
(21개 성공 후보 × 3회) 기준이며, 첫 반복이 끝난 뒤 예상 종료 시각을 표시한다.

최종 결과는 `results/stage2-evaluation-report.xlsx` 하나에서 확인한다. 원시 JSON,
CSV와 로그도 `results/`에 그대로 보존되므로 논문 수치의 추적과 재검증이 가능하다.
