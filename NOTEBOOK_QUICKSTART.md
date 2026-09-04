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
watch -n 30 '.venv/bin/python scripts/reporting/show_project_status.py'
```

기존 엔진과 437장 정확도 결과는 그대로 검증·재사용하고, 모든 성공 후보의
지연시간만 성능 우선 환경에서 다시 측정하려면 다음을 실행한다. 스크립트가
`powerprofilesctl` 프로필을 `performance`로 전환하고 실제 platform profile과
CPU EPP가 모두 `performance`인지 검사한다. 각 반복 직전에
첫 측정 전 화면 갱신이 멈추도록 30초를 기다린 뒤 CPU/GPU 부하와 GPU 온도를
1초 간격으로 3회 확인한다. 설정된 유휴 범위 및 첫
안정 상태의 온도 범위에 들 때만 측정을 시작한다. 5분 안에 안정화되지 않으면
측정을 강행하지 않고 사유를 기록한 뒤 자동으로 다음 안정화 확인을 계속한다.

```bash
set -o pipefail
bash run_notebook_pipeline.sh --remeasure-latency 2>&1 | tee results/controlled-rerun-console.log
```

실행 스크립트는 측정 중 화면 유휴 전환과 시스템 절전을 자동으로 막는다. 화면이
꺼지면서 GNOME 합성 부하가 사라져 세션 기준이 바뀌는 일을 방지하기 위해서다.

사용자가 중단하거나 시스템이 재시작된 뒤에는 위 명령을 그대로 다시 실행한다.
완료 후보는 재사용하고 남은 후보부터 이어서 측정한다. 부하 대기는 자동 재시도한다.
진행률은 최초 총 63회
(21개 성공 후보 × 3회) 기준이며, 첫 반복이 끝난 뒤 예상 종료 시각을 표시한다.
후보의 반복 median CV가 5%를 초과하면 해당 후보만 지연시간 3회를 한 번 더
측정하며, 진행률 총량도 3회만큼 자동 증가한다. 재측정이 수행되면 두 번째 3회를
공식값으로 사용하고 첫 3회도 근거로 보존한다.

최종 Pareto와 성능 순위에는 21개 후보의 동일 성능 우선 세션 결과만 사용한다.
과거 절전 모드 부분 결과는 공식 비교에 섞지 않는다.
정확도 gate 통과 후 Mask AP·median latency·engine 크기의 3축 Pareto를 계산하며,
그중 median latency 33.33 ms 이하인 후보를 30 FPS 배포 후보로 따로 표시한다.
P95가 33.33 ms를 넘으면 tail-latency 경고만 표시하며 후보를 탈락시키지 않는다.
지연시간 재측정 후에도 median CV가 5%를 넘으면 Pareto에는 유지하되 최종 권장
모델에서는 제외한다.

최종 결과는 `results/stage2-evaluation-report.xlsx` 하나에서 확인한다. 원시 JSON,
CSV와 로그도 `results/`에 그대로 보존되므로 논문 수치의 추적과 재검증이 가능하다.

데스크탑 데이터와 평가 데이터의 동일성을 확인할 때는 원본 데이터 대신 다음
fingerprint 보고서만 생성해 원격에 올린다.

```bash
.venv/bin/python scripts/data_preparation/fingerprint_coco_dataset.py generate \
  --dataset data/training/front_session_split_v1/test \
  --label notebook \
  --output docs/reports/metrics/dataset-fingerprint-notebook.json
```
