# Stage 2 지연시간 측정 부하 통제

## 적용 범위

이 프로토콜은 TensorRT 지연시간 측정에 적용한다. 기존 엔진과 437장 정확도 결과는
SHA-256이 일치할 때 재사용하고, Q03을 제외한 엔진 생성 성공 후보 21개의 지연시간을
각 3회씩 총 63회 다시 측정한다.

## 측정 시작 조건

각 반복 측정 직전에 1초 간격으로 시스템 및 GPU 상태를 표본화한다. 다음 조건을
5개 연속 표본이 모두 충족해야 warm-up과 본 측정을 시작한다.

| 조건 | 상한 |
|---|---:|
| CPU 사용률 | 20% |
| GPU 사용률 | 40% |
| GPU memory-controller 사용률 | 25% |
| GPU 온도 | 65°C |
| 5개 표본의 CPU 사용률 범위 | 10%p |
| 5개 표본의 GPU 사용률 범위 | 8%p |
| 5개 표본의 GPU memory-controller 사용률 범위 | 5%p |
| 5개 표본의 GPU 온도 범위 | 3°C |

AC 전원 연결도 필수다. 첫 번째로 통과한 5개 표본의 평균을 해당 재측정 세션의
기준값으로 고정한다. 이후 반복의 평균은 기준값과 비교해 CPU 10%p, GPU 8%p,
GPU memory-controller 5%p, GPU 온도 5°C 이내여야 한다.

이 노트북은 GNOME 화면 합성과 Codex 화면이 외장 GPU를 지속적으로 사용한다.
프로토콜 적용 전 확인된 상주 GPU 부하는 약 27~31%였다. GPU 0%를 요구하면 측정이
영구 대기하므로, 절대 상한과 세션 기준값을 함께 적용해 상주 부하는 허용하면서
후보별 시작 조건의 차이를 제한한다.
실행 스크립트는 `systemd-inhibit`의 `idle:sleep` 억제를 자동으로 적용해 화면 유휴
전환으로 이 상주 부하가 중간에 사라지지 않게 한다.

현재 재측정 세션의 최초 기준 평균은 CPU 2.72%, GPU 29.0%, GPU
memory-controller 17.4%, GPU 온도 47.0°C다. 실행 시작 시 CPU governor는
`powersave`, 노트북 platform profile은 `quiet`, 전원은 AC 연결 상태였다.

## 기록과 중단 동작

각 반복마다 CPU·메모리 사용률, load average, AC 상태, GPU 사용률, GPU memory 사용,
온도, 소비전력, SM/memory clock과 P-state를 JSON으로 기록한다. 이 기록은 해당
benchmark 원시 JSON에도 연결되며 최종 Excel의 실험환경 시트에 프로토콜과 세션
기준값이 포함된다.

조건이 300초 안에 충족되지 않으면 benchmark를 시작하지 않는다. 후보를 성능 실패로
확정하지 않고 전체 실행을 `paused-load-not-stable` 상태로 끝내며, 같은 명령을 다시
실행하면 완료 후보를 재사용하고 중단 지점부터 진행한다.

진행 상태에는 총 63회 중 완료 횟수, 현재 후보, 반복 번호, 현재 단계, 경과 시간과
예상 종료 시각을 저장한다. 원시 환경 기록은
`results/measurement-environment/<run-id>/<candidate>/repeat-XX.json`, 통제된
benchmark는 `results/benchmarks/notebook-controlled/<run-id>/` 아래에 보존한다.

## 실행과 확인

```bash
set -o pipefail
bash run_notebook_pipeline.sh --remeasure-latency 2>&1 | tee results/controlled-rerun-console.log
```

다른 터미널의 진행 확인 명령은 다음과 같다.

```bash
watch -n 5 '.venv/bin/python scripts/reporting/show_project_status.py'
```

## 후속 performance 조건 평가

현재 21개 후보 전체 재측정은 CPU governor `powersave`, 노트북 platform profile
`quiet`인 조건의 비교 결과로 보존한다. 완료 후 정확도 보존 gate를 통과한 후보 중
지연시간·FPS가 우수한 약 3개를 고른다. 이 후보들만 CPU governor와 platform
profile을 `performance`로 전환해 별도의 성능 우선 평가를 수행한다.

후속 평가에서도 후보별 반복 횟수와 부하 gate를 동일하게 적용하고 실제 governor,
platform profile, 온도, 전력과 클럭을 기록한다. 전원 모드가 다른 두 결과는 하나의
동일 조건 Pareto 표에 섞지 않고 `powersave/quiet` 전체 비교와 `performance` 상위
3개 확인 결과로 분리해 보고한다.
