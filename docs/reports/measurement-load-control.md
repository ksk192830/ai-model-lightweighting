# Stage 2 지연시간 측정 부하 통제

## 적용 범위

이 프로토콜은 보수적인 절전 전원 조건의 TensorRT 지연시간 측정에 적용한다. 기존 엔진과 437장 정확도 결과는
SHA-256이 일치할 때 재사용하고, Q03을 제외한 엔진 생성 성공 후보 21개의 지연시간을
각 3회씩 총 63회 다시 측정한다.

실행 전에 `powerprofilesctl`을 `power-saver`로 전환한다. platform profile은
`quiet`, AMD P-State EPP는 `power`인지 확인하며 AC 연결도 요구한다. 이 조건의
수치는 성능 우선 모드의 절대 성능으로 해석하지 않고 후보 간 보수적 비교에만
사용한다.

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

측정 도중 전원 정책을 바꾸지 않으며, 한 세션의 63회 측정에는 동일한 최초 안정
구간과 동일한 절전 정책만 사용한다. 다른 전원 정책에서 생성된 부분 결과는 공식
비교에 섞지 않는다.

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

## 전원 조건 결정

논문의 공식 비교는 성공 후보 21개 모두를 `power-saver`/`quiet` 조건에서
평가한다. 통제 재측정과 Pareto를 완료한 뒤 상위 3개만 `performance` 조건에서
별도 배포 검증한다. 성능 우선 결과는 별도 표로 제시하고 공식 Pareto 입력에는
섞지 않는다.
