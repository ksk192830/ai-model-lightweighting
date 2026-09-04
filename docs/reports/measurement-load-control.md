# Stage 2 지연시간 측정 부하 통제

## 적용 범위

이 프로토콜은 성능 우선 전원 조건의 TensorRT 지연시간 측정에 적용한다. 기존 엔진과 437장 정확도 결과는
SHA-256이 일치할 때 재사용하고, Q03을 제외한 엔진 생성 성공 후보 21개의 지연시간을
각 3회씩 최초 총 63회 다시 측정한다. 후보별 반복 median CV가 5%를 초과하면 해당
후보의 지연시간 3회만 한 번 더 측정한다.

실행 전에 `powerprofilesctl`을 `performance`로 전환한다. platform profile과 AMD
P-State EPP가 모두 `performance`인지 확인하며 AC 연결도 요구한다. AMD P-State
EPP 드라이버에서는 scaling governor 문자열이 `powersave`로 남을 수 있으므로,
governor 이름만으로 전원 모드를 판정하지 않는다.

## 측정 시작 조건

첫 반복 전에는 실행 직후 화면 갱신이 GPU 기준값에 포함되지 않도록 30초 동안
정착시킨다. 이후 각 반복 측정 직전에 1초 간격으로 시스템 및 GPU 상태를
표본화한다. 다음 조건을 3개 연속 표본이 모두 충족해야 warm-up과 본 측정을 시작한다.

| 조건 | 상한 |
|---|---:|
| CPU 사용률 | 30% |
| GPU 사용률 | 50% |
| GPU memory-controller 사용률 | 50% |
| GPU 온도 | 75°C |
| 3개 표본의 CPU 사용률 범위 | 20%p |
| 3개 표본의 GPU 사용률 범위 | 50%p |
| 3개 표본의 GPU memory-controller 사용률 범위 | 50%p |
| 3개 표본의 GPU 온도 범위 | 6°C |

AC 전원 연결도 필수다. 첫 번째로 통과한 3개 표본의 평균을 해당 재측정 세션의
기준값으로 고정한다. 이후 반복의 평균은 기준값과 비교해 CPU 20%p, GPU 50%p,
GPU memory-controller 50%p, GPU 온도 15°C 이내여야 한다. 이 완화 기준은 장시간
대기를 피하면서 지속적인 고부하 상태에서 측정을 시작하는 것은 막는 수준이다.

이 노트북은 화면이 갱신되는 순간 GNOME과 Codex가 외장 GPU를 약 27~33% 사용하지만,
화면이 정지하면 GPU 사용률은 0% 부근으로 내려간다. 실행 직후 값을 바로 기준으로
삼으면 이후 실제 유휴 상태를 오히려 거부하므로 30초 정착 후의 낮은 부하를 기준으로
사용한다. 진행 확인 화면도 30초 간격으로 갱신해 안정 표본 사이의 간섭을 줄인다.
실행 스크립트는 `systemd-inhibit`의 `idle:sleep` 억제를 자동으로 적용해 화면 유휴
전환으로 이 상주 부하가 중간에 사라지지 않게 한다.

앞서 `power-saver`/`quiet` 상태에서 생성된 부분 재측정값은 최종 비교에서 제외한다.
성능 우선 재측정은 새 run ID와 새 최초 안정 구간을 사용해 0/63회부터 시작한다.

## 기록과 중단 동작

각 반복마다 CPU·메모리 사용률, load average, AC 상태, GPU 사용률, GPU memory 사용,
온도, 소비전력, SM/memory clock과 P-state를 JSON으로 기록한다. 이 기록은 해당
benchmark 원시 JSON에도 연결되며 최종 Excel의 실험환경 시트에 프로토콜과 세션
기준값이 포함된다.

조건이 300초 안에 충족되지 않으면 benchmark를 시작하지 않는다. 후보를 성능 실패로
확정하지 않고 거부 원인과 표본을 기록한 뒤 다음 300초 안정화 확인을 자동으로
시작한다. 사용자가 중단하거나 시스템이 재시작된 경우 같은 명령을 실행하면 완료
후보를 재사용하고 중단 후보부터 진행한다.

진행 상태에는 최초 총 63회 중 완료 횟수, 현재 후보, 반복 번호, 현재 단계, 경과
시간과 예상 종료 시각을 저장한다. CV 재측정이 시작되면 총량을 후보당 3회씩
자동 증가시킨다. 첫 라운드와 재측정 라운드를 모두 보존하고 두 번째 라운드를 공식
지연시간으로 사용한다. 재측정 후에도 CV가 5%를 넘으면 Pareto에는 유지하되 최종
권장 후보에서는 제외한다. 원시 환경 기록은
`results/measurement-environment/<run-id>/<candidate>/repeat-XX.json`, 통제된
benchmark는 `results/benchmarks/notebook-controlled/<run-id>/` 아래에 보존한다.

## 실행과 확인

```bash
set -o pipefail
bash run_notebook_pipeline.sh --remeasure-latency 2>&1 | tee results/controlled-rerun-console.log
```

다른 터미널의 진행 확인 명령은 다음과 같다.

```bash
watch -n 30 '.venv/bin/python scripts/reporting/show_project_status.py'
```

## 전원 조건 결정

논문의 공식 비교는 성공 후보 21개 모두를 `performance` 조건에서 평가한다.
`power-saver`/`quiet` 부분 결과와 성능 우선 결과를 섞지 않는다. 최종 Pareto와
성능 순위에는 새 성능 우선 세션의 공식 측정 라운드만 사용한다.
