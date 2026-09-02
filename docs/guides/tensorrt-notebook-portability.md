# NVIDIA 노트북에서 TensorRT engine 생성·검증

TensorRT engine은 생성한 GPU, CUDA와 TensorRT 버전에 종속된다. 저장소에서는
portable ONNX를 만들고, 실제 평가에 사용할 노트북에서 engine을 생성한다.

Google Colab에서는
[`notebooks/front_tensorrt_ready4.ipynb`](../../notebooks/front_tensorrt_ready4.ipynb)를
열고 위에서 아래로 실행한다. 이 노트북은 `notebook-front-ready4.tar.gz` 업로드,
고정 SHA-256 확인, 안전한 압축 해제, 환경 검증, ready4 엔진 순차 생성, test
437장 평가와 결과 다운로드까지 포함한다. B03 보정은 묶음에 고정된 train 128장만
사용하며, engine 출력 query 수와 후처리 `num_select`가 일치하는지도 smoke test에서
확인한다.

## 전달 대상

```text
shared-models/<candidate>.onnx
shared-models/manifest.yaml
configs/experiments/
scripts/lightweighting/
scripts/evaluation/
data/training/front_session_split_v1/test/
```

INT8·FP8 calibration이 필요한 경우 test가 아니라 다음 train 경로를 사용한다.

```text
data/training/front_session_split_v1/train/
```

패키징 도구는 `shared-models/manifest.yaml`에 SHA-256과 크기가 등록된 ONNX만
받는다. 로컬 `artifacts/experiments/`에 ONNX가 있더라도 recovery fine-tuning이
끝나지 않은 S01·S02·M01 원형은 전달 대상이 아니다.

현재 즉시 전달 가능한 B01·R01 경로는 파일을 복사하지 않는 점검으로 확인한다.

```bash
.venv/bin/python scripts/experiments/package_notebook_bundle.py \
  --suite ready4 --dry-run
```

실제 전달 묶음은 위 점검이 통과한 뒤에만 `--dry-run`을 빼서 만든다.

## 보정·평가 데이터 묶음

`ready4` 전달 폴더에는 B03 INT8 보정용 train 표본과 최종 평가용 test를
다음 명령으로 추가한다.

```bash
nice -n 15 ionice -c3 .venv/bin/python \
  scripts/data_preparation/package_notebook_data.py
```

보정 표본은 train 3,625장을 파일명 순으로 정렬한 뒤 Python
`random.Random(42)`로 섞어 앞의 128장을 고정 선택한다. valid와 test는 후보에
넣지 않는다. test는 437장과 COCO 주석 593개를 전부 포함하며 보정에 사용하지
않는다. 현재 묶음의 데이터 payload는 568개 파일, 46,580,018 bytes이고 전체
train과 test 사이 파일명 및 SHA-256 중복은 모두 0개다. 선정 목록과 세부 통계는
`delivery/notebook-front-ready4/data-manifest.json`에 기록되어 있다.

노트북에 전달한 뒤 저장소 루트에서 무결성을 먼저 확인한다.

```bash
cd delivery/notebook-front-ready4
sha256sum -c data-checksums.sha256
cd ../..
```

로컬 전달 폴더는 원본을 수정하거나 복제하지 않는 hard link로 만들었다. 폴더를
압축하거나 다른 파일시스템·노트북으로 복사하면 독립된 일반 파일로 전달된다.
기본 경로를 유지했으므로 B03 engine 생성은 추가 경로 인자 없이 이 128장만
calibration에 사용하고, TensorRT 평가는 별도의 `test` 437장을 사용한다.

## 노트북 환경 확인

```bash
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU 모델, compute capability, driver, CUDA와 TensorRT 버전을 engine metadata에
기록한다. FP8과 NVIDIA 2:4는 지원되는 GPU와 tactic이 실제 선택됐는지 build
log에서 확인한다.

## 생성 원칙

1. FP32·FP16·일반 INT8은 검증된 baseline FP32 ONNX를 공유한다.
2. Structured pruning, 입력 해상도 변경과 Q/DQ 양자화는 별도 ONNX를 사용한다.
3. Engine 파일은 `artifacts/experiments/<ID>/front/model.engine`에 둔다.
4. ONNX checksum과 engine 생성 환경을 함께 기록한다.
5. 모든 후보를 동일한 test 437장과 평가 설정으로 측정한다.

## 재현 가능한 생성·평가 순서

첫 단계는 복구 학습과 무관한 네 engine이다.

```bash
# 명령과 입력 준비 상태만 확인
.venv/bin/python scripts/experiments/build_engine_suite.py \
  --suite ready4 --dry-run

# 노트북에서만 실제 생성 및 한 장 smoke test
.venv/bin/python scripts/experiments/build_engine_suite.py \
  --suite ready4 --force
```

생성 순서는 B01 FP32, B02 FP16, B03 INT8 PTQ, R01 432×432 FP16이다. B03은
`data/training/front_session_split_v1/train`을 정확한 calibration 경로로 사용하고
test 이미지는 calibration에 사용하지 않는다.

S01과 S02의 recovery 및 valid 비교가 끝나면 선택한 ID를 C01의
`artifact_source`와 `selected_experiment`에 기록한다. M01도 고정 2:4 mask를
유지한 recovery를 완료해야 한다. 각 최종 PTH에서 다시 생성하고 동등성·COCO
평가를 통과한 ONNX만 다음 이름으로 shared manifest에 등록한다.

```text
shared-models/S01-front-structured-recovery.onnx  # S02 선택 시 ID/파일명도 S02
shared-models/M01-M02-front-2to4-recovery.onnx
```

그 뒤 `final8`은 C01에 기록된 structured 선택을 자동으로 읽어 다음 대조 순서로
engine을 생성한다.

```text
B01 FP32 → B02 FP16 → B03 INT8 → R01 FP16
→ selected structured FP32 → C01 structured FP16
→ M01 2:4-weight dense FP16 → M02 2:4 sparse FP16
```

```bash
.venv/bin/python scripts/experiments/package_notebook_bundle.py \
  --suite final8 --dry-run
.venv/bin/python scripts/experiments/build_engine_suite.py \
  --suite final8 --dry-run
```

둘 중 하나라도 recovery 미완료, ONNX 누락, shared checksum 불일치를 보고하면
실제 빌드를 시작하지 않는다. 통과 후 노트북에서 `--dry-run`을 제거한다.

각 engine의 고정 이미지 latency를 충분한 warm-up과 반복 횟수로 측정하고, 동일한
437장 test에 대해 bbox AP, mask AP와 mask mIoU를 계산한다.

```bash
.venv/bin/python scripts/evaluation/benchmark_baseline.py \
  --camera front --backend fp16 \
  --engine artifacts/experiments/R01/front/model.engine \
  --image data/training/front_session_split_v1/test/frame_000005_20251222_185050_737914_png.rf.fe04634c99d43c9e986eaf41db38fd7a.jpg \
  --warmup 20 --runs 200

.venv/bin/python scripts/evaluation/evaluate_coco_tensorrt.py \
  --experiment R01 --camera front \
  --dataset-dir data/training/front_session_split_v1/test \
  --no-update-csv
```

위 두 명령은 engine ID마다 같은 설정으로 반복한다. R01은 입력 연산량 자체가
다르므로 504×504 후보와 별도 표에도 병기한다. M01과 M02는 가중치와 FP16 설정이
같고 sparse tactic 허용 여부만 달라야 하며, M02 build log와
`sparse-tactics.json`에서 실제 sparse tactic 선택을 확인한다.

등록된 후보의 engine 생성 예시는 다음과 같다.

```bash
.venv/bin/python scripts/experiments/build_candidate.py \
  B02 --camera front --target engine --force
```

구체적인 후보 순서는 [신규 Front 경량화 2차 계획](front-lightweighting-round-2.md)을
따른다.
