# INT8 양자화 정리

## INT8이란?

INT8 양자화는 모델의 가중치와 일부 연산 값을 32비트 또는 16비트
부동소수점 대신 8비트 정수로 표현하는 경량화 방식이다.

```text
FP32: 숫자 하나당 4 bytes
FP16: 숫자 하나당 2 bytes
INT8: 숫자 하나당 1 byte
```

가중치만 단순 비교하면 INT8은 FP32의 약 25%, FP16의 약 50% 크기로
표현할 수 있다. NVIDIA GPU에서 INT8 연산을 지원하면 모델 크기, GPU
메모리 사용량 및 추론 시간을 추가로 줄일 수 있다.

모델 구조와 파라미터 개수는 그대로 유지되며, 숫자를 표현하는 방식이
바뀐다.

## 부동소수점을 정수로 바꾸는 방법

FP32 값은 연속적인 실수 범위를 표현하지만 INT8은 제한된 정수 범위만
표현한다.

```text
signed INT8 범위: -128 ~ 127
```

따라서 실수 값을 INT8 범위에 대응시키는 scale과 zero point가 필요하다.

개념적으로 다음과 같이 변환한다.

```text
quantized_value = round(real_value / scale) + zero_point
```

추론 결과를 해석할 때는 반대 변환을 사용한다.

```text
real_value ≈ (quantized_value - zero_point) × scale
```

이 과정에서 반올림과 범위 제한이 발생하므로 원본 FP32 값과 작은 차이가
생길 수 있다.

## Calibration이란?

INT8은 모델의 중간 activation이 실제 데이터에서 어느 범위로 나오는지
알아야 적절한 scale을 정할 수 있다. 대표 이미지를 모델에 입력해 이
범위를 수집하는 과정이 calibration이다.

```text
대표 이미지
→ FP32/FP16 모델 실행
→ 레이어별 activation 범위 수집
→ INT8 scale 결정
→ TensorRT INT8 engine 생성
```

Calibration 이미지는 실제 운영 데이터를 대표해야 한다.

이번 프로젝트에서는 전면과 후면 카메라의 시점과 클래스가 다르므로
각 모델에 맞는 이미지를 별도로 사용한다.

```text
전면 모델 calibration → data/calibration/front
후면 모델 calibration → data/calibration/rear
```

권장 원칙:

- 학습 또는 calibration 전용 이미지 사용
- 너무 적은 이미지보다 다양한 환경을 포함
- 밝기, 거리, 각도 및 객체 배치가 편중되지 않도록 선정
- 최종 정확도 평가용 test 데이터와 분리
- 사용한 이미지 목록과 데이터 버전 기록

## PTQ와 QAT

### PTQ: Post-Training Quantization

학습이 끝난 모델을 calibration한 후 INT8로 변환한다.

```text
학습 완료 checkpoint
→ calibration
→ INT8 engine
```

장점:

- 추가 학습이 필요하지 않음
- 구현과 실험이 비교적 빠름
- 기존 checkpoint를 그대로 활용 가능

단점:

- 모델에 따라 정확도 저하가 클 수 있음
- calibration 데이터 품질에 민감함

이번 프로젝트에서는 먼저 TensorRT PTQ INT8을 적용한다.

### QAT: Quantization-Aware Training

학습 중에 양자화 오차를 모사하여 INT8 환경에 모델이 적응하도록
재학습하는 방식이다.

장점:

- PTQ보다 정확도를 잘 유지할 가능성이 있음

단점:

- 학습 코드와 데이터가 필요함
- 실험 시간과 구현 난이도가 증가함

PTQ 정확도가 충분하지 않을 때 후속 실험으로 검토한다.

## TensorRT INT8

NVIDIA GPU에서 사용할 기본 변환 흐름은 다음과 같다.

```text
PyTorch checkpoint
→ ONNX
→ calibration
→ TensorRT INT8 engine
```

예상 산출물:

```text
artifacts/
├── onnx/
│   ├── front/parking_front.onnx
│   └── rear/parking_rear.onnx
├── calibration/
│   ├── front/
│   └── rear/
└── tensorrt/
    ├── front/parking_front_int8.engine
    └── rear/parking_rear_int8.engine
```

TensorRT engine은 생성한 GPU 계열, TensorRT 버전 및 빌드 옵션에 영향을
받을 수 있다. 따라서 다음 정보를 함께 기록한다.

- GPU 모델과 메모리
- NVIDIA 드라이버 버전
- CUDA 및 TensorRT 버전
- ONNX 파일과 opset
- 입력 shape와 batch size
- calibration 이미지 목록
- calibration cache
- engine 생성 명령

## 정확도 저하 가능성

INT8은 FP16보다 표현 가능한 값이 적기 때문에 정확도 저하 위험이 더
크다.

현재 모델은 instance segmentation 모델이므로 다음 항목을 비교한다.

- 검출 클래스 변화
- confidence 변화
- bounding box 위치 변화
- mask 경계와 면적 변화
- detection mAP
- mask AP 또는 IoU

특히 작은 객체, 얇은 경계선 및 낮은 confidence의 객체는 양자화 오차에
더 민감할 수 있다.

## 이번 프로젝트의 INT8 실험군

| ID | Framework | Precision | 설명 |
|---|---|---|---|
| B0 | PyTorch | FP32 | 원본 기준 |
| F1 | TensorRT | FP16 | FP16 기준 |
| Q1 | TensorRT | INT8 PTQ | calibration 기반 INT8 |

비교 항목:

- engine 파일 크기
- 평균·중앙값·p95 추론 시간
- FPS
- GPU 메모리 사용량
- detection 및 segmentation 정확도
- FP32 대비 정확도 감소율

## 구현 순서

1. 전면·후면 calibration 이미지 선정
2. calibration 목록과 데이터 버전 저장
3. 기존 ONNX 구조와 입력 shape 확인
4. NVIDIA 장비에서 TensorRT INT8 engine 생성
5. INT8 추론 결과가 정상인지 샘플 이미지로 확인
6. FP32 및 TensorRT FP16과 동일 조건으로 비교
7. 정확도 저하가 크면 calibration 구성을 개선
8. 필요할 경우 QAT 검토

현재 calibration 목록은 각 카메라에서 숫자 ID가 5의 배수인 이미지를
선택하며, test 목록은 나머지가 2인 이미지를 선택한다.

```text
image000005.png
image000010.png
image000015.png
...
```

```text
test: image000002.png, image000007.png, image000012.png, ...
```

다음 명령으로 전면·후면 calibration/test 목록을 동시에 생성한다.

```bash
python3 scripts/data_preparation/create_calibration_splits.py
```

목록과 재현용 메타데이터는 `splits/`에 저장하며 원본 이미지는 복사하지
않는다.

핵심은 다음과 같다.

> INT8은 숫자를 8비트 정수로 줄여 FP16보다 더 강하게 경량화하지만,
> 실제 데이터로 activation 범위를 정하는 calibration이 필요하며 정확도
> 저하를 반드시 검증해야 한다.


## TensorRT INT8 엔진 생성

NVIDIA GPU, CUDA 지원 PyTorch 및 TensorRT Python 패키지가 설치된
컴퓨터에서 실행한다. calibration 이미지는 `data/calibration/front`와
`data/calibration/rear`를 사용한다.

카메라 하나만 변환하려면 다음 명령을 실행한다.

```bash
.venv/bin/python scripts/lightweighting/build_tensorrt_int8.py --camera front
.venv/bin/python scripts/lightweighting/build_tensorrt_int8.py --camera rear
```

ONNX, TensorRT FP16, TensorRT INT8을 전면·후면 모델에 한 번에 적용하려면
다음 명령을 실행한다.

```bash
.venv/bin/python scripts/lightweighting/run_lightweighting.py \
  --camera all \
  --methods onnx tensorrt-fp16 tensorrt-int8
```

NVIDIA 장비로 옮기기 전에 입력 파일과 실행 명령을 확인하려면
`--dry-run`을 사용한다.

```bash
.venv/bin/python scripts/lightweighting/run_lightweighting.py \
  --camera all \
  --methods tensorrt-int8 \
  --dry-run
```

산출물은 `artifacts/tensorrt/<camera>/` 아래의 INT8 `.engine`, calibration
`.cache`, 재현 정보 `.json`이다. JSON에는 사용한 ONNX 체크섬, calibration
이미지 목록, 전처리 방식, GPU 및 TensorRT 버전이 기록된다.

기본 설정은 INT8을 지원하지 않는 레이어에 FP16을 허용한다. 완전한 INT8
제약을 시험하려면 `--no-fp16-fallback`을 사용한다.
