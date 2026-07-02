# 최종 TensorRT engine 선정

## Front 평가 조건

- GPU: NVIDIA GeForce RTX 3080
- TensorRT: 10.16.1.11
- Dataset: `data/labeled_test/front`, 296 images
- COCO categories: out_line, parking_lot, parking_space
- Confidence threshold: 0.001
- Latency: preprocessing, TensorRT execution, postprocessing 포함

| ID | 방식 | 크기 (bytes) | 평균 latency (ms) | bbox AP | mask AP | 선정 |
|---|---|---:|---:|---:|---:|---|
| B01 | FP32 baseline | 135,608,716 | 124.505 | 0.8325 | 0.6305 | 기준 |
| S01 | decoder 5→4 FP32 | 129,294,868 | 123.344 | 0.8367 | 0.6458 | structured 대표 |
| S02 | decoder 5→3 FP32 | 122,940,060 | 125.376 | 0.8343 | 0.6383 | 제외 |
| S03 | FFN 2048→1632 FP32 | 131,336,108 | 120.095 | 0.8262 | 0.6416 | 제외 |
| S04 | FFN 2048→1216 FP32 | 127,053,732 | 119.706 | 0.8177 | 0.6365 | 제외 |
| C01 | S01 + FP16 | 67,923,620 | 115.454 | 0.8372 | 0.6461 | 결합 대표 |
| C02 | S01 + INT8/PTQ | 67,931,036 | 115.528 | 0.8374 | 0.6467 | FP16 fallback으로 제외 |
| R01 | 432×432 + FP16 | 70,859,748 | 115.373 | 0.8226 | 0.6229 | 입력 최적화 대표 |

C02는 296장 calibration을 완료했지만 TensorRT build가 FP16 fallback을
허용하며, C01과 엔진 크기 및 지연시간 차이가 실질적으로 없다. 최종 전달
세트에서는 중복 후보로 제외하고 실험 증거로 보존한다.

## 최종 전달 후보

Front:

- B01, B02, B03
- M01, M02
- S01
- C01
- R01

최종 범위는 front 모델로 한정한다.

## 13개 생성 후 8개 선정

초기 계획은 front TensorRT 후보를 12~15개 생성한 뒤 실제 평가 결과로
5~8개를 선정하는 것이었다. 실제로 다음 13개 engine을 생성했다.

- Baseline: B01, B02, B03
- Unstructured 대조군: U02
- NVIDIA 2:4: M01, M02
- Structured: S01, S02, S03, S04
- 결합: C01, C02
- 입력 해상도: R01

최종 8개는 `B01, B02, B03, M01, M02, S01, C01, R01`이다.

제외 근거:

- U02: 비정형 zero가 dense TensorRT graph 감소로 이어지지 않는다는 대조군
- S02: S01보다 크기는 작지만 측정 latency가 느리고 정확도 우위가 없음
- S03/S04: 속도는 개선됐지만 bbox AP가 S01보다 낮음
- C02: INT8 calibration은 완료했으나 FP16 fallback이 활성화되어 C01과
  크기·속도가 실질적으로 중복

따라서 13개는 실험 증거로 모두 보존하고, 시각 비교와 최종 전달은 8개로
제한한다.
