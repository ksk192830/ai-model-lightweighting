# RF-DETR Desktop Preliminary Diagnostics (Legacy)

이 문서는 `scripts/reporting/generate_round2_candidate_summary.py`가 실제 registry, static-analysis/comparison JSON 및 평가 JSON에서 생성한다. 수치는 추정하지 않는다.
이 문서의 기존 decision은 데스크탑 사전 진단이며 1차 정적평가 통과 여부가 아니다. 1차의 단일 기준은 `results/stage1-static-evaluation.md`이다.

- ONNX initializer elements는 상수 tensor의 element 수이며 학습 가능 parameter 수와 동의어가 아니다.
- MACs/FLOPs는 Conv/MatMul/Gemm dense 산술만 포함한 부분 하한 추정치이며 실측 latency를 대체하지 않는다.
- 모든 `Δ`는 B01 대비 `candidate - B01`, 상대 변화율은 `100 × Δ / B01`이다.
- 복구 학습이 끝나지 않은 후보의 정확도는 복구 전 진단값 대신 `N/A (pending)`으로 표시한다.

## 기존 데스크탑 사전 판정(현재 1차 gate에 사용하지 않음)

| ID | Method detail | Decision | Reason |
|---|---|---|---|
| B01 | N/A | control | baseline control |
| U01 | sparsity=0.1 | rejected | 정확도는 보존됐으나 10% zero sparsity가 dense ONNX 크기, graph node, 추정 MACs를 줄이지 못해 사전 고정 효율 gate를 통과하지 못함. |
| U02 | sparsity=0.3 | rejected | 30% sparsity에서 dense ONNX 크기·MACs·FLOPs가 감소하지 않았고 정확도가 대폭 하락함. |
| U03 | sparsity=0.5 | rejected | 50% zero sparsity가 dense ONNX 비용을 줄이지 못했고 bbox/mask AP가 사실상 0으로 붕괴해 효율 및 정확도 gate를 모두 통과하지 못함. |
| R02 | input=480x480 | rejected | B01 대비 dense MACs 10.96% 감소와 PTH 정확도 gate는 통과했으나, 전체 437장 PTH-ONNX semantic mIoU 절대 차이가 0.008998로 사전 고정 parity 한계 0.005를 초과함. |
| R01 | input=432x432 | retained | 432 입력에서 MACs/FLOPs 30.20% 감소 대비 정확도 하락이 제한적임. |
| R03 | input=384x384 | rejected | B01 대비 dense MACs는 46.25% 감소했으나 mask AP가 0.014341 하락해 허용 한계 0.01을 초과함. |
| S01 | remove_layers=1 | retained | Decoder 1개 제거 후 6 epoch 복구 학습으로 bbox AP를 baseline 대비 0.000411 이내로 유지하고 mask AP와 semantic mIoU를 개선했으며, 최종 ONNX production 동등성 검증을 통과함. |
| S02 | remove_layers=2 | rejected | Decoder 2개 제거 후 8 epoch 복구했으나 S01 대비 bbox AP 0.017924, mask AP 0.015162가 하락해 정확도 보존 기준을 통과하지 못함. TensorRT 속도는 구조 축소 효과를 기록하기 위한 대조군으로만 측정함. |
| S03 | reduction=0.2 | rejected | FFN 20.31% 축소 대비 전체 parameter 3.00%, FLOPs 0.39%만 감소하고 ONNX 동등성 엄격 기준을 통과하지 못함. |
| S04 | reduction=0.4 | rejected | ONNX 크기는 6.53% 감소했으나 dense MACs 감소는 0.79%에 그쳤고 bbox AP 0.216339, mask AP 0.076490, semantic mIoU 0.054941가 하락해 정확도 gate를 통과하지 못함. |
| C03 | input=432x432 | retained | B01 대비 graph node 8.04%, dense MACs 30.76%를 줄이면서 세 정확도 gate와 production-image ONNX 동등성 및 전체 데이터셋 PTH-ONNX parity를 모두 통과함. |
| C04 | input=480x480 | retained | B01 대비 graph node 8.04%, dense MACs 11.51%를 줄이면서 세 정확도 gate와 production-image ONNX 동등성 및 전체 데이터셋 PTH-ONNX parity를 모두 통과함. |
| M01 | pattern=2:4 | rejected | 10 epoch 복구 학습 후에도 bbox AP 0.105390, mask AP 0.085736, semantic mIoU 0.029382가 하락했고 production-image ONNX 동등성도 통과하지 못함. |

## 구조 및 연산량

| ID | Method | Status | Res. | Initializer elements | Δ elements | Δ elements (%) | Nodes | Δ nodes | Δ nodes (%) | MACs* | Δ MACs | Δ MACs (%) | FLOPs* | Δ FLOPs | Δ FLOPs (%) | ONNX bytes | Δ bytes | Δ size (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B01 | none | onnx-exported | 504 | 32,537,747 | +0 | +0.000000% | 2,127 | +0 | +0.000000% | 54,083,668,224 | +0 | +0.000000% | 108,167,336,448 | +0 | +0.000000% | 130,677,849 | +0 | +0.000000% |
| U01 | global-magnitude | static-analysis-rejected | 504 | 32,537,747 | +0 | +0.000000% | 2,127 | +0 | +0.000000% | 54,083,668,224 | +0 | +0.000000% | 108,167,336,448 | +0 | +0.000000% | 130,677,820 | -29 | -0.000022% |
| U02 | global-magnitude | static-analysis-rejected | 504 | 32,537,747 | +0 | +0.000000% | 2,127 | +0 | +0.000000% | 54,083,668,224 | +0 | +0.000000% | 108,167,336,448 | +0 | +0.000000% | 130,677,820 | -29 | -0.000022% |
| U03 | global-magnitude | static-analysis-rejected | 504 | 32,537,747 | +0 | +0.000000% | 2,127 | +0 | +0.000000% | 54,083,668,224 | +0 | +0.000000% | 108,167,336,448 | +0 | +0.000000% | 130,677,820 | -29 | -0.000022% |
| R02 | input-resolution | onnx-exported | 480 | 32,474,771 | -62,976 | -0.193548% | 2,127 | +0 | +0.000000% | 48,158,634,240 | -5,925,033,984 | -10.955311% | 96,317,268,480 | -11,850,067,968 | -10.955311% | 130,424,617 | -253,232 | -0.193783% |
| R01 | input-resolution | onnx-exported | 432 | 32,358,035 | -179,712 | -0.552319% | 2,127 | +0 | +0.000000% | 37,749,285,120 | -16,334,383,104 | -30.202062% | 75,498,570,240 | -32,668,766,208 | -30.202062% | 129,955,207 | -722,642 | -0.552995% |
| R03 | input-resolution | onnx-exported | 384 | 32,253,587 | -284,160 | -0.873324% | 2,127 | +0 | +0.000000% | 29,067,358,464 | -25,016,309,760 | -46.254832% | 58,134,716,928 | -50,032,619,520 | -46.254832% | 129,535,207 | -1,142,642 | -0.874396% |
| S01 | decoder-layer | onnx-exported | 504 | 30,997,043 | -1,540,704 | -4.735128% | 1,956 | -171 | -8.039492% | 53,782,953,984 | -300,714,240 | -0.556017% | 107,565,907,968 | -601,428,480 | -0.556017% | 124,477,541 | -6,200,308 | -4.744728% |
| S02 | decoder-layer | onnx-exported | 504 | 29,456,339 | -3,081,408 | -9.470256% | 1,783 | -344 | -16.173014% | 53,482,239,744 | -601,428,480 | -1.112033% | 106,964,479,488 | -1,202,856,960 | -1.112033% | 118,276,976 | -12,400,873 | -9.489652% |
| S03 | ffn-dimension | static-analysis-rejected | 504 | 31,470,707 | -1,067,040 | -3.279391% | 2,127 | +0 | +0.000000% | 53,870,676,224 | -212,992,000 | -0.393819% | 107,741,352,448 | -425,984,000 | -0.393819% | 126,409,640 | -4,268,209 | -3.266207% |
| S04 | ffn-dimension | recovery-pending | 504 | 30,403,667 | -2,134,080 | -6.558782% | 2,127 | +0 | +0.000000% | 53,657,684,224 | -425,984,000 | -0.787639% | 107,315,368,448 | -851,968,000 | -0.787639% | 122,141,480 | -8,536,369 | -6.532376% |
| C03 | structured-resolution | onnx-exported | 432 | 30,817,331 | -1,720,416 | -5.287447% | 1,956 | -171 | -8.039492% | 37,448,570,880 | -16,635,097,344 | -30.758079% | 74,897,141,760 | -33,270,194,688 | -30.758079% | 123,754,899 | -6,922,950 | -5.297723% |
| C04 | structured-resolution | onnx-exported | 480 | 30,934,067 | -1,603,680 | -4.928676% | 1,956 | -171 | -8.039492% | 47,857,920,000 | -6,225,748,224 | -11.511328% | 95,715,840,000 | -12,451,496,448 | -11.511328% | 124,224,309 | -6,453,540 | -4.938511% |
| M01 | nvidia-2to4 | onnx-exported | 504 | 32,537,747 | +0 | +0.000000% | 2,127 | +0 | +0.000000% | 54,083,668,224 | +0 | +0.000000% | 108,167,336,448 | +0 | +0.000000% | 130,677,820 | -29 | -0.000022% |

## 정확도

| ID | Accuracy status | Source | BBox AP | Δ abs. | Δ rel. (%) | Mask AP | Δ abs. | Δ rel. (%) | mIoU | Δ abs. | Δ rel. (%) | Result JSON |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B01 | valid | PTH | 0.737791 | +0.000000 | +0.000000% | 0.601016 | +0.000000 | +0.000000% | 0.712200 | +0.000000 | +0.000000% | results/coco-evaluation/front-rfdetr-seg-large-v1-test.json |
| U01 | valid | PTH | 0.737531 | -0.000260 | -0.035203% | 0.609726 | +0.008710 | +1.449140% | 0.719163 | +0.006963 | +0.977676% | results/coco-evaluation/U01-front-pth.json |
| U02 | valid | PTH | 0.096442 | -0.641349 | -86.928274% | 0.082022 | -0.518994 | -86.352811% | 0.032849 | -0.679351 | -95.387679% | results/coco-evaluation/U02-front-pth.json |
| U03 | valid | PTH | 0.000014 | -0.737777 | -99.998041% | 0.000000 | -0.601016 | -100.000000% | 0.000000 | -0.712200 | -100.000000% | results/coco-evaluation/U03-front-pth.json |
| R02 | valid | PTH | 0.731265 | -0.006526 | -0.884540% | 0.600393 | -0.000623 | -0.103692% | 0.703650 | -0.008550 | -1.200453% | results/coco-evaluation/R02-front-480-pth.json |
| R01 | valid | PTH | 0.730661 | -0.007130 | -0.966449% | 0.592974 | -0.008042 | -1.338127% | 0.696878 | -0.015321 | -2.151288% | results/coco-evaluation/R01-front-432-pth.json |
| R03 | valid | PTH | 0.728659 | -0.009132 | -1.237711% | 0.586675 | -0.014341 | -2.386083% | 0.729042 | +0.016842 | +2.364788% | results/coco-evaluation/R03-front-384-pth.json |
| S01 | valid | PTH | 0.737380 | -0.000411 | -0.055749% | 0.602751 | +0.001735 | +0.288623% | 0.733921 | +0.021721 | +3.049911% | results/coco-evaluation/S01-front-after-recovery.json |
| S02 | valid | PTH | 0.719456 | -0.018335 | -2.485114% | 0.587589 | -0.013427 | -2.234098% | 0.762691 | +0.050492 | +7.089541% | results/coco-evaluation/S02-front-after-recovery.json |
| S03 | rejected | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| S04 | pending | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| C03 | valid | PTH | 0.730738 | -0.007054 | -0.956050% | 0.593797 | -0.007219 | -1.201094% | 0.739237 | +0.027037 | +3.796282% | results/coco-evaluation/C03-front-432-pth.json |
| C04 | valid | PTH | 0.733055 | -0.004736 | -0.641920% | 0.599448 | -0.001568 | -0.260932% | 0.728948 | +0.016748 | +2.351653% | results/coco-evaluation/C04-front-480-pth.json |
| M01 | valid | PTH | 0.632401 | -0.105390 | -14.284526% | 0.515280 | -0.085736 | -14.265111% | 0.682817 | -0.029382 | -4.125577% | results/coco-evaluation/M01-front-after-recovery.json |

## 평가 산출물 판정

| Experiment | Evaluation JSON | Backend | Disposition | Reason |
|---|---|---|---|---|
| B01 | results/coco-evaluation/B01-front-baseline-onnx.json | ONNX | valid-not-selected | protocol matches baseline |
| B01 | results/coco-evaluation/front-rfdetr-seg-large-v1-test.json | PTH | selected | protocol matches baseline |
| C03 | results/coco-evaluation/C03-front-432-onnx.json | ONNX | valid-not-selected | protocol matches baseline |
| C03 | results/coco-evaluation/C03-front-432-pth.json | PTH | selected | registry-selected; protocol matches baseline |
| C04 | results/coco-evaluation/C04-front-480-onnx.json | ONNX | valid-not-selected | protocol matches baseline |
| C04 | results/coco-evaluation/C04-front-480-pth.json | PTH | selected | registry-selected; protocol matches baseline |
| M01 | results/coco-evaluation/M01-front-after-recovery.json | PTH | selected | registry-selected; protocol matches baseline |
| R01 | results/coco-evaluation/R01-front-432-onnx.json | ONNX | valid-not-selected | protocol matches baseline |
| R01 | results/coco-evaluation/R01-front-432-pth.json | PTH | selected | registry-selected; protocol matches baseline |
| R02 | results/coco-evaluation/R02-front-480-onnx.json | ONNX | rejected | PyTorch parity failed |
| R02 | results/coco-evaluation/R02-front-480-pth.json | PTH | selected | registry-selected; protocol matches baseline |
| R03 | results/coco-evaluation/R03-front-384-pth.json | PTH | selected | registry-selected; protocol matches baseline |
| S01 | results/coco-evaluation/S01-front-after-recovery-onnx-invalid-numselect300.json | ONNX | rejected | filename marked invalid; ONNX result lacks explicit postprocess_num_select; PyTorch parity failed |
| S01 | results/coco-evaluation/S01-front-after-recovery-onnx.json | ONNX | valid-not-selected | protocol matches baseline |
| S01 | results/coco-evaluation/S01-front-after-recovery.json | PTH | selected | registry-selected; protocol matches baseline |
| S01 | results/coco-evaluation/S01-front-before-recovery.json | PTH | superseded | pre-recovery result superseded by completed recovery |
| S02 | results/coco-evaluation/S02-front-after-recovery-onnx.json | ONNX | valid-not-selected | protocol matches baseline |
| S02 | results/coco-evaluation/S02-front-after-recovery.json | PTH | selected | registry-selected; protocol matches baseline |
| S02 | results/coco-evaluation/S02-front-before-recovery.json | PTH | superseded | pre-recovery result superseded by completed recovery |
| S04 | results/coco-evaluation/S04-front-pth.json | PTH | selected | registry-selected; protocol matches baseline |
| U01 | results/coco-evaluation/U01-front-pth.json | PTH | selected | registry-selected; protocol matches baseline |
| U02 | results/coco-evaluation/U02-front-pth.json | PTH | selected | registry-selected; protocol matches baseline |
| U03 | results/coco-evaluation/U03-front-pth.json | PTH | selected | registry-selected; protocol matches baseline |
