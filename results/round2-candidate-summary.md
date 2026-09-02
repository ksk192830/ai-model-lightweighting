# RF-DETR Round-2 Candidate Summary

이 문서는 `scripts/reporting/generate_round2_candidate_summary.py`가 실제 registry, static-analysis/comparison JSON 및 평가 JSON에서 생성한다. 수치는 추정하지 않는다.

- ONNX initializer elements는 상수 tensor의 element 수이며 학습 가능 parameter 수와 동의어가 아니다.
- MACs/FLOPs는 Conv/MatMul/Gemm dense 산술만 포함한 부분 하한 추정치이며 실측 latency를 대체하지 않는다.
- 모든 `Δ`는 B01 대비 `candidate - B01`, 상대 변화율은 `100 × Δ / B01`이다.
- 복구 학습이 끝나지 않은 후보의 정확도는 복구 전 진단값 대신 `N/A (pending)`으로 표시한다.

## 구조 및 연산량

| ID | Method | Status | Res. | Initializer elements | Δ elements | Δ elements (%) | Nodes | Δ nodes | Δ nodes (%) | MACs* | Δ MACs | Δ MACs (%) | FLOPs* | Δ FLOPs | Δ FLOPs (%) | ONNX bytes | Δ bytes | Δ size (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B01 | none | onnx-exported | 504 | 32,537,747 | +0 | +0.000000% | 2,127 | +0 | +0.000000% | 54,083,668,224 | +0 | +0.000000% | 108,167,336,448 | +0 | +0.000000% | 130,677,849 | +0 | +0.000000% |
| U02 | global-magnitude | static-analysis-rejected | 504 | 32,537,747 | +0 | +0.000000% | 2,127 | +0 | +0.000000% | 54,083,668,224 | +0 | +0.000000% | 108,167,336,448 | +0 | +0.000000% | 130,677,820 | -29 | -0.000022% |
| R01 | input-resolution | onnx-exported | 432 | 32,358,035 | -179,712 | -0.552319% | 2,127 | +0 | +0.000000% | 37,749,285,120 | -16,334,383,104 | -30.202062% | 75,498,570,240 | -32,668,766,208 | -30.202062% | 129,955,207 | -722,642 | -0.552995% |
| S01 | decoder-layer | onnx-exported | 504 | 30,997,043 | -1,540,704 | -4.735128% | 1,956 | -171 | -8.039492% | 53,782,953,984 | -300,714,240 | -0.556017% | 107,565,907,968 | -601,428,480 | -0.556017% | 124,477,541 | -6,200,308 | -4.744728% |
| S02 | decoder-layer | onnx-exported | 504 | 29,456,339 | -3,081,408 | -9.470256% | 1,783 | -344 | -16.173014% | 53,482,239,744 | -601,428,480 | -1.112033% | 106,964,479,488 | -1,202,856,960 | -1.112033% | 118,276,976 | -12,400,873 | -9.489652% |
| S03 | ffn-dimension | static-analysis-rejected | 504 | 31,470,707 | -1,067,040 | -3.279391% | 2,127 | +0 | +0.000000% | 53,870,676,224 | -212,992,000 | -0.393819% | 107,741,352,448 | -425,984,000 | -0.393819% | 126,409,640 | -4,268,209 | -3.266207% |
| M01 | nvidia-2to4 | recovery-running | 504 | 32,537,747 | +0 | +0.000000% | 2,127 | +0 | +0.000000% | 54,083,668,224 | +0 | +0.000000% | 108,167,336,448 | +0 | +0.000000% | 130,677,820 | -29 | -0.000022% |

## 정확도

| ID | Accuracy status | Source | BBox AP | Δ abs. | Δ rel. (%) | Mask AP | Δ abs. | Δ rel. (%) | mIoU | Δ abs. | Δ rel. (%) | Result JSON |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B01 | valid | PTH | 0.737791 | +0.000000 | +0.000000% | 0.601016 | +0.000000 | +0.000000% | 0.712200 | +0.000000 | +0.000000% | results/coco-evaluation/front-rfdetr-seg-large-v1-test.json |
| U02 | valid | PTH | 0.096442 | -0.641349 | -86.928274% | 0.082022 | -0.518994 | -86.352811% | 0.032849 | -0.679351 | -95.387679% | results/coco-evaluation/U02-front-pth.json |
| R01 | valid | PTH | 0.730661 | -0.007130 | -0.966449% | 0.592974 | -0.008042 | -1.338127% | 0.696878 | -0.015321 | -2.151288% | results/coco-evaluation/R01-front-432-pth.json |
| S01 | valid | PTH | 0.737380 | -0.000411 | -0.055749% | 0.602751 | +0.001735 | +0.288623% | 0.733921 | +0.021721 | +3.049911% | results/coco-evaluation/S01-front-after-recovery.json |
| S02 | valid | PTH | 0.719456 | -0.018335 | -2.485114% | 0.587589 | -0.013427 | -2.234098% | 0.762691 | +0.050492 | +7.089541% | results/coco-evaluation/S02-front-after-recovery.json |
| S03 | rejected | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| M01 | pending | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## 평가 산출물 판정

| Experiment | Evaluation JSON | Backend | Disposition | Reason |
|---|---|---|---|---|
| B01 | results/coco-evaluation/B01-front-baseline-onnx.json | ONNX | valid-not-selected | protocol matches baseline |
| B01 | results/coco-evaluation/front-rfdetr-seg-large-v1-test.json | PTH | selected | protocol matches baseline |
| R01 | results/coco-evaluation/R01-front-432-onnx.json | ONNX | valid-not-selected | protocol matches baseline |
| R01 | results/coco-evaluation/R01-front-432-pth.json | PTH | selected | registry-selected; protocol matches baseline |
| S01 | results/coco-evaluation/S01-front-after-recovery-onnx-invalid-numselect300.json | ONNX | rejected | filename marked invalid; ONNX result lacks explicit postprocess_num_select; PyTorch parity failed |
| S01 | results/coco-evaluation/S01-front-after-recovery-onnx.json | ONNX | valid-not-selected | protocol matches baseline |
| S01 | results/coco-evaluation/S01-front-after-recovery.json | PTH | selected | registry-selected; protocol matches baseline |
| S01 | results/coco-evaluation/S01-front-before-recovery.json | PTH | superseded | pre-recovery result superseded by completed recovery |
| S02 | results/coco-evaluation/S02-front-after-recovery-onnx.json | ONNX | valid-not-selected | protocol matches baseline |
| S02 | results/coco-evaluation/S02-front-after-recovery.json | PTH | selected | registry-selected; protocol matches baseline |
| S02 | results/coco-evaluation/S02-front-before-recovery.json | PTH | superseded | pre-recovery result superseded by completed recovery |
| U02 | results/coco-evaluation/U02-front-pth.json | PTH | selected | registry-selected; protocol matches baseline |
