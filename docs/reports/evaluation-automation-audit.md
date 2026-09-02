# 평가 기준 및 자동화 감사 보고서

- 감사 상태: **IN_PROGRESS**
- 기준 원본: `configs/experiments/defaults.yaml`
- 검사 수: 43
- 필수 실패: 0
- 실행 대기: 5
- 권고 사항: 9

## 감사 판정

| 영역 | 검사 | 상태 | 근거 |
|---|---|---|---|
| dataset | dataset-leakage-integrity | PASS | augmentation-free=True; cross-split duplicates=0; session overlap=0; COCO reference errors=0 |
| dataset | evaluation-image-count | PASS | configured=437; actual COCO images=437 |
| dataset | split-ratio-realization | PASS | train=3625 (81.17%), valid=404 (9.05%), test=437 (9.79%) |
| accuracy | evaluation-B01 | PASS | results/coco-evaluation/front-rfdetr-seg-large-v1-test.json; protocol and metrics valid |
| reproducibility | evaluation-provenance-B01 | WARN | legacy result lacks embedded annotation SHA-256; current annotation independently hashes to 8bb43b8f94b182b51b5ff4484cc04964d606a3eff53304fa01cc00b7fbfb3eba |
| accuracy | evaluation-R01 | PASS | results/coco-evaluation/R01-front-432-pth.json; protocol and metrics valid |
| reproducibility | evaluation-provenance-R01 | WARN | legacy result lacks embedded annotation SHA-256; current annotation independently hashes to 8bb43b8f94b182b51b5ff4484cc04964d606a3eff53304fa01cc00b7fbfb3eba |
| accuracy | evaluation-S01 | PASS | results/coco-evaluation/S01-front-after-recovery.json; protocol and metrics valid |
| reproducibility | evaluation-provenance-S01 | WARN | legacy result lacks embedded annotation SHA-256; current annotation independently hashes to 8bb43b8f94b182b51b5ff4484cc04964d606a3eff53304fa01cc00b7fbfb3eba |
| accuracy | evaluation-S02 | PASS | results/coco-evaluation/S02-front-after-recovery.json; protocol and metrics valid |
| reproducibility | evaluation-provenance-S02 | WARN | legacy result lacks embedded annotation SHA-256; current annotation independently hashes to 8bb43b8f94b182b51b5ff4484cc04964d606a3eff53304fa01cc00b7fbfb3eba |
| accuracy | evaluation-U02 | PASS | results/coco-evaluation/U02-front-pth.json; protocol and metrics valid |
| reproducibility | evaluation-provenance-U02 | WARN | legacy result lacks embedded annotation SHA-256; current annotation independently hashes to 8bb43b8f94b182b51b5ff4484cc04964d606a3eff53304fa01cc00b7fbfb3eba |
| conversion | onnx-evaluation-B01-B01-front-baseline-onnx | PASS | results/coco-evaluation/B01-front-baseline-onnx.json; query count/parity valid |
| conversion | onnx-evaluation-R01-R01-front-432-onnx | PASS | results/coco-evaluation/R01-front-432-onnx.json; query count/parity valid |
| conversion | onnx-evaluation-S01-S01-front-after-recovery-onnx | PASS | results/coco-evaluation/S01-front-after-recovery-onnx.json; query count/parity valid |
| conversion | onnx-evaluation-S02-S02-front-after-recovery-onnx | PASS | results/coco-evaluation/S02-front-after-recovery-onnx.json; query count/parity valid |
| conversion | graph-equivalence-B01 | PASS | artifacts/experiments/B01/front/onnx-equivalence.json; passed=True; samples=10; seed=42; mode=real-images; protocol match=True |
| conversion | graph-equivalence-schema-B01 | WARN | legacy graph report predates strict active-membership/class/score fields; full-dataset ONNX accuracy parity is available |
| conversion | graph-equivalence-R01 | PASS | artifacts/experiments/R01/front/onnx-equivalence.json; passed=True; samples=10; seed=42; mode=real-images; protocol match=True |
| conversion | graph-equivalence-schema-R01 | WARN | legacy graph report predates strict active-membership/class/score fields; full-dataset ONNX accuracy parity is available |
| conversion | graph-equivalence-S01 | PASS | artifacts/experiments/S01/front/onnx-equivalence.json; passed=True; samples=10; seed=42; mode=real-images; protocol match=True |
| conversion | graph-equivalence-S02 | PASS | artifacts/experiments/S02/front/onnx-equivalence.json; passed=True; samples=10; seed=42; mode=real-images; protocol match=True |
| conversion | graph-equivalence-U02 | PASS | artifacts/experiments/U02/front/onnx-equivalence.json; candidate decision=rejected; equivalence passed=True; protocol match=True |
| conversion | graph-equivalence-S03 | PASS | artifacts/experiments/S03/front/onnx-equivalence.json; candidate decision=rejected; equivalence passed=False; protocol match=True |
| conversion | graph-equivalence-M01 | PENDING | artifacts/experiments/M01/front/onnx-equivalence.json; prototype report is superseded when recovery completes |
| static-analysis | static-analysis-B01 | PASS | artifacts/experiments/B01/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-U02 | PASS | artifacts/experiments/U02/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-M01 | PENDING | artifacts/experiments/M01/front/static-analysis.json; status=partial; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=awaiting refresh |
| static-analysis | static-analysis-M02 | PENDING | missing artifacts/experiments/M02/front/static-analysis.json |
| static-analysis | static-analysis-S01 | PASS | artifacts/experiments/S01/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9659090909090909; estimated/all-node fraction=0.0869120654396728 |
| static-analysis | static-analysis-S02 | PASS | artifacts/experiments/S02/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9691358024691358; estimated/all-node fraction=0.08805384183959619 |
| static-analysis | static-analysis-S03 | PASS | artifacts/experiments/S03/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-R01 | PASS | artifacts/experiments/R01/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| automation | desktop-pipeline-protocol-binding | PASS | pipeline status=waiting-for-recovery; running queue records the canonical protocol |
| automation | post-recovery-protocol-binding | PASS | M01 post-recovery status=waiting-for-training; canonical protocol match=True |
| latency | latency-results | PENDING | TensorRT engines/benchmarks are waiting for M01 recovery |
| automation | engine-summary | PENDING | desktop engine summary will be generated after recovery/build/evaluation |
| automation | pipeline-stage-wiring | PASS | engine inspection, paper table/figure, static report, and final audit are wired |
| automation | paper-report-source | PASS | paper outputs consume only the current desktop engine summary |
| automation | int8-calibration-binding | PASS | train-only deterministic calibration: count=128, seed=42 |
| study-design | fixed-benchmark-reuse | WARN | 437 images are a repeated comparative benchmark, not an untouched confirmatory test; collect a new session for external-generalization claims |
| study-design | training-seed-replication | WARN | current recovery results use seed 42 only; report this as a deterministic engineering comparison or add multiple training seeds for variance estimates |

## 사전 정의된 후보 수용 기준의 적용 결과

`REJECT`는 자동화 결함이 아니라 해당 후보가 정확도 보존 기준을 통과하지 못했다는 뜻이다.

| 후보 | 비교 | 판정 | Gate details |
|---|---|---|---|
| R01 | accuracy_vs_B01 | PASS | {"bbox_ap": true, "mask_ap": true, "semantic_miou": true} |
| S01 | accuracy_vs_B01 | PASS | {"bbox_ap": true, "mask_ap": true, "semantic_miou": true} |
| S02 | accuracy_vs_B01 | REJECT | {"bbox_ap": false, "mask_ap": false, "semantic_miou": true} |
| U02 | accuracy_vs_B01 | REJECT | {"bbox_ap": false, "mask_ap": false, "semantic_miou": false} |

## 타당성 한계

The 437-image split has been used repeatedly during candidate development. It is valid as a fixed comparative benchmark but is not an untouched confirmatory test set. Claims of final external generalization require a new held-out capture session.

정적 MAC/FLOP는 Conv·MatMul·Gemm의 dense 산술만 포함한 하한 추정치다. TensorRT layer fusion, 메모리 이동, resize·normalization·activation 비용은 포함하지 않으므로 실제 속도는 동일 장비의 측정 latency로 판단한다.
