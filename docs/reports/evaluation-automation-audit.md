# 평가 기준 및 자동화 감사 보고서

- 감사 상태: **PASS_WITH_ADVISORIES**
- 기준 원본: `configs/experiments/defaults.yaml`
- 검사 수: 83
- 필수 실패: 0
- 실행 대기: 0
- 권고 사항: 9

## 감사 판정

| 영역 | 검사 | 상태 | 근거 |
|---|---|---|---|
| dataset | dataset-leakage-integrity | PASS | augmentation-free=True; cross-split duplicates=0; session overlap=0; COCO reference errors=0 |
| dataset | evaluation-image-count | PASS | configured=437; actual COCO images=437 |
| dataset | split-ratio-realization | PASS | train=3625 (81.17%), valid=404 (9.05%), test=437 (9.79%) |
| accuracy | evaluation-B01 | PASS | results/coco-evaluation/front-rfdetr-seg-large-v1-test.json; protocol and metrics valid |
| reproducibility | evaluation-provenance-B01 | WARN | legacy result lacks embedded annotation SHA-256; current annotation independently hashes to 8bb43b8f94b182b51b5ff4484cc04964d606a3eff53304fa01cc00b7fbfb3eba |
| accuracy | evaluation-C03 | PASS | results/coco-evaluation/C03-front-432-pth.json; protocol and metrics valid |
| accuracy | evaluation-C04 | PASS | results/coco-evaluation/C04-front-480-pth.json; protocol and metrics valid |
| accuracy | evaluation-M01 | PASS | results/coco-evaluation/M01-front-after-recovery.json; protocol and metrics valid |
| accuracy | evaluation-R01 | PASS | results/coco-evaluation/R01-front-432-pth.json; protocol and metrics valid |
| reproducibility | evaluation-provenance-R01 | WARN | legacy result lacks embedded annotation SHA-256; current annotation independently hashes to 8bb43b8f94b182b51b5ff4484cc04964d606a3eff53304fa01cc00b7fbfb3eba |
| accuracy | evaluation-R02 | PASS | results/coco-evaluation/R02-front-480-pth.json; protocol and metrics valid |
| accuracy | evaluation-R03 | PASS | results/coco-evaluation/R03-front-384-pth.json; protocol and metrics valid |
| accuracy | evaluation-S01 | PASS | results/coco-evaluation/S01-front-after-recovery.json; protocol and metrics valid |
| reproducibility | evaluation-provenance-S01 | WARN | legacy result lacks embedded annotation SHA-256; current annotation independently hashes to 8bb43b8f94b182b51b5ff4484cc04964d606a3eff53304fa01cc00b7fbfb3eba |
| accuracy | evaluation-S02 | PASS | results/coco-evaluation/S02-front-after-recovery.json; protocol and metrics valid |
| reproducibility | evaluation-provenance-S02 | WARN | legacy result lacks embedded annotation SHA-256; current annotation independently hashes to 8bb43b8f94b182b51b5ff4484cc04964d606a3eff53304fa01cc00b7fbfb3eba |
| accuracy | evaluation-S04 | PASS | results/coco-evaluation/S04-front-pth.json; protocol and metrics valid |
| accuracy | evaluation-U01 | PASS | results/coco-evaluation/U01-front-pth.json; protocol and metrics valid |
| accuracy | evaluation-U02 | PASS | results/coco-evaluation/U02-front-pth.json; protocol and metrics valid |
| reproducibility | evaluation-provenance-U02 | WARN | legacy result lacks embedded annotation SHA-256; current annotation independently hashes to 8bb43b8f94b182b51b5ff4484cc04964d606a3eff53304fa01cc00b7fbfb3eba |
| accuracy | evaluation-U03 | PASS | results/coco-evaluation/U03-front-pth.json; protocol and metrics valid |
| conversion | onnx-evaluation-B01-B01-front-baseline-onnx | PASS | results/coco-evaluation/B01-front-baseline-onnx.json; query count/parity valid |
| conversion | onnx-evaluation-C03-C03-front-432-onnx | PASS | results/coco-evaluation/C03-front-432-onnx.json; query count/parity valid |
| conversion | onnx-evaluation-C04-C04-front-480-onnx | PASS | results/coco-evaluation/C04-front-480-onnx.json; query count/parity valid |
| conversion | onnx-evaluation-R01-R01-front-432-onnx | PASS | results/coco-evaluation/R01-front-432-onnx.json; query count/parity valid |
| conversion | onnx-evaluation-R02-R02-front-480-onnx | PASS | results/coco-evaluation/R02-front-480-onnx.json; candidate decision=rejected; dataset parity failure recorded |
| conversion | onnx-evaluation-S01-S01-front-after-recovery-onnx | PASS | results/coco-evaluation/S01-front-after-recovery-onnx.json; query count/parity valid |
| conversion | onnx-evaluation-S02-S02-front-after-recovery-onnx | PASS | results/coco-evaluation/S02-front-after-recovery-onnx.json; query count/parity valid |
| conversion | graph-equivalence-B01 | PASS | artifacts/experiments/B01/front/onnx-equivalence.json; passed=True; samples=10; seed=42; mode=real-images; protocol match=True |
| conversion | graph-equivalence-schema-B01 | WARN | legacy graph report predates strict active-membership/class/score fields; full-dataset ONNX accuracy parity is available |
| conversion | graph-equivalence-C03 | PASS | artifacts/experiments/C03/front/onnx-equivalence.json; passed=True; samples=10; seed=42; mode=real-images; protocol match=True |
| conversion | graph-equivalence-C04 | PASS | artifacts/experiments/C04/front/onnx-equivalence.json; passed=True; samples=10; seed=42; mode=real-images; protocol match=True |
| conversion | graph-equivalence-M01 | PASS | artifacts/experiments/M01/front/onnx-equivalence.json; candidate decision=rejected; equivalence passed=False; protocol match=True |
| conversion | graph-equivalence-R01 | PASS | artifacts/experiments/R01/front/onnx-equivalence.json; passed=True; samples=10; seed=42; mode=real-images; protocol match=True |
| conversion | graph-equivalence-schema-R01 | WARN | legacy graph report predates strict active-membership/class/score fields; full-dataset ONNX accuracy parity is available |
| conversion | graph-equivalence-R02 | PASS | artifacts/experiments/R02/front/onnx-equivalence.json; candidate decision=rejected; equivalence passed=True; protocol match=True |
| conversion | graph-equivalence-S01 | PASS | artifacts/experiments/S01/front/onnx-equivalence.json; passed=True; samples=10; seed=42; mode=real-images; protocol match=True |
| conversion | graph-equivalence-S02 | PASS | artifacts/experiments/S02/front/onnx-equivalence.json; candidate decision=rejected; equivalence passed=True; protocol match=True |
| conversion | graph-equivalence-S03 | PASS | artifacts/experiments/S03/front/onnx-equivalence.json; candidate decision=rejected; equivalence passed=False; protocol match=True |
| conversion | graph-equivalence-U02 | PASS | artifacts/experiments/U02/front/onnx-equivalence.json; candidate decision=rejected; equivalence passed=True; protocol match=True |
| static-analysis | static-analysis-B01 | PASS | artifacts/experiments/B01/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-B02 | PASS | artifacts/experiments/B02/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-B03 | PASS | artifacts/experiments/B03/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-U01 | PASS | artifacts/experiments/U01/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-U02 | PASS | artifacts/experiments/U02/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-U03 | PASS | artifacts/experiments/U03/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-M01 | PASS | artifacts/experiments/M01/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-M02 | PASS | artifacts/experiments/M02/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-S01 | PASS | artifacts/experiments/S01/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9659090909090909; estimated/all-node fraction=0.0869120654396728 |
| static-analysis | static-analysis-S02 | PASS | artifacts/experiments/S02/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9691358024691358; estimated/all-node fraction=0.08805384183959619 |
| static-analysis | static-analysis-S03 | PASS | artifacts/experiments/S03/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-S04 | PASS | artifacts/experiments/S04/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-C01 | PASS | artifacts/experiments/C01/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9659090909090909; estimated/all-node fraction=0.0869120654396728 |
| static-analysis | static-analysis-C02 | PASS | artifacts/experiments/C02/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9659090909090909; estimated/all-node fraction=0.0869120654396728 |
| static-analysis | static-analysis-C03 | PASS | artifacts/experiments/C03/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9659090909090909; estimated/all-node fraction=0.0869120654396728 |
| static-analysis | static-analysis-C04 | PASS | artifacts/experiments/C04/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9659090909090909; estimated/all-node fraction=0.0869120654396728 |
| static-analysis | static-analysis-R01 | PASS | artifacts/experiments/R01/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-R02 | PASS | artifacts/experiments/R02/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-R03 | PASS | artifacts/experiments/R03/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.08603667136812412 |
| static-analysis | static-analysis-Q01 | PASS | artifacts/experiments/Q01/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=1.0; estimated/all-node fraction=0.05157437567861021 |
| static-analysis | static-analysis-Q02 | PASS | artifacts/experiments/Q02/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.0553371635923798 |
| static-analysis | static-analysis-Q03 | PASS | artifacts/experiments/Q03/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.9631578947368421; estimated/all-node fraction=0.07966913365259033 |
| static-analysis | static-analysis-Q04 | PASS | artifacts/experiments/Q04/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.1368421052631579; estimated/all-node fraction=0.005195843325339729 |
| static-analysis | static-analysis-Q05 | PASS | artifacts/experiments/Q05/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.6789473684210526; estimated/all-node fraction=0.037240184757505776 |
| static-analysis | static-analysis-Q06 | PASS | artifacts/experiments/Q06/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.25263157894736843; estimated/all-node fraction=0.010282776349614395 |
| static-analysis | static-analysis-Q07 | PASS | artifacts/experiments/Q07/front/static-analysis.json; status=partial-lower-bound; resolved Conv/MatMul/Gemm coverage=0.18421052631578946; estimated/all-node fraction=0.007219471947194719 |
| static-analysis | stage1-all-candidate-coverage | PASS | registered=26; rows=26; unperformed=0; missing=none; extra=none; static-only=True |
| selection | deployment-30fps-median-budget | PASS | target_fps=30; median_latency_max_ms=33.333333 |
| selection | deployment-p95-diagnostic-only | PASS | P95 warns above the 30 FPS frame budget and is not a hard gate |
| selection | latency-cv-single-remeasurement-policy | PASS | CV > 5% triggers one full three-repetition retry; unresolved candidates remain in Pareto evidence but not final recommendations |
| automation | desktop-pipeline-protocol-binding | PASS | historical desktop queue is superseded by the terminal notebook Stage-2 result |
| automation | post-recovery-protocol-binding | PASS | historical M01 recovery queue is superseded by the completed Stage-1 artifact and notebook Stage-2 result |
| latency | latency-results | PASS | official notebook benchmark evidence=63/63 |
| automation | engine-summary | PASS | official notebook summary contains 21 completed engine rows |
| automation | pipeline-stage-wiring | PASS | engine inspection, paper table/figure, static report, and final audit are wired |
| automation | all-candidate-notebook-pareto-wiring | PASS | Stage-1 eligible set -> per-candidate terminal Stage 2 -> Pareto gate is wired |
| automation | notebook-single-excel-report-wiring | PASS | raw, analyzed, per-class, repeated latency, failure, protocol, and environment sheets are wired |
| automation | notebook-stage2-result-completeness | PASS | 21 completed rows include repeated/runtime/class-wise evidence |
| automation | notebook-terminal-report-gate | PASS | terminal=True; Excel=True; Pareto=True |
| automation | paper-report-source | PASS | paper outputs consume the frozen notebook Stage-3 Pareto result |
| automation | int8-calibration-binding | PASS | train-only deterministic calibration: count=128, seed=42 |
| study-design | fixed-benchmark-reuse | WARN | 437 images are a repeated comparative benchmark, not an untouched confirmatory test; collect a new session for external-generalization claims |
| study-design | training-seed-replication | WARN | current recovery results use seed 42 only; report this as a deterministic engineering comparison or add multiple training seeds for variance estimates |

## 사전 정의된 후보 수용 기준의 적용 결과

`REJECT`는 자동화 결함이 아니라 해당 후보가 정확도 보존 기준을 통과하지 못했다는 뜻이다.

| 후보 | 비교 | 판정 | Gate details |
|---|---|---|---|
| C03 | efficiency_vs_B01 | PASS | {"dense_macs": true, "onnx_nodes": true, "onnx_size": true} |
| C04 | efficiency_vs_B01 | PASS | {"dense_macs": true, "onnx_nodes": true, "onnx_size": false} |
| R01 | efficiency_vs_B01 | PASS | {"dense_macs": true, "onnx_nodes": false, "onnx_size": false} |
| R02 | efficiency_vs_B01 | PASS | {"dense_macs": true, "onnx_nodes": false, "onnx_size": false} |
| R03 | efficiency_vs_B01 | PASS | {"dense_macs": true, "onnx_nodes": false, "onnx_size": false} |
| S01 | efficiency_vs_B01 | PASS | {"dense_macs": false, "onnx_nodes": true, "onnx_size": false} |
| S02 | efficiency_vs_B01 | PASS | {"dense_macs": false, "onnx_nodes": true, "onnx_size": true} |
| S03 | efficiency_vs_B01 | REJECT | {"dense_macs": false, "onnx_nodes": false, "onnx_size": false} |
| S04 | efficiency_vs_B01 | PASS | {"dense_macs": false, "onnx_nodes": false, "onnx_size": true} |
| U01 | efficiency_vs_B01 | REJECT | {"dense_macs": false, "onnx_nodes": false, "onnx_size": false} |
| U02 | efficiency_vs_B01 | REJECT | {"dense_macs": false, "onnx_nodes": false, "onnx_size": false} |
| U03 | efficiency_vs_B01 | REJECT | {"dense_macs": false, "onnx_nodes": false, "onnx_size": false} |
| C03 | desktop_preliminary_accuracy_vs_B01 | PASS | {"bbox_ap": true, "mask_ap": true, "semantic_miou": true} |
| C04 | desktop_preliminary_accuracy_vs_B01 | PASS | {"bbox_ap": true, "mask_ap": true, "semantic_miou": true} |
| M01 | desktop_preliminary_accuracy_vs_B01 | REJECT | {"bbox_ap": false, "mask_ap": false, "semantic_miou": false} |
| R01 | desktop_preliminary_accuracy_vs_B01 | PASS | {"bbox_ap": true, "mask_ap": true, "semantic_miou": true} |
| R02 | desktop_preliminary_accuracy_vs_B01 | PASS | {"bbox_ap": true, "mask_ap": true, "semantic_miou": true} |
| R03 | desktop_preliminary_accuracy_vs_B01 | REJECT | {"bbox_ap": true, "mask_ap": false, "semantic_miou": true} |
| S01 | desktop_preliminary_accuracy_vs_B01 | PASS | {"bbox_ap": true, "mask_ap": true, "semantic_miou": true} |
| S02 | desktop_preliminary_accuracy_vs_B01 | REJECT | {"bbox_ap": false, "mask_ap": false, "semantic_miou": true} |
| S04 | desktop_preliminary_accuracy_vs_B01 | REJECT | {"bbox_ap": false, "mask_ap": false, "semantic_miou": false} |
| U01 | desktop_preliminary_accuracy_vs_B01 | PASS | {"bbox_ap": true, "mask_ap": true, "semantic_miou": true} |
| U02 | desktop_preliminary_accuracy_vs_B01 | REJECT | {"bbox_ap": false, "mask_ap": false, "semantic_miou": false} |
| U03 | desktop_preliminary_accuracy_vs_B01 | REJECT | {"bbox_ap": false, "mask_ap": false, "semantic_miou": false} |

## 타당성 한계

The 437-image split has been used repeatedly during candidate development. It is valid as a fixed comparative benchmark but is not an untouched confirmatory test set. Claims of final external generalization require a new held-out capture session.

정적 MAC/FLOP는 Conv·MatMul·Gemm의 dense 산술만 포함한 하한 추정치다. TensorRT layer fusion, 메모리 이동, resize·normalization·activation 비용은 포함하지 않으므로 실제 속도는 동일 장비의 측정 latency로 판단한다.
