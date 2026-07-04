# 문서 구조와 분류 기준

`docs/`는 문서가 설명하는 내용이 아니라 **독자가 문서를 사용하는 목적**에
따라 분류한다. 새 문서를 추가할 때도 아래 기준을 우선 적용한다.

## 디렉터리 기준

### `concepts/` — 무엇이며 왜 사용하는가

시간이 지나도 비교적 변하지 않는 원리, 용어, 평가 기준을 둔다.

- 기술 정의와 동작 원리
- 장단점과 적용 가능 조건
- 평가 지표의 수학적·실무적 의미
- 특정 실행 장비나 현재 진행 상태에 의존하지 않는 설명

현재 문서:

- [FP16](concepts/fp16.md)
- [INT8](concepts/int8.md)
- [NVIDIA 2:4 적용 가능성](concepts/2to4-eligibility.md)
- [평가 지표와 해석](concepts/evaluation-metrics.md)

### `guides/` — 어떻게 반복 실행하는가

다른 장비나 다음 실험에서도 재사용할 수 있는 절차를 둔다.

- 단계별 명령과 입력·출력
- 실험 순서와 통과 조건
- 학습, 변환, 복구, 이식 절차
- 실패 시 복구 방법

현재 문서:

- [전체 실험 워크플로](guides/experiment-workflow.md)
- [2:4 recovery fine-tuning](guides/2to4-fine-tuning.md)
- [학습 장비 이식](guides/training-portability.md)
- [TensorRT 노트북 생성·검증](guides/tensorrt-notebook-portability.md)

### `handoffs/` — 지금 무엇을 이어받아야 하는가

특정 시점의 저장소 상태와 다음 담당자의 행동을 연결하는 문서를 둔다.

- 완료·미완료 상태
- 현재 artifact 위치와 체크리스트
- 다음 담당자 또는 AI에게 전달할 실행 프롬프트
- 결과 재현에 필요한 현재 환경과 주의사항

현재 문서:

- [평가 작업 인수인계](handoffs/evaluation-handoff.md)
- [GPU 평가 실행 프롬프트](handoffs/gpu-evaluation-prompt.md)
- [고성능 데스크탑 recovery 프롬프트](handoffs/high-performance-recovery-prompt.md)
- [모델 artifact 인덱스](handoffs/model-artifact-index.md)

### `reports/` — 무엇을 관측하고 결정했는가

특정 실험 시점의 결과, 선정 근거, 논문용 서술을 둔다.

- 측정 결과와 비교표
- 후보 선정 또는 제외 결정
- 결과 해석과 논문 초안
- 재현 절차가 아니라 관측된 사실이 중심인 문서

현재 문서:

- [최종 TensorRT 엔진 선정](reports/final-engine-selection.md)
- [Experimental Results 초안](reports/results-draft.md)

## 경계가 애매할 때

한 문서에 개념·절차·결과가 함께 필요하면 주된 목적에 따라 위치를 정하고,
다른 성격의 상세 내용은 해당 디렉터리 문서로 링크한다.

- “2:4가 무엇인가?” → `concepts/`
- “2:4 모델을 어떻게 학습하는가?” → `guides/`
- “현재 어느 checkpoint까지 만들었는가?” → `handoffs/`
- “2:4 결과가 왜 탈락했는가?” → `reports/`

문서 제목에 `현재 상태`, `완료 결과`, `다음 작업`이 계속 늘어나면 개념
문서에 누적하지 말고 인수인계 또는 보고서로 분리한다.
