# 실험 설정

`defaults.yaml`은 모든 실험에 공통인 평가·변환 프로토콜이고,
`registry.yaml`은 후보별 차이와 상태를 기록한다.

ID 접두사는 B=baseline/precision, R=resolution, U=unstructured pruning,
S=structured pruning, M=2:4 semi-structured, C=결합 후보, Q/W=양자화 탐색을
뜻한다. `status`는 실행 상태이고 `result.decision`은 실험 판정이므로 서로
바꿔 쓰지 않는다.

후보 추가 시 가설, source artifact, 의존성, 목표 precision을 먼저 등록한 뒤
`scripts/experiments/` 진입점을 사용한다. 정확도 수용 한계와 latency 측정
횟수는 결과를 본 뒤 임의로 바꾸지 않는다.
