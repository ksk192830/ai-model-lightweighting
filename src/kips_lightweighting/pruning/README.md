# Pruning 구현

- `unstructured.py`: global magnitude sparsity
- `structured.py`: decoder layer 및 FFN 구조 축소
- `sparse_2to4.py`: NVIDIA 2:4 패턴 적용과 적격성 검사

구조가 바뀌는 후보는 checkpoint load 검증, ONNX 재수출, PTH–ONNX 동등성,
전체 정확도 평가를 모두 거쳐야 한다. 0 weight 비율만 증가한 경우 dense
ONNX의 MAC/FLOP 감소로 해석하지 않는다.
