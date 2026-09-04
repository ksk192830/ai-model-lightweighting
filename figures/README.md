# 그림

보고서와 논문에 삽입하는 PNG/PDF를 둔다. 원시 수치를 손으로 수정하지 않고
`scripts/reporting/` 또는 데이터 감사 스크립트에서 다시 생성한다.

`dataset_split_validity.*`는 무증강 세션 분할의 이미지 수와 비율을 보여준다.
논문 본문용 그림은 `paper/`, 상세 검증 그림은 `appendix/`에 둔다. 각 그림은
인쇄·본문 삽입용 PDF와 Notion·README 미리보기용 300 dpi PNG를 같은 이름으로
관리한다. 배치 위치와 캡션은 `../docs/paper/figure-placement-plan.md`를 따른다.

전체 논문 그림은 다음 명령으로 다시 생성한다.

```bash
.venv/bin/python scripts/data_preparation/audit_dataset_split.py
.venv/bin/python scripts/reporting/generate_paper_visuals.py
```

`paper/09_qualitative_source_onnx.*`는 최종 TensorRT engine이 아니라 B01·C01·R01에
대응하는 source ONNX graph의 정성 출력이다. 이 제한은 그림 캡션과 manifest에도
기록되어 있다. 기존 `stage3_*.png` 파일은 자동 생성 보고서의 호환성을 위해
보존한다.
