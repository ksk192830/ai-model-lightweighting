# Parking Front 무증강·누수 방지 데이터 분할

## 사용할 최종 데이터셋

```text
data/training/front_session_split_v1/
```

이 데이터셋은 Roboflow `parking_front` Version 9 COCO Segmentation
export를 소스로 사용한다. Version 9는 Version 8의 annotation을 계승하고,
preprocessing과 augmentation을 모두 빈 값으로 지정해 생성한 무증강
버전이다. 원본 Roboflow split을 전부 합친 후 촬영 세션 단위로
새로 분할했다.

| split | 이미지 | 비율 | 촬영 세션 | 용도 |
|---|---:|---:|---:|---|
| train | 3,625 | 81.17% | 주요 1 + 1장 세션 2 | 파라미터 학습 |
| valid | 404 | 9.05% | 독립 2 | checkpoint 선택·조기 종료 |
| test | 437 | 9.79% | 독립 2 | 학습 종료 후 최종 1회 평가 |

목표 80/10/10 대비 최대 절대 편차는 1.17%p이다.

## 분할 규칙

1. `frame_<프레임>_<날짜>_<uc2dc각>_<마이크로초>` 파일명에서 시각과
   프레임 번호를 추출한다.
2. 시간 간격이 2.0초를 초과하거나 프레임 번호가 초기화되면 새 세션으로
   구분한다.
3. 세션은 절대 둘 이상의 split으로 나누지 않는다.
4. 20장 미만의 세션은 평가 단위로 사용하지 않고 train에 고정한다.
5. 촬영 시각을 복원할 수 없는 `imageNNNNNN` 3,005장은 잘못된 세션
   추정을 피하기 위해 train에 고정한다.
6. valid/test에 각각 독립 세션을 최소 2개 배정하고, train에도
   주요 타임스탬프 세션을 1개 이상 남겨 성능 편향을 완화한다.
7. 가능한 30개 세션 배정을 전수 조사해 목표 비율 편차와 클래스 출현률
   편차를 순차적으로 최소화한다.
8. 완성 후 무증강 manifest, 이미지 SHA-256, 세션 중복, COCO 참조를
   전부 검사한다.

2초 임계값은 세션 내부 간격 최댓값 1.411초와 관측 경계 간격 최솟값
5.010초 사이의 빈 구간에 있다. 따라서 1.411–5.010초 사이에서
임계값을 변경해도 동일한 7개 세션이 생성된다.

## 재생성

원본 무증강 export가 있는 상태에서 저장소 루트에서 실행한다.

```bash
.venv/bin/python scripts/data_preparation/create_leakage_safe_split.py
```

기본 경로는 다음과 같다.

- 소스: `data/training/front_unaugmented`
- 출력: `data/training/front_session_split_v1`
- 결과 manifest: `data/training/front_session_split_v1/split_manifest.json`
- readiness: `data/training/front_session_split_v1/dataset_readiness.json`

이미지는 기본적으로 hard link로 연결해 추가 디스크 사용량을 줄인다.
독립 복사본이 필요하면 `--copy-images`를 사용한다.

## 사용 주의사항

- 증강은 학습 로더의 train 분기에서만 적용한다.
- valid로 하이퍼파라미터와 checkpoint를 선택하고 test는 최종 한 번만 사용한다.
- 현재 test는 서로 다른 두 촬영 세션을 포함하지만 새 주차장에 대한 외부
  일반화 test는 아니다.

수식, 통계, 한계와 논문 삽입용 문구는
`docs/reports/dataset-split-validity.md`에 정리했다.
