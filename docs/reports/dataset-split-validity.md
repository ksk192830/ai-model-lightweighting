# Parking Front 데이터 분할 기준 및 정량 타당성 감사

## 최종 판정

Roboflow `parking_front` Version 8의 원본 annotation을 계승하되,
오프라인 증강과 전처리를 모두 빈 값으로 지정한 Version 9를 새로
생성했다. Version 9의 4,466장을 원래 Roboflow split과 무관하게 합친 뒤,
촬영 세션 단위로 다시 분할했다.

`front_session_split_v1`은 다음 조건을 모두 만족한다.

- 오프라인 증강 사용: 0장
- split 간 바이트 단위 동일 이미지: 0개
- split 간 촬영 세션 중복: 0개
- COCO 이미지·annotation 참조 오류: 0개
- 논문용 학습·평가 준비 판정: **PASS**

재현 가능한 감사 산출물은 다음과 같다.

- `docs/reports/metrics/dataset-split-audit.json`
- `docs/reports/metrics/dataset-split-summary.csv`
- `docs/reports/metrics/dataset-class-presence.csv`
- `docs/reports/metrics/dataset-session-summary.csv`
- `docs/reports/metrics/dataset-augmentation-audit.csv`
- `figures/dataset_split_validity.png`
- `figures/dataset_split_validity.pdf`

## 1. 무증강 원본의 정의

Roboflow Version 9는 Version 8의 설정을 그대로 재사용하지 않고,
preprocessing과 augmentation을 명시적으로 `{}`로 설정해 생성했다.
이를 `roboflow_export_manifest.json`에 원본 Version 8, 생성 Version 9,
설정 값과 함께 기록했다. 분할 스크립트는 이 manifest에서 무증강이
확인되지 않으면 기본적으로 실행을 중단한다.

이 정의에서 `무증강`은 다운로드 전 Roboflow가 생성한 회전·전단·
노이즈 등의 오프라인 파생 이미지가 없다는 뜻이다. 학습 시점의 온라인
증강은 train에만 적용할 수 있으며, valid/test에는 결정적
resize·normalize만 적용한다.

## 2. 촬영 세션의 정량적 정의

이미지 \(i\)의 원본 파일명이
`frame_<frame>_<date>_<time>_<microsecond>`이면 촬영 시각 \(t_i\)와
프레임 번호 \(f_i\)를 추출한다. 시간순으로 정렬한 두 이미지
\((i-1,i)\) 사이에서 다음 조건 중 하나를 만족하면 새 세션을 시작한다.

\[
\operatorname{new\_session}(i)=
\mathbb{1}\left[(t_i-t_{i-1})>\tau
\;\lor\;
(t_i\ne t_{i-1}\land f_i<f_{i-1})\right],
\qquad \tau=2.0\ \mathrm{s}.
\]

하나의 세션은 둘 이상의 split으로 나누지 않는다. 이 규칙은 연속 프레임이
train과 test에 동시에 들어가 성능이 과대 평가되는 것을 방지한다.

### 2.1 2초 임계값의 근거

양의 세션 내부 시간 간격 1,454개의 분포는 다음과 같다.

| 통계량 | 시간 간격 |
|---|---:|
| 중앙값 | 0.310 s |
| 95 백분위수 | 1.010 s |
| 99 백분위수 | 1.213 s |
| 세션 내부 최댓값 | 1.411 s |
| 관측된 세션 경계 최솟값 | 5.010 s |

따라서 \(1.411 < \tau < 5.010\)인 모든 임계값에서 동일한 7개 세션이
생성된다. 2.0초는 두 관측 분포 사이의 빈 구간에 있으므로 임계값의
작은 변화에 분할 결과가 바뀌지 않는다.

## 3. 추적 불가능 표본과 평가 세션

Version 9에서 타임스탬프를 복원할 수 있는 이미지는 1,461장이고,
`imageNNNNNN` 형식의 추적 불가능 이미지는 3,005장이다. 후자의 원본명은
1,337개이며, 873개가 반복되고 537개는 원래 Roboflow split을
가로지른다. 이름 숫자를 촬영 순서로 간주하면 서로 다른 원본 영상을
잘못 묶을 수 있으므로, 이 3,005장은 모두 train에만 고정했다.

타임스탬프 세션 크기는 1, 1, 165, 179, 225, 272, 618장이다.
평가 후보의 최소 크기를 \(n_g\ge20\)장으로 정했다. 실제로는
\(2\le n_{\min}\le165\)의 전 구간에서 동일한 5개 주요 세션이
평가 후보가 되므로, 20장이라는 단일 선택이 결과를 좌우하지 않는다.
1장짜리 세션 2개는 독립 평가 단위로 불안정해 train에 두었다.

Validation과 test에는 각각 최소 2개의 독립 세션을 배정하고,
train에도 최소 1개의 주요 타임스탬프 세션을 남겼다. 주요 세션이
5개이므로 train/valid/test의 주요 세션 수는 1/2/2개로 고정된다.

## 4. 세션 배정 목적함수

전체 이미지 수를 \(N\), split \(s\in\{tr,val,te\}\)의 이미지 수를
\(n_s\), 목표 비율을 \(r^*=(0.8,0.1,0.1)\)로 정의했다. 클래스
\(c\)가 한 이미지에 한 번 이상 존재하는 비율을 \(p_{s,c}\), 전체 데이터의
해당 비율을 \(p_c\)라 했다.

\[
D_{\infty}=\max_s\left|\frac{n_s}{N}-r_s^*\right|,
\qquad
D_1=\sum_s\left|\frac{n_s}{N}-r_s^*\right|,
\]

\[
D_{cls}(s)=\sum_{c\in\{out\_line,parking\_lot,parking\_space\}}
(p_{s,c}-p_c)^2.
\]

세션 완전성과 최소 세션 수를 만족하는 30개 배정을 전수 조사하고,
다음 벡터를 사전순으로 최소화했다.

\[
\operatorname*{lexmin}
\left[D_{\infty},D_1,D_{cls}(te),D_{cls}(val),D_{cls}(tr),
-\overline{t}_{te}\right].
\]

마지막 항은 앞의 모든 값이 같을 때만 더 최근 세션을 test로 선택하는
결정적 tie-breaker다. 따라서 같은 입력과 설정에서 결과가 항상 동일하다.

## 5. 최종 분할 결과

| split | 이미지 | 실제 비율 | 목표 비율 | 절대 편차 | timestamp 세션 |
|---|---:|---:|---:|---:|---:|
| train | 3,625 | 81.17% | 80% | 1.17%p | 3개(주요 1, 소규모 2) |
| valid | 404 | 9.05% | 10% | 0.95%p | 2개 |
| test | 437 | 9.79% | 10% | 0.21%p | 2개 |

최대 비율 편차 \(D_\infty\)는 1.17%p다. 정확한 80/10/10을 맞추기 위해
연속 프레임을 쪼개는 것보다, 세션 완전성과 누수 방지를 우선했다.

### 5.1 이미지 단위 조건 출현률

각 비율은 해당 객체가 한 번 이상 존재하는 이미지의 비율이다.
Negative는 세 평가 클래스의 annotation이 하나도 없는 이미지다.
클래스는 상호 배타적이지 않아 한 이미지가 여러 열에 포함될 수 있다.

| split | Negative | out_line | parking_lot | parking_space |
|---|---:|---:|---:|---:|
| train | 34.10% | 64.11% | 37.77% | 32.28% |
| valid | 22.77% | 76.73% | 62.13% | 45.30% |
| test | 37.53% | 58.81% | 40.50% | 36.38% |

네 조건의 train 대비 평균 절대 출현률 차이는 validation 15.33%p,
test 3.89%p다. Validation은 양성 객체 출현률이 더 높은 독립 세션으로
구성되며, test는 train과 상대적으로 구성이 가깝다. 이 차이가 난이도 차이를
뜻한다고 단정하지 않고 전체 mAP와 클래스별 AP를 함께 보고한다.

## 6. 검증 절차와 재현성

`create_leakage_safe_split.py`는 분할 직후 다음을 자동 검사한다.

1. 소스 manifest의 augmentation/preprocessing이 빈 값인지 확인한다.
2. 모든 이미지의 SHA-256을 계산해 split 간 동일 바이트를 검사한다.
3. 촬영 세션 ID가 split을 가로지르지 않는지 검사한다.
4. COCO image ID, annotation image ID와 실제 파일의 참조 무결성을 검사한다.
5. 분할 정책, 세션 배정, 클래스 통계와 검증 결과를
   `split_manifest.json`과 `dataset_readiness.json`에 기록한다.

최종 감사에서 오프라인 증강, 정확 중복, 세션 중복, COCO 참조 오류는
모두 0이었다. 이 결과는 `dataset-split-audit.json`에 PASS로 저장된다.

## 7. 평가 프로토콜과 한계

- Validation은 checkpoint 선택과 early stopping에만 사용한다.
- Test는 하이퍼파라미터, threshold, checkpoint 선택에 사용하지 않고
  학습 종료 후 최종 모델에 대해 한 번 평가한다.
- 연속 프레임을 개별 독립 표본으로 간주한 신뢰구간은 제시하지 않는다.
  신뢰구간이 필요하면 세션 단위 bootstrap을 사용한다.
- 현재 validation/test는 각각 2개의 촬영 세션만 포함한다. 따라서
  이 평가는 **현재 수집 환경 내 세션 분리 성능**을 측정하며, 새로운
  주차장·카메라·날씨로의 일반화를 보장하지 않는다. 외부 일반화는
  별도 장소의 추가 세션을 잠금 test set으로 수집해 확인해야 한다.
- 타임스탬프가 없는 3,005장은 평가에서 제외됐다. 이는 잘못된 세션
  추정으로 인한 누수를 막기 위한 보수적 선택이며, 학습 분포과 평가
  분포의 차이로 남는다.

## 8. 논문 본문 삽입용 문구

> Roboflow Version 8의 annotation을 계승하되 오프라인 증강과 전처리를
> 적용하지 않은 Version 9를 생성하였다. 데이터 누수를 방지하기 위해
> 개별 프레임이 아닌 촬영 세션 단위로 데이터를 분할하였다. 파일명에서
> 촬영 시각과 프레임 번호를 추출하고, 연속 프레임 간 시간 차가 2.0초를
> 초과하거나 프레임 번호가 초기화되는 지점에서 새 세션을 정의하였다.
> 세션 내부 시간 간격의 최댓값은 1.411초, 관측된 세션 경계 간격의
> 최솟값은 5.010초로, 선택한 임계값은 두 분포 사이의 비관측 구간에
> 위치한다. 평가에는 20장 이상의 세션을 validation과 test에 각각
> 두 개씩 완전한 단위로 배정하였다. 조건을 만족하는 30개 배정을 전수
> 조사하고 80:10:10 목표 비율의 최대·총 절대 편차, test·validation·
> train의 클래스 출현률 제곱 편차를 순차적으로 최소화하였다. 최종
> 분할은 train 3,625장(81.17%), validation 404장(9.05%), test
> 437장(9.79%)이며, split 간 동일 이미지, 촬영 세션 중복과 COCO 참조
> 오류는 모두 0개였다. 데이터 증강은 분할 후 train에만 적용하였다.

## 그림 캡션

**그림 X. 무증강 Parking Front 데이터의 촬영 세션 기반 분할 감사.**
(a) 80:10:10 목표 비율과 실제 분할 비율, (b) split별 이미지 단위
조건 출현률, (c) 세션 내부 간격 최댓값(1.411초)과 관측 경계 간격
최솟값(5.010초) 사이의 2초 임계값, (d) 촬영 세션 전체를 단일
split에 배정한 결과. 타임스탬프가 없는 3,005장은 누수 방지를 위해
train에만 포함하였다.
