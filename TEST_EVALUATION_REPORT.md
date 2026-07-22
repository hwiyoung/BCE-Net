# Frozen baseline test 평가 보고서

## 결론

`checkpoint-best.pth`를 고정하고, manifest상 train에 배정되지 않은 test
100장을 threshold `0.5`로 평가했다. 이후 spatial overlap audit에서 서로
다른 split의 일부 patch가 실제 픽셀을 공유함을 확인했으므로, 이 결과를
엄격한 spatially-independent test라고 표현하지 않는다. 평가 대상은
논문의 완전 재현 모델이 아니라 **BCE-Net 기반 noisy-label robust
baseline**이다.

validation과 직접 비교하기 위해 각 1024×1024 원본의 중앙 512×512
crop만 평가했다. 이번 수치에는 1024 sliding-window/full-tile 결과가
포함되지 않는다.

핵심 결과:

- omission F1: `0.638687`
- excess F1: `0.805987`
- omission/excess macro F1: `0.722337`
- combined change F1: `0.754748`

test는 모델이나 threshold 선택에 사용하지 않았다.

## 사후 확인된 spatial overlap 제한

`spatial_group`은 split 사이에 공유되지 않고 sample ID와 파일 경로도 모두
고유하지만, 8,192픽셀 group 경계를 가로지르는 1024 patch가 존재한다.

- train–validation source overlap: 36 pairs
- train–test source overlap: 45 pairs
- validation–test source overlap: 7 pairs
- 총 88 pairs에서 image/map/label 교차 픽셀이 모두 100% 동일
- test 24장은 train 또는 validation의 1024 source 영역과 겹침
- train crop 가능 범위 또는 validation 중앙 crop과 test 중앙 평가 영역이
  겹칠 수 있는 test는 8장
- validation `excess_00323`과 test `excess_00324`의 중앙 512 crop은
  204,885 pixels, 즉 78.16%가 동일

따라서 아래 수치는 file-disjoint frozen test metric이지만 strict spatial
generalization metric은 아니다. 근거는
`training_monitor/test-split-overlap-audit-20260722/REPORT.md`에 보존했다.

## 고정 평가 조건

| 항목 | 값 |
|---|---|
| checkpoint | `/home/work/models/BCE-Net/map-ortho-robust-v2-20260720/checkpoint-best.pth` |
| checkpoint epoch | 24 (checkpoint 내부 zero-based 23) |
| checkpoint SHA-256 | `4c944eac1bb80b76339c438cea6cddbddc133e30d8f6806b27229311080633ac` |
| split | manifest의 기존 `test` 100장 |
| manifest SHA-256 | `1d3a9f482a2fc01a5d5142449458015760085ecd871b9ca5ea13ad3e6f9bdf2c` |
| 입력 범위 | 1024×1024 원본의 중앙 512×512, origin `(256, 256)` |
| 총 평가 픽셀 | 26,214,400 |
| threshold | validation과 같은 `0.5` |
| augmentation | 없음 |
| sliding-window | 사용하지 않음 |

validation도 같은 중앙 512×512 crop을 사용하므로 아래 비교는 평가
공간이 같다. 반면 향후 1024 sliding-window 결과는 네 배의 원본 공간과
window blending 영향을 포함하므로 이 표에 섞으면 안 된다.

## test 정량 결과

| class | precision | recall | F1 | IoU | prediction rate | target rate |
|---|---:|---:|---:|---:|---:|---:|
| omission | 0.681656 | 0.600815 | 0.638687 | 0.469170 | 0.032104 | 0.036424 |
| excess | 0.783668 | 0.829615 | 0.805987 | 0.675024 | 0.079813 | 0.075393 |
| combined | 0.754409 | 0.755088 | 0.754748 | 0.606101 | 0.111917 | 0.111817 |

원시 집계:

| class | TP | FP | FN |
|---|---:|---:|---:|
| omission | 573,677 | 267,916 | 381,155 |
| excess | 1,639,634 | 452,623 | 336,745 |
| combined | 2,213,322 | 720,528 | 717,889 |

## validation과 비교

best epoch 24의 validation과 동일한 평가 범위에서 비교했다.

| metric | validation | test | test - validation |
|---|---:|---:|---:|
| omission F1 | 0.758699 | 0.638687 | -0.120012 |
| excess F1 | 0.701018 | 0.805987 | +0.104969 |
| macro F1 | 0.729859 | 0.722337 | -0.007521 |
| combined F1 | 0.736561 | 0.754748 | +0.018187 |

macro F1은 validation 대비 약 0.75 percentage point 낮아 전체적인 두
branch 균형은 비슷하다. 그러나 이를 클래스별 안정성으로 해석하면 안
된다. test target rate는 omission `3.64%`, excess `7.54%`이고 validation은
각각 `7.21%`, `4.02%`로 클래스 구성이 크게 다르다. test에서는 omission
precision과 recall이 모두 낮아지고 excess는 모두 높아졌다. combined
target rate는 두 split이 약 `11.2%`로 비슷하기 때문에 combined 지표만
보면 이 이동이 가려진다.

## 정성 오류 검토

상위 FP/FN 패널은 오류가 큰 샘플을 찾는 도구이지 clean-label 판정을
자동 확정하는 도구가 아니다. 아래 분류는 before 영상, after footprint,
GT와 prediction을 함께 본 1차 육안 판독이며, 최종 확정에는 사람의 원본
해상도 검수가 필요하다.

라벨 오류 또는 누락이 강하게 의심되는 예:

- `omission_01017`, `omission_01018`: GT omission 사각형 위치가 육안상
  빈 초지이며 건물로 보이지 않는다. 큰 FN을 곧바로 모델 오류로 볼 수
  없다.
- `omission_00958`: omission GT의 큰 부분이 테니스 코트에 놓여 있다.
- `omission_00962`, `omission_00918`, `omission_00954`: reference 밖의
  명확한 지붕 형태를 모델이 omission으로 검출했지만 GT에 없어 큰 FP로
  집계된다. omission 라벨 누락 가능성이 높다.
- `omission_00953`: 영상에서는 비건물/공사 영역으로 보이는 큰 reference
  footprint가 unchanged GT로 표시되지만 모델은 excess로 검출한다.
- `excess_00017`, `excess_00290`: 공사 또는 비건물로 보이는 reference
  영역 중 GT에 포함되지 않은 부분까지 모델이 excess로 검출한다. excess
  라벨의 부분 누락 가능성이 있다.

모델 오류가 실제로 보이는 예:

- `omission_00756`, `omission_00758`: 건물로 보이는 omission GT 내부를
  부분적으로만 채우거나 끊어서 예측해 객체 내부 FN과 경계 누락이 생긴다.
- `excess_00321`: 일부 visible roof/reference 일치 영역도 excess로 넓게
  검출하는 양상이 있어 실제 FP 가능성이 있다.

판정이 불확실한 예:

- `excess_00191`, `excess_00319`, `excess_00298`은 그림자, off-nadir roof,
  공사 상태와 polygon 정합이 함께 얽혀 있다. 현재 패널만으로 GT와 모델
  중 어느 쪽이 잘못됐는지 확정하지 않는다.

따라서 보고된 수치는 **현재 noisy test label에 대한 pixel metric**이다.
clean-label 일반화 성능으로 재해석하지 않으며, FP/FN 상위 샘플의 수동
audit 결과는 원본 라벨을 덮어쓰지 않는 별도 CSV로 관리해야 한다.

## 산출물

평가 디렉터리:

```text
/home/work/models/BCE-Net/map-ortho-robust-v2-test-center512-best-e24-20260721
```

VS Code 링크:

```text
training_monitor/test-center512-best-e24-20260721
```

주요 파일:

- `metrics.json`: protocol, provenance, 전체 지표
- `per_sample_metrics.csv`: 100장별 TP/FP/FN과 지표
- `qualitative/fixed.png`: 고정 omission/excess 샘플
- `qualitative/top_{omission,excess,combined}_{fp,fn}.png`: 오류 상위 패널
- `qualitative/samples/*.png`: 선택된 샘플의 개별 패널
- `qualitative/selections.json`: 정성 샘플 선정 근거와 수치

CSV 합계와 JSON 전체 집계가 세 클래스에서 모두 일치함을 재검산했다.
평가 전후 checkpoint와 manifest SHA-256도 동일하다. 기존 학습 디렉터리의
checkpoint, metrics, 정성 결과는 수정하지 않았다.

### Pilot label audit 패키지

30장 two-stage blinded 검수 자료는 다음 위치에 있다.

```text
training_monitor/test-pilot-audit-v1-20260721
```

- `01_stage1_gt_only`: prediction이 없는 GT 우선 검수 이미지
- `02_stage2_predictions`: stage 1 완료 후 확인할 prediction/error 이미지
- `source_crops`: lossless before/reference/state crop
- `pilot_review.csv`: 사용자가 작성해 반환할 검수 양식
- `03_unblind/selection_manifest.csv`: 마지막에 확인할 실제 ID와 선정 근거

표본은 error-ranked 16장, high-agreement control 4장, seeded random control
10장이다. error-ranked 표본만으로 성능을 추정하지 않고 대조군과 함께
라벨 오류 유형 및 metric 적격성을 판정한다.

pilot을 제외한 나머지 70장 패키지:

```text
training_monitor/test-audit-remaining70-v1-20260722
```

pilot 30장과 remaining 70장의 실제 sample ID 합집합은 test 100장과 정확히
같고 교집합은 없다.

canonical full-100 검수 패키지:

```text
training_monitor/test-audit-full100-v1-20260722
```

작성 대상은 `full100_review.csv` 하나다. 기존 `P001..P030`과
`R001..R070` ID를 유지해 앞선 두 패키지와 대응이 이어진다.

## Formula (7) 수정 구현 계획

이번 test 평가에서는 loss를 호출하지 않았고 Formula (7) 관련 모델/loss
코드를 수정하지 않았다. 다음 구현은 별도 변경과 별도 학습 디렉터리로
진행한다.

### 1. feature 반환 계약 명시

- `Baseline34.forward`의 익명 `feat_all`, `feat_mov` 두 값을 더 이상
  Formula (7)에 재사용하지 않는다.
- 동일한 공간 해상도와 channel 의미를 갖는 `F`, `F_BG`, `F_FG`를 명시적
  이름으로 반환한다.
- `F_BG = (1-M) × F`, `F_FG = M × (1-sigmoid(F))`의 mask resize 방식,
  sigmoid 적용 위치, feature shape를 forward contract와 테스트에 고정한다.
- segmentation logits의 기존 의미와 순서는 회귀 테스트로 보존하되,
  contrastive feature는 named structure로 분리한다.

### 2. Formula (7) pair 교정

- 신규/omission term은 `D(F_BG, F)`만 사용한다.
- 제거/excess term은 `D(F, F_FG)`만 사용한다.
- 현재처럼 omission과 excess에 하나의 `(feature_all, feature_split)` pair를
  재사용하는 경로는 legacy profile에만 남기고 신규 profile에서 제거한다.

### 3. bounding-box patch cosine similarity

- target의 connected component마다 bounding box를 구한다.
- 현재 코드처럼 component pixel만 boolean-select하지 않고 box의 전체
  `C×H×W` rectangular patch를 flatten해 cosine similarity를 계산한다.
- box 좌표는 label에서만 결정하고 tensor slicing은 graph를 유지한다.
- paper-faithful profile의 projection kernel, normalization, sigmoid 순서는
  논문 정의에 맞춰 별도 모듈로 고정한다.

### 4. 신규와 제거의 독립 평균

다음 두 항을 별도로 평균한 뒤 합한다.

```text
L_new     = mean_n 0.5 × (1 - D(F_BG[n], F[n]))
L_removed = mean_r 0.5 × (1 + D(F[r], F_FG[r]))
L_con     = L_new + L_removed
```

현재처럼 모든 신규/제거 instance term을 한 list에서 평균하지 않는다.
한 class의 instance가 없는 batch에서의 differentiable zero 처리와 전체
loss scaling도 명시적으로 테스트한다.

### 5. 단위 및 gradient 테스트

- 손계산 tensor로 cosine `1`, `0`인 신규/제거 항의 예상 loss를 검증한다.
- irregular component에서 component pixel 수가 아니라 bounding-box 전체
  patch가 사용되는지 검증한다.
- 신규 1개/제거 3개처럼 instance 수가 다른 경우 pooled mean이 아니라
  독립 평균의 합인지 검증한다.
- 신규 또는 제거가 0개인 batch, min-size box, multi-image batch를 검증한다.
- double precision `gradcheck`와 backward test로 `F`, `F_BG`, `F_FG`의
  gradient가 finite/non-zero인지 확인한다.
- pair 독립성을 위해 new term이 `F_FG`에, removed term이 `F_BG`에 잘못된
  gradient를 만들지 않는지 검증한다.

### 6. 실험 profile 분리

- `legacy-robust-v2`: 현재 완료 checkpoint를 재현하기 위한 기존 2-feature
  contrastive와 GCE/가중치 설정. 기존 결과는 계속 frozen reference로 둔다.
- `paper-faithful-formula7`: 세 feature, 정확한 pair/bounding-box/독립 평균,
  BCE+Dice와 비가중 설정을 사용한다.
- `robust-formula7`: 교정된 Formula (7)은 유지하되 GCE, positive/boundary/
  secondary weights를 적용한다.

모든 profile 이름과 Formula (7) mode를 config/checkpoint/metrics에 기록한다.
`paper-faithful-formula7`도 random sample generator 등 다른 미구현 요소가
남아 있는 동안은 “논문 BCE-Net 정확 재현”이라고 부르지 않는다.

### 7. 구현 후 검증 순서

1. 위 손계산/gradient/forward contract 테스트
2. 기존 legacy profile의 segmentation output 및 loss 회귀 테스트
3. 작은 train/validation smoke run
4. buffered spatial split v2에서 paper-faithful와 robust baseline을 모두
   별도 디렉터리에서 실행
5. v2 train/validation에서 설정을 확정한 뒤에만 v2 test를 한 번 평가

이번 baseline test 결과를 보고 Formula (7) 구현이나 threshold를 조정하지
않는다.
