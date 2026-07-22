# BCE-Net robust baseline on spatial split v2

## 결론

기존 v1 manifest와 모든 checkpoint를 보존한 채, 1024 source patch의 실제
공간 중복을 차단하는 buffered spatial split v2를 별도로 만들었다. 현재
robust baseline을 공개 WHU checkpoint에서 동일 설정으로 100 epoch 다시
학습했고, validation으로 선택한 best checkpoint를 test에 한 번만 적용했다.

- cross-split source 1024 overlap: **0 pairs**
- cross-split 256px buffer violations: **0 pairs**
- best epoch: **46**
- validation macro omission/excess F1: **0.738586**
- test macro omission/excess F1: **0.722276**
- test combined F1: **0.722043**

이 결과는 **BCE-Net 기반 noisy-label robust baseline**의 결과이며 논문
BCE-Net의 정확 재현 성능이 아니다. Formula (7) 코드는 수정하지 않았다.

## Spatial split v2

생성 파일:

```text
dataset/map_ortho_manifest_spatial_v2.csv
dataset/map_ortho_spatial_split_v2_report.json
dataset/map_ortho_spatial_split_v2_report.md
```

감사 결과:

```text
training_monitor/spatial-v2-audit-20260722
```

각 1024 patch를 동일 source mosaic의 `xoff/yoff` rectangle로 정의했다.
두 rectangle이 겹치거나 x/y axis-aligned gap이 모두 256px 미만이면 같은
component로 연결하고, component 전체를 하나의 split에만 배정했다.

| buffer | components | largest component | total의 비율 |
|---:|---:|---:|---:|
| 0px | 313 | 66 | 6.6% |
| 128px | 267 | 73 | 7.3% |
| **256px** | **215** | **73** | **7.3%** |
| 512px | 140 | 276 | 27.6% |
| 1024px | 78 | 544 | 54.4% |

512px부터 하나의 component가 전체의 27.6%로 급증한다. 분할 가능성과
주변 문맥 분리를 함께 고려해 256px를 사용했다. GSD 0.12m 기준 30.72m다.

| split | samples | components | center omission/excess | omission target rate | excess target rate | combined target rate |
|---|---:|---:|---:|---:|---:|---:|
| train | 800 | 146 | 372/428 | 0.054595 | 0.046848 | 0.101443 |
| validation | 100 | 46 | 46/54 | 0.055910 | 0.046947 | 0.102857 |
| test | 100 | 23 | 47/53 | 0.054137 | 0.047201 | 0.101338 |

독립 재검산 결과:

- train–validation source overlap: 0
- train–test source overlap: 0
- validation–test source overlap: 0
- train possible-crop union과 validation/test center crop overlap: 0
- validation/test center crop overlap: 0
- 실제 최소 axis-aligned gap: train–val 262px, train–test 276px,
  val–test 416px
- duplicate ID/path와 cross-split shared component: 0

v1과 v2의 test는 4장만 같고, validation은 11장만 같다. 따라서 v1/v2
수치를 같은 표본에 대한 전후 성능처럼 직접 차감하지 않는다.

## 재학습

출력:

```text
/home/work/models/BCE-Net/map-ortho-robust-spatial-v2-20260722
training_monitor/robust-spatial-v2-20260722
```

초기 checkpoint:

```text
/home/work/models/BCE-Net/checkpoint-best-whu.pth
sha256=ea2eb4ed490fc42586678103340be7f8bcc7cd0ba337dd6f8dbde24cce16f6f7
```

v1에서 이미 사용자 데이터 test 후보를 보았을 가능성이 있는 checkpoint는
초기화에 사용하지 않았다. 학습 설정은 v1 robust baseline과 동일하다.

- input: train 512 crop, center origin에 x/y ±128 jitter
- GCE q=0.7 + weighted Dice
- omission/excess positive weight=4
- boundary weight=0.25, secondary change weight=0.5
- contrastive weight=1, 현재 legacy Formula (7)
- SGD, lr=0.001, cosine, 100 epochs, batch size 4
- seed=1024, threshold=0.5
- best selection=validation omission/excess macro F1

best checkpoint SHA-256:

```text
150d5ffc63e2bbe9d347c1a1be8a45cd70b379218ec84ecef99e68b32b22a100
```

## Validation과 frozen test

test 출력:

```text
/home/work/models/BCE-Net/map-ortho-robust-spatial-v2-test-center512-best-e46-20260722
training_monitor/test-spatial-v2-center512-best-e46-20260722
```

평가 범위는 validation과 동일한 중앙 512 crop이다. 1024 sliding-window
수치와 섞지 않았으며, checkpoint와 threshold는 test 결과로 선택하지 않았다.

| split/class | precision | recall | F1 | IoU | prediction rate | target rate |
|---|---:|---:|---:|---:|---:|---:|
| validation omission | 0.834885 | 0.776658 | 0.804720 | 0.673248 | 0.052011 | 0.055910 |
| validation excess | 0.606292 | 0.754822 | 0.672453 | 0.506538 | 0.058448 | 0.046947 |
| validation combined | 0.713928 | 0.766692 | 0.739370 | 0.586508 | 0.110459 | 0.102857 |
| test omission | 0.787697 | 0.695330 | 0.738637 | 0.585586 | 0.047789 | 0.054137 |
| test excess | 0.641741 | 0.784351 | 0.705915 | 0.545494 | 0.057690 | 0.047201 |
| test combined | 0.707870 | 0.736795 | 0.722043 | 0.564998 | 0.105478 | 0.101338 |

test와 validation의 target rate가 매우 가깝다. test에서는 omission F1이
`-0.066083`, excess F1이 `+0.033463`, macro F1이 `-0.016310`, combined
F1이 `-0.017327` 변했다. 따라서 combined만 보는 것보다 두 branch를
분리해 보는 것이 여전히 필요하다.

CSV의 TP/FP/FN 합계는 `metrics.json`과 세 클래스 모두 정확히 일치했다.
평가 전후 v1/v2 manifest와 v1/v2 best checkpoint 해시도 변하지 않았다.

test 100장 전체를 정량 결과와 같은 중앙 512 crop으로 확인하는 독립 검수
패키지는 다음 위치에 있다.

```text
/home/work/models/BCE-Net/map-ortho-test-audit-full100-spatial-v2-20260722
training_monitor/test-audit-full100-spatial-v2-20260722
```

`results_gallery.html`은 GT와 prediction/error를 함께 넘겨보는 전체 갤러리다.
편향을 줄여 정식 검수하려면 `README.md` 순서에 따라
`01_stage1_gt_only/`를 먼저 판독하고 `full100_review.csv`를 작성한다.
직접 CSV 값을 입력하는 대신 `review_form.html`의 dropdown과 CSV 다운로드를
사용할 수 있다. 입력 내용은 브라우저 local storage에도 자동 저장된다.
`quantitative/`의 원본 복사본은 frozen 평가 파일과 SHA-256이 일치한다.

## 정성 해석

명확하거나 강하게 의심되는 모델 오류:

- `omission_00714`: 큰 omission GT를 전혀 예측하지 못해 FN 32,324px가
  발생했다.
- `omission_00799`: omission을 놓치고 인접 roof 일부를 excess로 예측하는
  branch 혼동이 보인다.
- `excess_00218`, `excess_00245`: 큰 excess polygon의 일부만 검출해 recall이
  각각 약 0.17, 0.51이다.
- `omission_01119/01121/01122/01123`: 같은 공간의 큰 omission 객체를
  반복적으로 부분 검출한다.

라벨 범위와 모델 오류가 함께 의심되는 사례:

- `excess_00378`: excess는 잘 맞추지만 영상에 보이는 큰 roof 영역을
  omission으로도 검출한다. 누락 omission인지 branch FP인지 원본 검수가
  필요하다.
- `excess_00597`, `excess_00380`: 공사 상태와 footprint 범위가 얽혀 있어
  넓은 excess FP를 모델 오류로만 확정하기 어렵다.
- `excess_00567`: 온실·수목·그림자와 polygon 경계가 겹쳐 두 change head가
  모두 과검출한다.

상위 패널은 `qualitative/top_*_{fp,fn}.png`, 개별 이미지는
`qualitative/samples/`에 있다. 이는 1차 육안 판독이며 clean-label 확정이
아니다.

## Test 내부 공간 상관 한계

cross-split leakage는 0이지만 같은 split 안의 인접 patch는 남아 있다.

- test source-1024 overlap: 122 pairs
- test center-512 overlap: 48 pairs
- test spatial components: 23
- largest test component: 39 samples

따라서 test는 100개의 파일이지만 100개의 완전 독립 공간이라고 볼 수
없다. 가장 큰 39장 component를 제외한 진단값은 macro F1 `0.733550`,
combined F1 `0.734359`였다. 공식 점수보다 높으므로 해당 component가 공식
점수를 부풀린 정황은 없지만, 이 값은 표본과 target rate가 달라 공식 test
성능을 대체하지 않는다. Formula (7) 비교에는 같은 v2 split과 동일한
평가 단위를 사용해야 한다.

## 다음 단계

현재 결과를 control로 고정하고 Formula (7)만 교정한 paired experiment를
수행한다. 첫 비교에서는 RSG, scaling, polygon inference를 동시에 켜지
않는다. 그래야 Formula (7)의 효과를 분리할 수 있다.

1. `F`, `F_BG`, `F_FG` 명시적 반환
2. omission `D(F_BG,F)`, excess `D(F,F_FG)`
3. bounding-box 전체 patch cosine
4. omission/excess 독립 평균
5. 손계산 및 gradient 테스트
6. legacy/paper-faithful/robust profile 분리
