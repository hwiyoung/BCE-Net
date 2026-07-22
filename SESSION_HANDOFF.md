# BCE-Net map-ortho 작업 인계서

## 1. 새 세션에서 먼저 읽을 파일

다음 순서로 읽는다.

1. `SESSION_HANDOFF.md`
2. `PAPER_IMPLEMENTATION_ALIGNMENT.md`
3. `TRAINING_DESIGN.md`
4. `README.md`
5. `dataset/map_ortho_audit.json`
6. `training_monitor/current/config.json`
7. `training_monitor/current/metrics.jsonl`

저장소 실제 경로:

```text
/home/work/projects/BCE-Net
```

일부 workspace context에는 `/home/work/project/BCE-Net`처럼 `projects`의
`s`가 빠진 경로가 표시될 수 있다. 실제 명령은 위의 `projects` 경로에서
실행한다.

## 2. 현재 목표

before 정사영상과 after footprint를 입력으로 BCE-Net을 학습해 다음을
분리한다.

- omission/class 2: before 영상에는 건물이 있지만 after footprint에는 없음
- excess/class 3: after footprint에는 있지만 before 영상에는 건물이 없음

v1 학습과 frozen test 뒤 split 사이 pixel-level spatial overlap을 발견했다.
기존 산출물을 보존한 채 256px buffered spatial split v2를 별도 생성했고,
cross-split source/crop overlap 0을 독립 검증했다. 현재 robust baseline도
WHU checkpoint에서 v2로 100 epoch 재학습했으며, best epoch 46의 frozen
test macro F1은 `0.722276`이다. 다음 목표는 이 v2 control을 고정하고
Formula (7)만 교정한 paired experiment를 준비하는 것이다.

## 3. 데이터

데이터 경로:

```text
/home/work/data/change_detection/building/map-ortho
```

파일 형식과 크기:

- PNG
- 1024×1024
- 영상 1,000장
- 상태/footprint 마스크 1,005장
- 대응 영상이 없는 마스크 5장은 audit 후 제외

라벨:

| 값 | 의미 | before 영상 | after footprint |
|---:|---|---:|---:|
| 0 | background | 건물 아님 | 건물 아님 |
| 1 | no change | 건물 | 건물 |
| 2 | omission | 건물 | 없음 |
| 3 | excess | 건물 아님 | 건물 |

manifest:

```text
dataset/map_ortho_manifest.csv             # v1, 보존
dataset/map_ortho_manifest_spatial_v2.csv  # 256px buffered split
```

spatial split:

| split | 수량 | 용도 |
|---|---:|---|
| train | 800 | parameter 학습 |
| validation | 100 | best checkpoint 선택과 모니터링 |
| test | 100 | 중앙 512×512 frozen 평가 완료, spatial overlap 제한 있음 |

인접 patch가 서로 다른 split에 섞이는 것을 줄이기 위해
`spatial_group` 단위로 분리했다.

## 4. branch mapping

```text
reference input       = class 1 or class 3
existing target       = class 1 or class 2
new_out target        = class 2 omission
removed/mov_out target= class 3 excess
```

저자 코드의 `new`와 `removed`는 절대 연도가 아니라 reference mask와
image의 집합 관계를 뜻한다. 현재 데이터에서 `new_out`을 시간적 신규
건물이라고 해석하면 안 된다.

## 5. 구현 파일

| 파일 | 역할 |
|---|---|
| `train_bcenet_map_ortho.py` | train/validation loop, metrics, checkpoints, qualitative outputs |
| `dataset/bcenet_map_ortho.py` | manifest loader, crop/augmentation, target/weight 생성 |
| `utils/bcenet_loss.py` | GCE/BCE, Dice, instance contrastive loss |
| `utils/bcenet_visualization.py` | 정성 패널과 training curve |
| `scripts/prepare_map_ortho_manifest.py` | 데이터 audit와 spatial split 생성 |
| `scripts/show_bcenet_training_status.py` | 실행 상태 요약 |
| `Testmodel/CDResWHU.py` | 학습에 사용한 저자 BCE-Net architecture |
| `TRAINING_DESIGN.md` | 사용자 데이터 학습 설계 |
| `PAPER_IMPLEMENTATION_ALIGNMENT.md` | 논문과 현재 구현의 대응 및 차이 |

## 6. 완료된 학습

실행 디렉터리:

```text
/home/work/models/BCE-Net/map-ortho-robust-v2-20260720
```

workspace link:

```text
training_monitor/current
```

초기 checkpoint:

```text
/home/work/models/BCE-Net/checkpoint-best-whu.pth
```

설정:

```text
epochs=100
batch_size=4
crop_size=512
train_jitter=128
optimizer=SGD
lr=0.001
momentum=0.9
weight_decay=0.0001
scheduler=cosine
pixel_loss=GCE(q=0.7)
dice_weight=1
positive_weight omission/excess=4
boundary_width=2
boundary_weight=0.25
secondary_change_weight=0.5
contrastive_weight=1
threshold=0.5
seed=1024
AMP=true, DCNv2 only FP32
best_metric=macro omission/excess F1
```

학습 상태:

```text
100/100 epochs 완료
background training process 종료
```

best checkpoint:

```text
/home/work/models/BCE-Net/map-ortho-robust-v2-20260720/checkpoint-best.pth
epoch=24
validation macro F1=0.7298585383
validation combined change F1=0.7365607778
validation omission F1=0.7586989608
validation excess F1=0.7010181157
validation loss=1.9735034990
```

last checkpoint:

```text
/home/work/models/BCE-Net/map-ortho-robust-v2-20260720/checkpoint-last.pth
epoch=100
validation macro F1=0.6967006537
validation combined change F1=0.7154140013
validation omission F1=0.7754808236
validation excess F1=0.6179204839
validation loss=2.0251157188
```

추론에는 `checkpoint-best.pth`를 우선 사용한다.

### 6.1 Spatial split v2 robust baseline

v2 split과 감사:

```text
manifest=dataset/map_ortho_manifest_spatial_v2.csv
report=dataset/map_ortho_spatial_split_v2_report.md
audit=training_monitor/spatial-v2-audit-20260722
buffer=256px (30.72m)
cross-split source/crop overlap=0
```

재학습과 test:

```text
training=/home/work/models/BCE-Net/map-ortho-robust-spatial-v2-20260722
monitor=training_monitor/robust-spatial-v2-20260722
best epoch=46
validation macro F1=0.7385861947

test=/home/work/models/BCE-Net/map-ortho-robust-spatial-v2-test-center512-best-e46-20260722
test monitor=training_monitor/test-spatial-v2-center512-best-e46-20260722
test omission F1=0.7386369463
test excess F1=0.7059153413
test macro F1=0.7222761438
test combined F1=0.7220427637
```

v2 test는 cross-split leakage가 없지만 100장이 23개 spatial component에
속하고 test 내부 center-crop overlap이 48쌍 있다. 표준 pixel metric은
공식 결과로 유지하되 100개의 완전 독립 공간으로 표현하지 않는다. 전체
해석은 `SPATIAL_V2_BASELINE_REPORT.md`에 있다. Formula (7) 코드는 이
재학습에서 수정하지 않았다.

v2 test 100장 전체 검수 패키지:

```text
output=/home/work/models/BCE-Net/map-ortho-test-audit-full100-spatial-v2-20260722
monitor=training_monitor/test-audit-full100-spatial-v2-20260722
gallery=training_monitor/test-audit-full100-spatial-v2-20260722/results_gallery.html
form=training_monitor/test-audit-full100-spatial-v2-20260722/review_form.html
review_form=training_monitor/test-audit-full100-spatial-v2-20260722/full100_review.csv
```

`V001..V100`은 실제 sample ID를 가린 presentation ID다. stage 1 GT-only,
stage 2 prediction/error, sample별 lossless source crop 4종, frozen 전체 및
sample별 정량 결과를 포함한다. 100개 sample ID 집합은 spatial v2 test와
정확히 일치하며 정량 복사본의 SHA-256은 frozen 평가 원본과 같다.
`review_form.html`은 고정 enum dropdown, 자동 저장, CSV import/export를
제공한다.

## 7. 학습 결과 해석

- branch collapse는 발생하지 않았다.
- train loss는 감소했고 omission/excess 모두 non-zero prediction을 유지했다.
- validation 최고점은 epoch 24였고 이후에는 정체와 약한 과적합이 나타났다.
- validation 라벨에는 육안상 오류가 의심되는 샘플이 있다.
- `omission_00641`은 주차장처럼 보이는 영역 전체가 omission GT로
  지정되어 모델이 예측하지 않을 때 큰 FN으로 계산된다.
- validation 수치는 모델 학습 가능성을 보여주지만 clean-label 성능을
  보장하지 않는다.

frozen test 중앙 crop 결과:

```text
output=/home/work/models/BCE-Net/map-ortho-robust-v2-test-center512-best-e24-20260721
monitor=training_monitor/test-center512-best-e24-20260721
omission F1=0.6386873930
excess F1=0.8059870679
macro F1=0.7223372305
combined F1=0.7547481603
```

이 수치는 1024 sliding-window/full-tile 결과가 아니며, 상세 해석과
Formula (7) 계획은 `TEST_EVALUATION_REPORT.md`에 기록했다.

test label pilot audit 패키지:

```text
output=/home/work/models/BCE-Net/map-ortho-test-pilot-audit-v1-20260721
monitor=training_monitor/test-pilot-audit-v1-20260721
review_form=training_monitor/test-pilot-audit-v1-20260721/pilot_review.csv
```

30장은 오류 상위 고유 16장, high-agreement control 4장, seeded random
control 10장으로 구성했다. `01_stage1_gt_only` 검수를 모두 마친 다음에만
`02_stage2_predictions`를 열어야 한다. 실제 ID와 선정 이유는
`03_unblind/selection_manifest.csv`에 분리했다.

나머지 test 70장 검수 패키지:

```text
output=/home/work/models/BCE-Net/map-ortho-test-audit-remaining70-v1-20260722
monitor=training_monitor/test-audit-remaining70-v1-20260722
review_form=training_monitor/test-audit-remaining70-v1-20260722/remaining70_review.csv
```

pilot 30장과 remaining 70장은 실제 sample ID 기준 교집합이 없고 합집합이
test 100장과 일치한다.

canonical test 100장 통합 검수 패키지:

```text
output=/home/work/models/BCE-Net/map-ortho-test-audit-full100-v1-20260722
monitor=training_monitor/test-audit-full100-v1-20260722
review_form=training_monitor/test-audit-full100-v1-20260722/full100_review.csv
```

앞으로는 위 `full100_review.csv` 하나만 작성한다. 기존 P/R audit ID를
유지했고, stage/source 파일은 외부 링크가 없는 독립 복사본이다.

split overlap 감사:

```text
output=/home/work/models/BCE-Net/map-ortho-split-overlap-audit-20260722
monitor=training_monitor/test-split-overlap-audit-20260722
```

- ID/path 중복 0, shared spatial group 0
- 실제 source pixel overlap: train–val 36 pairs, train–test 45 pairs,
  val–test 7 pairs
- 총 88 pairs에서 image/map/label overlap pixel이 모두 100% 동일
- validation `excess_00323`과 test `excess_00324`의 중앙 crop 78.16% 중복
- source-overlap 24장 combined F1=0.786803, non-overlap 76장=0.739550
- possible eval-overlap 8장 combined F1=0.840666, 나머지 92장=0.742370

subset 차이는 target rate와 공간/클래스 구성 차이도 포함하므로 leakage의
인과 효과로 단정하지 않는다. 상세 수치는
`training_monitor/test-split-overlap-audit-20260722/SUBSET_METRICS.md`에 있다.

v1 manifest는 수정하지 않았다. source patch footprint에 256px buffer를
적용한 spatial split v2를 별도 manifest로 만들고 현재 robust baseline을
다시 학습했다. 이후 variant도 같은 v2를 사용해야 한다.

정성 결과:

```text
training_monitor/current/qualitative/epoch-0100.png
```

패널 색상:

- GT/prediction: green=unchanged, orange=omission, magenta=excess
- error: green=TP, red=FP, blue=FN, yellow=wrong change class

## 8. 논문 충실도 결론

현재 모델의 올바른 명칭:

```text
BCE-Net-based robust domain adaptation/fine-tuning
```

현재 상태에서 사용할 수 없는 명칭:

```text
exact BCE-Net paper reproduction
```

핵심 이유:

- architecture, 3-head mapping, DCN, contrastive 방향은 논문 의도에 맞음
- 완료 실행은 논문 BCE+Dice 대신 GCE+Dice와 신뢰도 가중치를 사용
- random sample generator 미구현
- contrastive Formula (7)의 feature pair/평균/bounding-box 계산과 차이
- polygon-level removed 판정과 regularization 미구현
- paper-faithful ablation 미실행

세부 근거는 `PAPER_IMPLEMENTATION_ALIGNMENT.md`를 따른다.

## 9. Git 상태

현재 기준 핵심 커밋:

```text
2385e17 feat: add robust map-ortho BCE-Net training
efa0678 docs: record full AOI inference approval
```

모델 checkpoint와 training PNG는 저장소 밖
`/home/work/models/BCE-Net`에 보존되며 Git에 포함하지 않는다.

새 세션 시작 시 반드시 실행:

```bash
cd /home/work/projects/BCE-Net
git status --short
git log -5 --oneline --decorate
readlink -f training_monitor/current
```

사용자 변경이 있으면 보존하고, unrelated 파일을 한 커밋에 섞지 않는다.

## 10. 다음 작업 순서

### 1단계: test 100장 frozen 평가 (완료, spatial overlap 제한)

- best checkpoint 고정
- threshold는 우선 validation에서 사용한 0.5 고정
- test를 model selection에 사용하지 않음
- omission/excess/combined의 precision, recall, F1, IoU 기록
- 클래스별 prediction rate와 target rate 기록
- test 정성 샘플 저장

### 2단계: label audit

- test의 FP/FN이 큰 sample을 순위화
- before image, after footprint, GT, probability, prediction, error panel 생성
- 30장 blinded two-stage pilot 패키지 생성 완료, 사람 검수 대기
- 나머지 70장 blinded two-stage 패키지 생성 완료, 사람 검수 대기
- 사람이 확인한 clean subset을 별도 CSV로 저장
- 원본 라벨을 자동 덮어쓰지 않음

### 3단계: 1024 전체 추론

- 모델은 512 crop으로 학습됐으므로 1024 영상을 overlap sliding-window로 처리
- 경계에서는 probability blending 사용
- omission/excess probability를 각각 보존
- 0/1/2/3 상태 마스크 생성
- split 정보를 결과 metadata에 기록

### 4단계: 전체 1,000장 운영용 결과

- train/validation/test 모두 추론할 수 있음
- train 800장 결과는 일반화 성능 근거로 사용하지 않음
- 최종 성능 보고는 buffered split v2의 untouched test와 clean audit
  기준을 사용

### 5단계: 논문 충실도 실험 (다음 작업)

- exact BCE+Dice profile
- exact contrastive Formula (7)
- random sample generator
- polygon-level inference
- DCN/contrastive/RSG ablation

## 11. 안전 조건

- `checkpoint-best.pth`, `checkpoint-last.pth`, `metrics.jsonl`을 삭제하거나
  덮어쓰지 않는다.
- 새 평가와 추론 결과는 새로운 output directory에 저장한다.
- test 결과를 본 뒤 hyperparameter를 수정하면 test가 더 이상 완전한
  독립 평가가 아니므로 변경 내역을 기록한다.
- 라벨 오류 의심과 모델 오류를 구분한다.
- 전체 1,000장 metric을 일반화 성능으로 표현하지 않는다.
- 논문 SI-BU/WHU 수치와 현재 validation 수치를 직접 비교하지 않는다.
- 현재 v1 test를 strict spatial-independent 성능으로 표현하지 않는다.

## 12. 새 세션 시작 프롬프트

```text
/home/work/projects/BCE-Net 작업을 이어서 진행해줘.

먼저 SESSION_HANDOFF.md, PAPER_IMPLEMENTATION_ALIGNMENT.md,
TRAINING_DESIGN.md를 모두 읽고 git 상태와 training_monitor/current를
확인해.

v1 split overlap 감사 뒤 256px buffered spatial split v2와 현재 robust
baseline 재학습/frozen test까지 완료됐다. `SPATIAL_V2_BASELINE_REPORT.md`
를 읽고 v2 control을 기준으로 Formula (7) 수정 실험을 준비해.

기존 checkpoint와 학습 산출물은 수정하거나 삭제하지 마. 구현 전에는
기존 checkpoint/평가 산출물을 수정하지 말고 Formula (7) 외 RSG/scaling/
polygon inference는 첫 paired experiment에 동시에 켜지 마.
```
