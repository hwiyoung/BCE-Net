# BCE-Net 사용자 데이터 학습 설계서

## 1. 목적과 현재 상태

이 문서는 `/home/work/data/change_detection/building/map-ortho` 데이터로
BCE-Net을 학습하기 위해 추가한 데이터 처리, 손실함수, 평가, 오류 완화,
체크포인트 및 시각화 설계를 설명한다.

논문의 각 수식·모듈과 현재 구현의 일치 여부 및 재현 한계는
`PAPER_IMPLEMENTATION_ALIGNMENT.md`에서 별도로 추적한다.

첫 번째 진단 학습은 모니터링 설계의 문제를 확인하기 위해 중단했다.
epoch 1의 omission F1은 0.715였지만 epoch 10부터 omission 출력이 모든
픽셀에서 0으로 붕괴했다. 전체 loss와 excess F1만 보면 이 문제를 놓칠 수
있었기 때문에, 재학습 버전에는 다음 안전장치를 추가했다.

- omission/excess 각각의 precision, recall, F1, prediction rate 기록
- 두 변경 클래스 F1의 평균인 macro change F1로 best checkpoint 선택
- 변경 클래스 양성 픽셀 가중치
- 미세조정 learning rate 하향
- 변경 head의 prediction rate가 3 epoch 연속 0이면 자동 중단
- 고정 validation 샘플의 초기 및 주기적 정성 결과 저장
- epoch별 loss/F1/precision/recall 곡선 저장

## 2. 데이터의 시간 및 클래스 의미

실제 입력 구성은 다음과 같다.

- 영상 `I`: before 시점 정사영상
- footprint `M`: after 시점 건물 footprint
- 상태 라벨 `Y`: 영상과 footprint의 관계를 나타내는 픽셀 라벨

상태 라벨의 실제 값은 데이터셋의 `class_legend.csv`와 전수 검사를
기준으로 해석한다.

| 값 | 데이터셋 의미 | before 영상 | after footprint |
|---:|---|---:|---:|
| 0 | background | 건물 아님 | 건물 아님 |
| 1 | no change | 건물 | 건물 |
| 2 | omission | 건물 | footprint 없음 |
| 3 | excess | 건물 없음 | footprint 있음 |

여기서 BCE-Net 코드의 `new`/`removed` branch 이름은 절대 연도가 아니라
입력 footprint를 기준으로 붙은 이름이다. 따라서 다음과 같이 연결한다.

- BCE-Net reference 입력: `M = (Y == 1) or (Y == 3)`
- existing-building 보조 정답: `E = (Y == 1) or (Y == 2)`
- `new_out` 정답: `Y == 2` (omission)
- `mov_out` 정답: `Y == 3` (excess)

즉, branch 이름만 보고 class 2와 class 3을 시간적 신규/제거로 다시
뒤집지 않는다.

## 3. 데이터 검증과 분할

원본에는 영상 1,000장과 상태/footprint 마스크 1,005장이 있다. 대응하는
영상이 없는 마스크 5장은 audit에 기록하고 제외했다. 사용 가능한
1,000장은 공간 블록 단위로 분할하여 인접 패치가 서로 다른 split에
섞이는 것을 줄였다.

| split | 샘플 수 |
|---|---:|
| train | 800 |
| validation | 100 |
| test | 100 |

검증된 구조 조건은 다음과 같다.

- 모든 사용 샘플의 크기: 1024×1024
- 모든 상태 라벨 값: `{0, 1, 2, 3}`의 부분집합
- footprint 마스크 값: `{0, 255}`
- 모든 픽셀에서 `footprint == (class 1 or class 3)`

생성 파일:

- `dataset/map_ortho_manifest.csv`
- `dataset/map_ortho_audit.json`

## 4. 학습 입력과 증강

모델에는 1024×1024 원본에서 512×512 crop을 입력한다. 각 원본 패치는
변경 후보가 중심에 오도록 생성되었으므로 validation은 중앙 crop을
사용한다. train은 중심을 최대 128픽셀 jitter하여 후보를 유지하면서
문맥을 다양화한다.

train 증강:

- 좌우 및 상하 반전
- 0/90/180/270도 회전
- 밝기와 대비의 작은 변화

영상, footprint, 상태 라벨에는 동일한 기하 변환을 적용한다. 영상은
OpenCV BGR에서 RGB로 바꾸고 `[0, 1]` 범위로 변환한다. 기존 공개
checkpoint의 입력 방식을 유지하기 위해 ImageNet mean/std 정규화는
기본적으로 사용하지 않는다.

## 5. 모델과 출력

사용 모델은 `Testmodel/CDResWHU.py`의 `Baseline34`이다.

- ResNet-34 encoder
- existing-building segmentation branch
- omission과 excess를 분리하는 multi-task branch
- 두 위치에서 사용하는 DCNv2 정합 모듈
- instance-constrained contrastive feature 출력

forward 출력 순서:

```text
existing_logit, removed_logit, new_logit, feature_all, feature_split
```

구형 DCNv2 확장은 FP16 kernel을 지원하지 않기 때문에 DCNv2 연산만
FP32로 수행한다. 나머지 모델은 AMP FP16을 사용한다.

## 6. 라벨 오류 완화

라벨 오류가 많은 상황에서 모든 픽셀을 같은 신뢰도로 학습시키지 않는다.

### 6.1 구조적으로 불가능한 픽셀 제외

다음 픽셀의 loss weight는 0으로 둔다.

- omission 픽셀이 footprint 내부에 있는 경우
- excess 픽셀이 footprint 외부에 있는 경우
- footprint와 `(class 1 or class 3)`가 일치하지 않는 경우

현재 전수 검사에서는 이 구조 오류가 발견되지 않았지만, 이후 데이터
추가 시 자동 방어 역할을 한다.

### 6.2 경계 가중치

footprint와 각 정답 마스크의 2픽셀 경계는 정합 및 rasterization 오류가
집중될 가능성이 높다. 해당 픽셀의 loss weight는 `0.25`로 낮춘다.

### 6.3 중심 후보와 부수 후보

패치는 한 변경 객체를 중심으로 추출되었다. 중심 객체는 높은 신뢰도로
사용하고, 같은 crop에 우연히 포함된 다른 변경 객체는 loss weight를
`0.5`로 낮춘다. 부수 객체를 완전히 버리지는 않는다.

### 6.4 robust pixel loss와 클래스 불균형

각 segmentation head에는 Generalized Cross Entropy(GCE)와 Dice loss를
함께 사용한다. GCE의 `q=0.7`은 큰 손실을 내는 의심 라벨의 영향을
일반 BCE보다 완만하게 만든다.

중앙 crop 기준 양성 비율은 omission 약 5.5%, excess 약 4.4%이다.
첫 진단 실행에서는 이 불균형 때문에 omission head가 배경만 출력하는
방향으로 붕괴했다. 이를 막기 위해 omission/excess 양성 픽셀에는
`4.0`의 추가 가중치를 사용한다. 이는 단순 역빈도 17~22배보다 보수적인
제곱근 수준의 값으로, 라벨 오류까지 과도하게 증폭하지 않기 위한
절충이다.

전체 loss는 다음 구조다.

```text
L =
  L_existing(GCE + Dice)
  + L_omission(GCE_positive_weighted + Dice)
  + L_excess(GCE_positive_weighted + Dice)
  + L_instance_contrastive
```

contrastive term은 omission instance의 두 feature를 가깝게 하고 excess
instance의 두 feature를 멀어지게 한다. 너무 작은 객체와 신뢰 가중치가
낮은 객체는 contrastive 계산에서 제외한다.

## 7. 최적화 설정

오류 완화 학습의 기본 설정:

| 항목 | 값 |
|---|---:|
| 초기 가중치 | 공개 WHU BCE-Net best checkpoint |
| optimizer | SGD |
| learning rate | 0.001 |
| momentum | 0.9 |
| weight decay | 0.0001 |
| scheduler | cosine annealing |
| batch size | 4 |
| epochs | 100 |
| AMP | 사용, DCNv2만 FP32 |
| random seed | 1024 |

`0.001`은 이미 학습된 WHU checkpoint를 사용자 데이터에 미세조정하기
위한 값이다. 첫 진단 실행의 `0.01`은 초기에 좋은 omission feature를
빠르게 잃어버리는 불안정성을 보였다.

## 8. 정량 지표와 정상 학습 판독법

픽셀 정확도는 배경 비율이 매우 높아 사용하지 않는다. validation에서
다음 지표를 omission, excess, combined change별로 계산한다.

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
IoU       = TP / (TP + FP + FN)
```

추가 핵심 지표:

```text
macro change F1 = (omission F1 + excess F1) / 2
prediction rate = 모델이 양성으로 예측한 픽셀 / 전체 픽셀
target rate     = 정답 양성 픽셀 / 전체 픽셀
```

best checkpoint는 combined change F1이 아니라 macro change F1로
선택한다. combined 지표는 한 클래스만 잘 맞혀도 높게 나와 다른 head의
붕괴를 숨길 수 있기 때문이다.

### 정상 진행 신호

- 첫 5~10 epoch 동안 train loss가 전반적으로 감소한다.
- validation loss가 심하게 발산하지 않는다.
- omission과 excess F1이 모두 0보다 충분히 크고 함께 유지 또는 상승한다.
- macro change F1의 best 값이 초반 이후에도 갱신된다.
- 각 클래스에서 precision과 recall 중 하나만 0에 가깝게 고정되지 않는다.
- prediction rate가 0으로 수렴하지 않는다.
- 정성 결과에서 변경 객체 내부가 연결된 영역으로 검출되고 경계가
  건물 형태를 따른다.

### 경고 신호

- train loss는 감소하지만 validation loss가 계속 증가: 과적합 가능성
- precision만 높고 recall이 급락: 거의 아무것도 예측하지 않는 상태
- recall만 높고 precision이 급락: 넓은 영역을 변경으로 과검출
- omission 또는 excess prediction rate가 0: 해당 branch 붕괴
- combined F1은 유지되지만 macro F1이 급락: 한 클래스에만 편향
- probability panel이 전체적으로 단색: 출력 포화 또는 붕괴
- 도로, 그림자, 건물 경계 전체에 점 형태 FP가 반복: 라벨/학습률/경계
  가중치 재검토 필요

절대적인 합격 F1은 라벨 오류 수준과 validation 정제 여부에 따라 달라서
사전에 고정하지 않는다. 다만 첫 진단 checkpoint의 기준값은 omission
F1 0.715, excess F1 0.532, macro F1 약 0.624였다. 수정 학습은 적어도
두 class를 동시에 유지하면서 이 macro 기준을 개선하는지를 먼저 본다.

validation 라벨에도 오류가 있으므로 최종 모델 선택 전에는 소수의
validation 샘플을 사람이 검수한 clean subset으로 별도 평가하는 것이
필요하다.

## 9. 정성 결과

학습 시작 직전, 첫 epoch, 이후 5 epoch마다, 마지막 epoch에 고정된
validation 샘플을 출력한다. 샘플은 omission 중심 2장과 excess 중심
2장을 동일하게 유지하므로 epoch 간 변화를 직접 비교할 수 있다.

각 행의 패널:

1. before 정사영상
2. after footprint 입력(cyan)
3. 정답 상태: unchanged=green, omission=orange, excess=magenta
4. threshold 0.5의 예측 상태
5. omission 확률 heatmap
6. excess 확률 heatmap
7. 오류: TP=green, FP=red, FN=blue, class 혼동=yellow

출력 위치:

```text
<output-dir>/qualitative/initial-epoch-0000.png
<output-dir>/qualitative/epoch-0001.png
<output-dir>/qualitative/epoch-0005.png
<output-dir>/qualitative/epoch-0010.png
...
<output-dir>/training-curves.png
```

정성 결과에서는 단순히 색이 많아지는지를 보지 않는다. omission과
excess의 방향이 맞는지, 객체 단위로 연결되어 있는지, 건물 경계를
따르는지, 라벨 오류로 의심되는 객체를 모델이 일관되게 다르게 보는지를
확인한다.

## 10. 체크포인트와 재현성

매 epoch 다음 파일을 갱신한다.

- `checkpoint-last.pth`: 가장 최근 완료 epoch
- `checkpoint-best.pth`: validation macro change F1 최고 epoch
- `metrics.jsonl`: epoch별 전체 loss component와 정량 지표
- `config.json`: 실행 인자
- `training-curves.png`: 지표 곡선
- `qualitative/*.png`: 고정 샘플 정성 결과
- `train.log`: stdout/stderr
- `train.pid`: 분리 실행 프로세스 ID

중단 시 `checkpoint-last.pth`의 모델, optimizer, scheduler, AMP scaler를
복원하여 이어갈 수 있다. 실험 설정 자체가 바뀐 경우에는 optimizer까지
복원하는 resume를 사용하지 않고, 기존 best checkpoint를 새로운
`--init-checkpoint`로 사용해 새 실험으로 시작한다.

## 11. 실행 전 확인 순서

1. manifest/audit와 클래스 의미 확인
2. 데이터 및 loss 단위 smoke test
3. 실제 DCNv2 forward/backward 1배치 확인
4. batch size 4 메모리 확인
5. 초기 정성 결과 생성
6. 1 epoch 지표와 이미지 확인
7. 이상이 없을 때 분리된 전체 학습 실행

첫 진단 실행의 결과는 별도 디렉터리에 보존하고, 수정 학습은 새로운
출력 디렉터리를 사용한다.

## 12. 실험 프로파일

오류 라벨에 대응하는 실용 모델과 논문 구성에 가까운 비교 실험을
분리한다.

### 오류 완화 모델

```text
pixel loss=GCE, positive weight=4,
boundary weight=0.25, secondary change weight=0.5,
best metric=macro change F1
```

### 논문 구성 비교용 모델

```text
pixel loss=BCE, positive weight=1,
boundary weight=1, secondary change weight=1,
contrastive loss 사용
```

두 실행은 같은 spatial split과 seed를 사용해야 한다. 두 번째 프로파일은
공개 코드에 완전한 원 학습 설정이 없기 때문에 “논문과 동일한 재현”이
아니라, 이 저장소에서 확인 가능한 BCE-Net 구성 요소를 유지한 비교
실험으로 기록한다.
