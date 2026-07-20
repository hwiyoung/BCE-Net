# BCE-Net 논문-구현 대응표

## 결론

현재 코드는 BCE-Net의 ResNet34, 3개 segmentation branch, reference 기반
feature 분할, DCN, contrastive loss 방향을 구현한다.

다만 완료된 학습은 논문의 완전 재현이 아니다.

- BCE+Dice 대신 GCE+weighted Dice 사용
- random sample generator 미구현
- contrastive Formula (7)의 feature 구성과 평균 방식이 다름
- polygon-level removed 판정 미구현
- 논문 `lr=0.01` 대신 WHU checkpoint fine-tuning용 `lr=0.001 + cosine`

따라서 모델의 정확한 명칭은 **BCE-Net 기반 noisy-label robust
fine-tuning**이다.

## 기준 자료

- [논문 원문](https://arxiv.org/pdf/2304.07076)
- [출판본](https://doi.org/10.1016/j.isprsjprs.2023.05.011)
- [저자 저장소](https://github.com/liaochengcsu/BCE-Net)
- 코드 기준 commit: `2385e17`
- 실행 기준: `/home/work/models/BCE-Net/map-ortho-robust-v2-20260720`

판정:

- **일치**: 논문 역할과 코드 계산이 직접 대응
- **변형**: 논문 의도는 유지하지만 계산이나 설정을 변경
- **미구현**: 대응 코드 없음
- **검증 필요**: 비슷하게 구현됐지만 논문 수식과 동일하지 않음

## 논문 구성과 코드 위치

| 논문 구성 | 논문 위치 | 현재 코드 위치 | 코드가 실제로 하는 일 | 판정 |
|---|---|---|---|---|
| 입력 `I`, reference `M` | §3.1.2 | [`__getitem__`](dataset/bcenet_map_ortho.py#L267-L322), [`run_epoch`](train_bcenet_map_ortho.py#L220-L238) | before RGB와 after footprint를 읽어 `model(image, reference)` 호출 | 변형 |
| 시간 순서 독립성 | §4.5 | [`derive_targets_and_weights`](dataset/bcenet_map_ortho.py#L131-L175) | 논문과 시간은 반대지만 reference 기준 집합 관계를 유지 | 일치 |
| pre-trained ResNet34 | §3.1.1 | [`Baseline34`](Testmodel/CDResWHU.py#L1215-L1240), [`build_model`](train_bcenet_map_ortho.py#L130-L145) | ResNet34 encoder 사용, 공개 WHU checkpoint strict loading | 일치 |
| multi-level feature fusion | §3.1.1, Fig. 4 | [`New_Fusion`](Testmodel/CDResWHU.py#L602-L640), [full decoder](Testmodel/CDResWHU.py#L1294-L1308) | 네 encoder level을 upsample/concatenate하여 복원 | 일치 |
| `F_BG=(1-M)×F` | Formula (1) | [`New_Fusion.forward`](Testmodel/CDResWHU.py#L622-L640) | reference를 convolution한 뒤 `backfeat=out*(1-lab)` 계산 | 변형 |
| new-building head `H_N` | §3.2.1, Formula (1)–(2) | [new output](Testmodel/CDResWHU.py#L1325-L1331), [target](dataset/bcenet_map_ortho.py#L171-L175), [loss 연결](utils/bcenet_loss.py#L236-L245) | 현재 데이터에서는 `new_out → omission/class 2`로 학습 | 일치 |
| `F_FG=M×(1-sigmoid(F))` | Formula (3) | [removed output](Testmodel/CDResWHU.py#L1318-L1319) | `(1-sigmoid(d4))*reference`를 removed convolution에 입력 | 일치 |
| removed-building head `H_R` | §3.2.2, Formula (3)–(4) | [removed output](Testmodel/CDResWHU.py#L1318-L1319), [target](dataset/bcenet_map_ortho.py#L171-L175), [loss 연결](utils/bcenet_loss.py#L226-L235) | 현재 데이터에서는 `mov_out → excess/class 3`로 학습 | 일치 |
| existing-building head `H_E` | §3.2.3, Formula (5) | [existing decoder](Testmodel/CDResWHU.py#L1294-L1316), [target](dataset/bcenet_map_ortho.py#L131-L175), [loss 연결](utils/bcenet_loss.py#L216-L225) | `class 1 or class 2`를 보조 building task로 학습 | 일치 |
| DCN feature transform | §3.3, Formula (6) | [DCN 선언](Testmodel/CDResWHU.py#L1269), [full feature DCN](Testmodel/CDResWHU.py#L1310-L1314), [split feature DCN](Testmodel/CDResWHU.py#L1325-L1327) | full feature와 new/background feature의 offset 학습 | 일치 |
| branch BCE+Dice | Formula (2), (4), (5) | [`weighted_pixel_loss`](utils/bcenet_loss.py#L24-L52), [`weighted_dice_loss`](utils/bcenet_loss.py#L55-L73), [`segmentation_loss`](utils/bcenet_loss.py#L76-L96) | 코드는 BCE를 지원하지만 완료 실행은 GCE(q=0.7)+weighted Dice | 변형 |
| contrastive projection | §3.4 | [projection output](Testmodel/CDResWHU.py#L1264-L1267), [feature 반환](Testmodel/CDResWHU.py#L1332-L1339) | 3×3 conv로 `feat_all`, `feat_mov` 두 개를 one-channel로 반환 | 검증 필요 |
| instance contrastive loss | Formula (7) | [`_instance_terms`](utils/bcenet_loss.py#L99-L134), [`instance_contrastive_loss`](utils/bcenet_loss.py#L137-L185) | omission은 `0.5(1-D)`, excess는 `0.5(1+D)`로 cosine 방향 적용 | 검증 필요 |
| 전체 loss | Formula (8) | [`BCENetCriterion`](utils/bcenet_loss.py#L202-L274) | existing + removed + new + contrastive 합산 | 일치 |
| random sample generator | §3.4, Fig. 8 | 대응 코드 없음 | positive weight와 crop 신뢰도 가중치로 불균형을 완화하지만 합성 label은 만들지 않음 | 미구현 |
| flip augmentation | Fig. 8, §3.5 | [augmentation](dataset/bcenet_map_ortho.py#L227-L245) | 좌우·상하 flip | 일치 |
| rotation augmentation | Fig. 8, §3.5 | [augmentation](dataset/bcenet_map_ortho.py#L246-L252) | 0/90/180/270도 회전 | 변형 |
| scaling augmentation | Fig. 8, §3.5 | 대응 코드 없음 | scaling 미적용 | 미구현 |
| color enhancement | Fig. 8, §3.5 | [augmentation](dataset/bcenet_map_ortho.py#L253-L258) | brightness/contrast 변경 | 일치 |
| SGD, 100 epochs | §3.5 | [CLI 기본값](train_bcenet_map_ortho.py#L54-L74), [optimizer](train_bcenet_map_ortho.py#L473-L483) | SGD와 100 epoch는 같고, LR/scheduler는 다름 | 변형 |
| precision/recall/F1/IoU | §3.5 | [`BinaryMetrics`](train_bcenet_map_ortho.py#L148-L180), [metric 연결](train_bcenet_map_ortho.py#L248-L264) | omission, excess, combined change별 pixel metric 계산 | 일치·확장 |
| best model 보존 | §3.5 | [selection](train_bcenet_map_ortho.py#L567-L608) | omission/excess macro F1 최고 checkpoint 저장 | 변형 |
| removed polygon 평균 `θ=0.5` | §3.2.2 | [pixel threshold](train_bcenet_map_ortho.py#L248-L264) | pixel threshold 0.5만 사용; polygon 평균과 regularization 없음 | 미구현 |
| qualitative 결과 | Fig. 11–14 | [`write_qualitative_results`](train_bcenet_map_ortho.py#L359-L409), [`qualitative_row`](utils/bcenet_visualization.py#L119-L193) | 고정 validation 샘플의 GT/prediction/probability/error 출력 | 일치·확장 |

## branch와 현재 라벨의 대응

| 논문 branch | 논문 의미 | 현재 target | 코드 |
|---|---|---|---|
| `H_E` | image에 존재하는 building | class 1 or 2 | [`target_existing`](dataset/bcenet_map_ortho.py#L173) |
| `H_N` | image에는 있고 reference에는 없음 | class 2 omission | [`target_new_head`](dataset/bcenet_map_ortho.py#L174) |
| `H_R` | reference에는 있고 image에는 없음 | class 3 excess | [`target_removed_head`](dataset/bcenet_map_ortho.py#L175) |

현재 reference는 after footprint이므로 논문의 branch 이름을 절대적인
신규/제거 연도로 해석하면 안 된다.

```text
reference = class 1 ∪ class 3
existing  = class 1 ∪ class 2
new_out   = class 2 omission
mov_out   = class 3 excess
```

## Formula (7)이 검증 필요한 이유

| 논문 Formula (7) | 현재 코드 |
|---|---|
| `F`, `F_BG`, `F_FG` 세 feature 역할 | `feat_all`, `feat_mov` 두 개만 반환 |
| new는 `(F_BG, F)` 비교 | omission과 excess 모두 같은 feature pair 사용 |
| removed는 `(F, F_FG)` 비교 | 별도 `F_FG` contrastive feature 없음 |
| instance bounding box 전체 patch | connected-component pixel만 선택 |
| new/removed를 각각 `1/(2N)`, `1/(2R)` 평균 | 두 클래스의 모든 instance term을 한 번에 평균 |
| 1×1 conv + normalization + sigmoid | 3×3 conv, 별도 normalization 없음, loss에서 sigmoid |

추가 코드 근거:

- [`New_Fusion`](Testmodel/CDResWHU.py#L638-L640)은
  `(mask 적용 후 backfeat, mask 적용 전 outf)`를 반환한다.
- [`Baseline34`](Testmodel/CDResWHU.py#L1292)는 두 번째 `outf`를
  `featn`으로 받는다.
- [`feat_mov`](Testmodel/CDResWHU.py#L1334-L1335)은 이 mask 적용 전
  `featn`에서 생성된다.
- criterion은
  [`feature_all`, `feature_split`](utils/bcenet_loss.py#L246-L255) 한 쌍을
  두 변화 클래스에 재사용한다.

따라서 contrastive term의 **가깝게/멀게 하는 방향은 맞지만 논문
Formula (7)의 feature 구성과 동일하지 않다.**

## 완료 실행 설정

| 항목 | 논문 | 완료 실행 |
|---|---:|---:|
| 초기화 | pre-trained ResNet34 | 저자 WHU best checkpoint |
| pixel loss | BCE + Dice | GCE(q=0.7) + weighted Dice |
| optimizer | SGD | SGD |
| learning rate | 0.01 | 0.001 |
| scheduler | 구체적으로 미기재 | cosine annealing |
| epochs | 최대 100 | 100 |
| augmentation | flip/rotation/scale/color | flip/90도 rotation/color |
| best 기준 | best evaluation result | macro omission/excess F1 |

실제 config:
[`training_monitor/current/config.json`](training_monitor/current/config.json)

## 현재 결과가 입증하는 범위

입증함:

- 저자 BCE-Net architecture와 WHU checkpoint가 사용자 데이터에서 학습됨
- existing/omission/excess branch가 모두 non-zero prediction을 유지함
- DCNv2 포함 forward/backward가 100 epoch 완료됨
- best validation macro F1은 epoch 24의 `0.729859`

입증하지 못함:

- Formula (7)의 정확한 재현
- random sample generator의 효과
- DCN/contrastive 각각의 독립 효과
- untouched test 100장의 일반화 성능
- 논문 SI-BU/WHU 수치의 재현

## 다음 검증 순서

1. `checkpoint-best.pth`로 test 100장 독립 평가
2. `F`, `F_BG`, `F_FG`를 명시적으로 반환하도록 contrastive output 수정
3. Formula (7)의 클래스별 평균과 bounding-box patch 계산 구현
4. random sample generator와 scaling augmentation 구현
5. polygon 평균 probability와 regularization 구현
6. baseline/RSG/contrastive/DCN/robust-loss ablation 수행
