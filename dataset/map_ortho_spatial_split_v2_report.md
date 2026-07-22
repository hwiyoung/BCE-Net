# Map-ortho spatial split v2

## 결론

기존 v1 manifest는 수정하지 않았다. 동일한 source mosaic의 1024 patch가
겹치거나 axis-aligned gap이 `256`픽셀 미만이면 같은
spatial component로 묶고, component 전체를 하나의 split에 배정했다.

- source 1024 cross-split overlap: **0 pairs**
- cross-split buffer violations: **0 pairs**
- train possible 512 crop union과 held-out center crop overlap: **0 pairs**
- validation/test center crop overlap: **0 pairs**

## Buffer 후보

| buffer px | components | largest samples | largest fraction |
|---:|---:|---:|---:|
| 0 | 313 | 66 | 6.6% |
| 128 | 267 | 73 | 7.3% |
| 256 | 215 | 73 | 7.3% |
| 512 | 140 | 276 | 27.6% |
| 1024 | 78 | 544 | 54.4% |

`256px`는 GSD 0.12m 기준 30.72m다. `512px`부터 component가 276장으로
급증하므로, split 구성 가능성과 주변 문맥 분리를 함께 고려해 256px를
선택했다.

## Split 분포

| split | samples | components | center omission/excess | omission rate | excess rate | combined rate |
|---|---:|---:|---:|---:|---:|---:|
| train | 800 | 146 | 372/428 | 0.054595 | 0.046848 | 0.101443 |
| val | 100 | 46 | 46/54 | 0.055910 | 0.046947 | 0.102857 |
| test | 100 | 23 | 47/53 | 0.054137 | 0.047201 | 0.101338 |

## 독립 검증

| split pair | source overlap | buffer violation | center overlap | train-union vs held-center |
|---|---:|---:|---:|---:|
| train_val | 0 | 0 | 0 | 0 |
| train_test | 0 | 0 | 0 | 0 |
| val_test | 0 | 0 | 0 | 0 |

`xoff/yoff`는 단일 merged source raster의 공통 원점에 대한 픽셀 offset이다.
원본 COG는 현재 작업공간에 없지만, 기존 audit에서 좌표로 정렬한 88개
교차 영역의 image/map/label 픽셀이 모두 100% 일치했다.
