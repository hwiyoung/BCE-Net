# BCE-Net Korea PoC Dashboard

Last updated: 2026-07-15

## Current Mode

Managed Container Mode — environment restored and reusable

- The current shell is inside a managed GPU container; nested Docker is not used.
- Repository-local environment: `/home/work/BCE-Net/.venv-bcenet-geo`
- One-command setup/repair: `make setup`
- One-command verification: `make verify`
- Session-independent runner: `./scripts/run_in_env.sh <command>`
- VS Code interpreter: `.venv-bcenet-geo/bin/python` selected by workspace settings.
- A new terminal does not require reinstalling packages while the workspace persists.
- A fresh cloud container runs `make setup`; a compatible bind-mounted venv can be reused.

| Item | Value |
| --- | --- |
| Repo root | `/home/work/BCE-Net` |
| GPU | NVIDIA H200, compute capability 9.0 |
| Python | 3.12.3 |
| PyTorch | `2.10.0a0+b558c986e8.nv25.11` |
| CUDA / nvcc | 13.0 |
| DCNv2 | Linux extension rebuilt for `sm_90`; CUDA forward passed |
| Geo stack | NumPy 2.2.6, rasterio 1.5.0, geopandas 1.1.3, pyogrio 0.12.1 |
| Weight | `/home/work/models/BCE-Net/checkpoint-best-whu.pth` |
| SIBU weight | Not available |
| Real data | Available under `/home/work/data/change_detection/building` |
| Real PoC output | `/home/work/data/results/real_poc/site_001` |

## Environment Verification

Verified on 2026-07-15:

- Required Python imports: pass
- CUDA visibility and H200 allocation: pass
- DCNv2 16×16 CUDA forward: pass
- GeoTIFF and GeoPackage read/write: pass
- Rasterize and polygonize: pass
- WHU checkpoint strict load: pass
- BCE-Net 512×512 CUDA forward: pass, all outputs finite

See [ENVIRONMENT.md](ENVIRONMENT.md) for setup and session reuse details.

## Stage Dashboard

| Stage | Status | Purpose | Summary |
| --- | --- | --- | --- |
| Stage 1–2 | ✅ 완료 | Host inspection and reproducibility files | Docker path recorded; managed-container path selected. |
| Stage 3M–6M | ✅ 완료 | Runtime, DCNv2, model load and forward | H200/CUDA 13 compatibility patch and smoke tests passed. |
| Stage 7M | ✅ 복구 완료 | Geospatial environment | Reusable local venv and one-command bootstrap verified on 2026-07-15. |
| Stage 8M–8V | ✅ 완료 | Synthetic pipeline and vectorization | End-to-end synthetic inference and candidate vectorization passed. |
| Stage 9B | ✅ 완료 | Real small-AOI inference | 25 tiles processed; 0 MISSING and 311 EXCESS candidates at baseline. |
| Stage 9C | ✅ 완료 | Threshold analysis and review package | 225 threshold combinations and 50 top-candidate quicklooks created. |
| Stage 9D | ✅ 완료 | Stratified qualitative inference | 100 tiles, 0.3765 km²; QGIS review package created. |
| Stage 10B | 🕓 대기 | Reviewed object-level evaluation | Waiting for human review labels; no precision/recall/F1 yet. |
| Full AOI | ⏸ 보류 | Production-scale candidate generation | Hold until qualitative review is accepted. |

## Stage 9D Result

| Threshold | MISSING | EXCESS | Total |
| --- | ---: | ---: | ---: |
| Baseline (`new=0.5`, `removed=0.5`, `p90`, `10m²`) | 18 | 465 | 483 |
| Review (`new=0.3`, `removed=0.9`, `mean`, `20m²`) | 16 | 162 | 178 |

- Processed tiles: 100 / 100
- Processed union area: 0.376484659 km²
- QGIS package: `/home/work/data/results/real_poc/site_001/stage9d/qgis_package/stage9d_qgis_review.gpkg`
- Review-threshold candidates remain `UNREVIEWED`.
- Candidates are reviewer-facing evidence, not confirmed mapping errors.

## Current Decision

- Reuse `.venv-bcenet-geo`; do not reinstall it for every terminal session.
- Use `make setup` after workspace recreation, base-image/PyTorch changes, or failed verification.
- Use `./scripts/run_in_env.sh` when shell activation should not be assumed.
- Review `candidate_excess_review_threshold` first in QGIS.
- Review `tile_missing_search` to assess sparse MISSING behavior.
- Do not calculate accuracy metrics until review labels exist.
- Do not run full-AOI inference until qualitative review is accepted.
- Do not upload raw geospatial imagery or reviewer quicklooks externally.

## Active Risks

| Risk | Status | Note |
| --- | --- | --- |
| Environment persistence | ✅ 완화 | Venv and DCNv2 build persist with the workspace; bootstrap is idempotent. |
| Base image ABI change | ⚠️ 감시 | `make setup` rebuilds DCNv2 when the existing extension no longer imports. |
| WHU-only pretrained weight | ⚠️ 위험 | Korea imagery has a domain gap; SIBU weight is unavailable. |
| Human review labels | 🕓 대기 | Current review CSVs remain unreviewed. |
| Full-AOI readiness | ⏸ 보류 | Qualitative acceptance is required first. |
| Candidate interpretation | ✅ 통제 | Outputs are candidates, not confirmed errors or accuracy evidence. |

## Next Gate

Stage 10B can start after:

- QGIS candidates and tile-level missing-search samples receive reviewer labels.
- Ambiguous cases include review comments rather than forced positive/negative labels.
- Label counts are sufficient for an explicitly scoped object-level evaluation.

Until then, do not report precision, recall, F1, or confirmed-error counts.
