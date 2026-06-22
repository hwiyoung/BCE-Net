# BCE-Net Korea PoC Dashboard

Last updated: 2026-06-22

## Current Mode

Managed Container Mode

- Current shell is already inside a managed GPU container.
- Evidence: `/.dockerenv` exists and PID 1 is `docker-init`.
- Docker CLI is not available and `/var/run/docker.sock` is not visible.
- Do not use nested Docker, Docker Compose, Docker socket mount, or Docker-in-Docker in this environment.
- Treat the current managed container as the active PoC development environment.
- Keep `docker/Dockerfile.bcenet` and `docker-compose.yml` for future reproducibility on a real Docker host.

Current managed container:

| Item | Value |
| --- | --- |
| Repo root | `/home/work/BCE-Net` |
| GPU | NVIDIA H200 |
| Python | 3.12.3 |
| PyTorch | `2.10.0a0+b558c986e8.nv25.11` |
| CUDA | 13.0 |
| nvcc | CUDA 13.0 |
| gcc/g++ | 13.3.0 |
| Weight | `/home/work/models/BCE-Net/checkpoint-best-whu.pth` |
| SIBU weight | Not available |
| Real ortho/vector data | Not available yet |
| Geospatial stack | `.venv-bcenet-geo` repaired: `rasterio`, `geopandas`, `pyogrio` OK |

## Stage Dashboard

| Stage | Status | Purpose | Summary |
| --- | --- | --- | --- |
| Stage 1 | ⛔ 차단 | Host Docker/GPU prerequisite check | Original Docker-host path blocked because `docker` CLI and `nvidia-ctk` are unavailable in this shell. |
| Stage 2 | ✅ 완료 | Docker development files | Dockerfile, compose, ignore, and README created for future real Docker host use. |
| Stage 3 | ⛔ 차단 | Docker-container repo/runtime/weight inspection | Original Docker-entry path blocked before container creation. |
| Stage 4 | ⛔ 차단 | Docker-container DCNv2 build/import smoke test | Original Docker-entry path blocked before container creation. |
| Stage 1B | ✅ 완료 | Managed Container Mode decision | Current managed GPU container accepted as the development environment. |
| Stage 3M | ✅ 완료 | Managed container env/weight inspection | Environment and WHU weight inspection scripts ran successfully. |
| Stage 4M | ✅ 완료 | DCNv2 source analysis + build/import smoke test | Original build failure was diagnosed as old DCNv2 legacy API compatibility. |
| Stage 4M-Patch | ✅ 완료 | DCNv2 compatibility patch | Minimal patch applied; `_ext` built and DCNv2 import/CUDA forward smoke passed. |
| Stage 5M | ✅ 완료 | BCE-Net WHU model load smoke test | WHU model imported; checkpoint loaded strict=True after `module.` prefix removal; model moved to CUDA. |
| Stage 6M | ✅ 완료 | Synthetic model forward smoke test | 512x512 dummy forward passed; `predicts_b`, `predicts_mov`, and `predicts_new` produced with no NaN/Inf. |
| Stage 7M | ✅ 완료 | Geospatial stack repair | Local venv repaired geospatial stack; GeoTIFF/GeoPackage/rasterize/polygonize smoke passed. |
| Stage 8M | ✅ 완료 | Synthetic BCE-Net data pipeline | Synthetic raster/vector scene, tiling, dataloader, inference smoke, probability/mask outputs, and output checks passed. |
| Stage 8V | ✅ 완료 | Synthetic candidate vectorization smoke test | Synthetic model-output and controlled probability outputs were vectorized into schema-valid reviewer-facing candidate layers. |
| Stage 9 | 🕓 대기 | Real data mounting and validation | Real orthoimage and building vector data are required before this stage can run. |
| Stage 10 | 🕓 대기 | Candidate generation PoC | Produce reviewer-facing missing/excess building candidates after data arrives. |

## Current Decision

- Continue in Managed Container Mode.
- Do not retry Docker commands from this shell.
- Do not run real-data inference until real ortho/vector data is mounted and validated.
- Interpret BCE-Net `newly constructed` later as digital mapping missing-building candidates.
- Interpret BCE-Net `removed` later as digital mapping excess-building candidates.
- Do not mark candidates as confirmed errors.
- Do not expand scope to roads, depiction-error modeling, Aux Heads, MapRepair, or DragOSM.

## Prompt Panel

### Previous Prompt

Stage 8V: synthetic candidate vectorization smoke test

- Convert synthetic `new` and `removed_raw` masks/probabilities into object-level candidate features.
- Treat `predicts_new` as missing-building candidate support.
- Treat `predicts_mov`/removed as excess-building candidate support.
- Preserve confidence and source tile metadata.
- Do not mark candidates as confirmed errors.

### Current Prompt

Stage 9: real data requirement checklist

- Wait for real orthoimage and existing building vector data.
- Validate mounted paths, CRS, resolution, tile plan, ID columns, and output directories after data arrives.
- Do not upload raw geospatial data externally.
- Keep candidate outputs as reviewer-facing missing/excess candidates.

### Next Prompt

Stage 10: object-level evaluation planning after real data and review samples

- Plan object-level QA/evaluation only after real data and review samples are available.
- Keep BCE-Net outputs as candidate evidence unless reviewer confirmation is added.
- Do not expand scope to roads, depiction-error modeling, Aux Heads, MapRepair, or DragOSM.

## Active Risks

| Risk | Status | Note |
| --- | --- | --- |
| H200 / CUDA / DCNv2 compatibility | ✅ 완료 | Minimal DCNv2 patch built `_ext` and passed import/CUDA forward smoke in the current managed container. |
| WHU-only pretrained weight | ⚠️ 위험 | SIBU weight is unavailable; Korea PoC has domain gap risk. |
| Geospatial packages | ✅ 완료 | `.venv-bcenet-geo` imports `rasterio`, `geopandas`, `pyogrio`; synthetic geospatial smoke passed. |
| Venv NumPy constraints | ⚠️ 위험 | venv uses `numpy 2.5.0`; pip reported unrelated constraints for `astropy`, `catboost`, and `numba`. |
| Docker reproducibility path | ⏸ 보류 | Docker files are preserved, but not executable from this managed container. |
| Real data availability | 🕓 대기 | Actual orthoimage and building vector data are not uploaded yet. |
| Candidate interpretation | ✅ 완료 | Outputs are reviewer-facing candidates, not confirmed errors. |
| Synthetic pipeline interpretation | ✅ 완료 | Stage 8M was an end-to-end smoke test only, not model performance evidence. |
| Candidate vectorization | ✅ 완료 | Stage 8V generated `MISSING`/`EXCESS` candidate layers with `UNREVIEWED` status in synthetic smoke tests. |

## Next Gate

Current Stage 9 gate resolves only when:

- Real orthoimage and existing building vector data are mounted in the managed container.
- CRS, transform/resolution, building ID column, geometry validity, and output paths are validated.
- A real-data tiling/inference dry run plan is recorded before execution.
- No raw geospatial data is uploaded externally.
- Candidate outputs remain reviewer-facing missing/excess candidates.

Stage 9 must not include:

- Accuracy claims without review/reference samples.
- Marking candidates as confirmed errors.
- Docker command execution from this managed container.

## Update Rules

- Update `Last updated` after each stage completes.
- Change completed stages to `✅ 완료`.
- Change the new active stage to `🔄 진행`.
- Move `Current Prompt` to `Previous Prompt`.
- Move `Next Prompt` to `Current Prompt`.
- Write a new `Next Prompt`.
- Mark failed stages as `⛔ 차단` or `⚠️ 위험` and include a short reason.
- Keep detailed logs in `RUNBOOK.md`.
- Keep this dashboard short and status-oriented.
