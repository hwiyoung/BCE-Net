# BCE-Net Korea PoC Runbook

## Stage 1: Host Prerequisite Check

Date: 2026-06-22 16:40 UTC

Scope:
- Confirm host GPU visibility.
- Confirm Docker and Docker Compose availability.
- Confirm NVIDIA Container Toolkit availability.
- Do not modify BCE-Net source code.
- Do not create data directories.
- Do not install Python packages on the host.
- Do not run containerized BCE-Net development yet.

### Host GPU

Command:

```bash
nvidia-smi
```

Result: pass

Observed:
- NVIDIA driver is recognized.
- GPU name: NVIDIA H200
- Driver version: 580.126.20
- CUDA version displayed by driver: 13.0
- GPU memory: 71886 MiB
- No running GPU processes were reported.

### Docker

Commands:

```bash
docker version
docker compose version
docker ps
```

Result: host prerequisite 미충족

Observed:
- `docker version` failed: `/bin/bash: line 1: docker: command not found`
- `docker compose version` failed: `/bin/bash: line 1: docker: command not found`
- `docker ps` failed: `/bin/bash: line 1: docker: command not found`

Interpretation:
- Docker CLI is not available on the host PATH.
- Docker daemon status could not be checked because the Docker CLI is missing.
- Docker Compose v2 availability could not be checked because the Docker CLI is missing.

### NVIDIA Container Toolkit

Command:

```bash
nvidia-ctk --version || true
```

Result: host prerequisite 미충족

Observed:
- `nvidia-ctk` is not installed or not available on the host PATH.
- Output: `/bin/bash: line 1: nvidia-ctk: command not found`

Interpretation:
- NVIDIA Container Toolkit is currently unavailable.
- Docker GPU runtime configuration cannot be verified at this stage.
- Installing Docker and NVIDIA Container Toolkit requires administrator privileges, typically `sudo`.
- No installation or configuration commands were executed in this run.

### Docker GPU Runtime

Status: not checked

Reason:
- NVIDIA Container Toolkit is not available.
- Docker CLI is not available.

If NVIDIA Container Toolkit is installed later, the following commands should be proposed for user-approved execution:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

These commands require administrator privileges and must not be run without explicit approval.

### GPU Container Smoke Test

Executable command for the next stage after Docker and NVIDIA Container Toolkit are available:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Status: not executed

Reason:
- Current blocker is missing Docker CLI and missing NVIDIA Container Toolkit.
- Host NVIDIA driver is working, based on successful `nvidia-smi`.
- Docker GPU visibility is untested.
- NVIDIA Container Toolkit integration is untested.
- Image pull/network availability is untested.

### Stage 1 Summary

Host prerequisite status: host prerequisite 미충족

Pass:
- Host NVIDIA driver and GPU are visible.

Blocked:
- Docker CLI is unavailable.
- Docker daemon status is unknown.
- Docker Compose v2 availability is unknown.
- NVIDIA Container Toolkit is unavailable.
- Docker GPU runtime smoke test cannot be run yet.

Next required action:
- Install or enable Docker Engine/CLI and Docker Compose v2 on the host.
- Install NVIDIA Container Toolkit on the host.
- After user approval, configure Docker runtime with `nvidia-ctk` and restart Docker.
- Re-run Stage 1 checks before starting container-based BCE-Net development.

## Stage 2: Docker Development Environment

Date: 2026-06-22 16:42 UTC

Scope:
- Create Docker environment files for BCE-Net PoC development.
- Do not run Python on the host.
- Do not build DCNv2 on the host.
- Do not run BCE-Net inference.
- Do not modify original BCE-Net model or inference source files.

Repository:
- Repo root: `/home/work/BCE-Net`
- Git commit: `d55e1a1`

Created files:
- `docker/Dockerfile.bcenet`
- `docker-compose.yml`
- `.dockerignore`
- `docker/README.md`

Host directory checks:
- `../models/BCE-Net` exists and is mounted read-only in the container.
- `../data` exists and may be empty.
- `../results` exists and is mounted read/write for development and inference outputs.

### Base Image

Primary candidate:

```text
pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
```

Rationale:
- Matches the requested first candidate for BCE-Net/DCNv2 compatibility work.
- Includes PyTorch and CUDA devel tooling needed for later DCNv2 compilation inside the container.

Compatibility risk to validate later:
- Host GPU is NVIDIA H200.
- PyTorch 1.8.1 with CUDA 11.1 is old for Hopper/H200 hardware.
- DCNv2 CUDA extension build or runtime may fail on H200 because of CUDA architecture support.

Alternative candidates to record if the primary image is unavailable or DCNv2 fails:
- `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel`
- `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel`

These alternatives are more suitable for Hopper/H200, but may require BCE-Net/DCNv2 compatibility fixes.

### Original Requirements Review

Existing file:

```text
requirements.txt
```

Observation:
- The file is a Windows conda export (`platform: win-64`).
- It includes Windows-only packages such as `pywin32`, `vc`, `vs2015_runtime`, and `wincertstore`.
- It pins `pytorch=1.8.1`, `torchvision=0.9.1`, and `cudatoolkit=10.2`, while the Docker base candidate uses CUDA 11.1.
- It includes many packages unrelated to this PoC runtime, including GUI, labeling, web, and TensorFlow packages.

Decision:
- Do not install `requirements.txt` directly in the Dockerfile.
- Install the minimal Linux/Python 3.8-compatible runtime and geospatial packages needed for BCE-Net PoC preparation.
- Validate missing imports later inside the container during synthetic smoke testing.

### Dockerfile Contents

The Dockerfile installs:
- `build-essential`, `git`, `gcc`, `g++`, `ninja-build`
- OpenCV headless system libraries
- GDAL/GEOS/PROJ/spatial index system libraries
- `numpy`, `pandas`, `tqdm`
- `opencv-python-headless`
- `scikit-image`, `scikit-learn`, `scipy`
- `matplotlib`, `pillow`
- `rasterio`, `geopandas`, `shapely`, `fiona`, `pyogrio`, `pyproj`, `rtree`

Container settings:
- Working directory: `/workspace/BCE-Net`
- `PYTHONPATH=/workspace/BCE-Net:/workspace/BCE-Net/DCNv2`
- `TZ=Asia/Seoul`
- `PYTHONUNBUFFERED=1`

### Docker Compose

Service:

```text
bcenet
```

Mounts:
- `.:/workspace/BCE-Net`
- `../models/BCE-Net:/workspace/models/BCE-Net:ro`
- `../data:/workspace/data`
- `../results:/workspace/results`

GPU settings:
- `gpus: all`
- `NVIDIA_VISIBLE_DEVICES=all`
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`

Runtime settings:
- `shm_size: "16gb"`
- `ipc: host`
- interactive shell enabled with `stdin_open: true` and `tty: true`

### Build Commands

Run from the BCE-Net repository root after Stage 1 prerequisites are satisfied:

```bash
docker compose build bcenet
docker compose run --rm bcenet bash
```

Compose GPU fallback:

```bash
docker run --rm --gpus all -it \
  --ipc=host \
  --shm-size=16g \
  -v "$PWD":/workspace/BCE-Net \
  -v "$(realpath ../models/BCE-Net)":/workspace/models/BCE-Net:ro \
  -v "$(realpath ../data)":/workspace/data \
  -v "$(realpath ../results)":/workspace/results \
  -w /workspace/BCE-Net \
  bcenet-poc:dev \
  bash
```

### Stage 2 Status

Result: Docker development environment files prepared.

Not executed:
- `docker compose build bcenet`
- `docker compose run --rm bcenet bash`
- DCNv2 build
- Python import checks

Reason:
- Stage 1 host prerequisites are still not satisfied on this host because Docker CLI and NVIDIA Container Toolkit are unavailable.

## Stage 3: Container Repository and Runtime Inspection

Date: 2026-06-22 16:43 UTC

Scope:
- Enter the BCE-Net development container.
- Inspect the mounted repository, pretrained weights, Python/PyTorch/CUDA runtime, and geospatial imports from inside the container only.
- Do not use host Python.
- Do not build DCNv2.
- Do not run inference.

Container entry command:

```bash
docker compose run --rm bcenet bash
```

Result: blocked

Observed:

```text
/bin/bash: line 1: docker: command not found
```

Interpretation:
- Docker CLI is still unavailable on the host.
- The BCE-Net development container could not be started.
- Because container entry failed, no repository inspection commands were run.
- Because container entry failed, `scripts/inspect_env.py` was not created.
- Because container entry failed, `scripts/inspect_bcenet_weights.py` was not created.
- Because container entry failed, `/workspace/results/bcenet_weight_inspection.json` was not created.
- Host Python was not used.

Required next action:
- Satisfy Stage 1 host prerequisites first: Docker Engine/CLI, Docker Compose v2, and NVIDIA Container Toolkit.
- Re-run Stage 1 checks.
- Build the Stage 2 image with `docker compose build bcenet`.
- Re-run Stage 3 from container entry.

## Stage 4: DCNv2 Build and Import Smoke Test

Date: 2026-06-22 16:44 UTC

Scope:
- Analyze BCE-Net DCNv2 import usage from inside the container.
- Build the original BCE-Net `DCNv2` extension from inside the container.
- Run a DCNv2 import and minimal forward smoke test from inside the container.
- Do not run `python setup.py build develop` on the host.
- Do not bypass DCNv2.
- Do not run inference.

Container entry command attempted:

```bash
docker compose run --rm bcenet bash
```

Result: blocked

Observed:

```text
/bin/bash: line 1: docker: command not found
```

Interpretation:
- Docker CLI is still unavailable on the host.
- The BCE-Net development container could not be started.
- The required container-only DCNv2 import usage analysis was not run.
- `python setup.py build develop` was not run.
- `scripts/smoke_dcnv2.py` was not created because development must occur inside the container.
- `python scripts/smoke_dcnv2.py` was not run.
- Host Python was not used.

Planned container commands after prerequisites are fixed:

```bash
grep -R "DCN\|dcn\|DCNv2" -n Testmodel utils dataset *.py || true
cd /workspace/BCE-Net/DCNv2
python setup.py build develop
cd /workspace/BCE-Net
python scripts/smoke_dcnv2.py
```

Failure diagnostics to capture if DCNv2 build fails later:
- Python version
- torch version
- CUDA version reported by torch
- `nvcc --version`
- `gcc --version`
- `g++ --version`
- Core build failure log, last 50 relevant lines

Failure causes to classify later:
- PyTorch/CUDA mismatch
- GCC version mismatch
- DCNv2 source compatibility
- GPU driver/container runtime mismatch

Required next action:
- Satisfy Stage 1 host prerequisites first: Docker Engine/CLI, Docker Compose v2, and NVIDIA Container Toolkit.
- Build the Stage 2 image.
- Re-run Stage 3 successfully.
- Then re-run Stage 4 from inside the container.

## Stage 1 Recovery Gate: Host vs Container Diagnosis

Date: 2026-06-22 16:45 UTC

Purpose:
- Do not force Stage 3/4 retries.
- Determine whether the current shell is the actual Docker host or a restricted container/sandbox.
- Document the host prerequisite recovery plan before resuming Stage 2/3/4.

Reference checked:
- Docker Engine Ubuntu install guide: https://docs.docker.com/engine/install/ubuntu/
- NVIDIA Container Toolkit install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

### Current Shell Detection

Commands and observed outputs:

```bash
hostname
```

```text
main1
```

```bash
whoami
```

```text
work
```

```bash
pwd
```

```text
/home/work/BCE-Net
```

```bash
cat /etc/os-release
```

```text
PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.3 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
UBUNTU_CODENAME=noble
```

```bash
command -v docker || echo "docker CLI not found"
```

```text
docker CLI not found
```

```bash
command -v docker-compose || echo "legacy docker-compose not found"
```

```text
legacy docker-compose not found
```

```bash
docker compose version || true
```

```text
/bin/bash: line 1: docker: command not found
```

```bash
ls -l /var/run/docker.sock 2>/dev/null || true
```

```text
No output. Docker socket is not visible from this shell.
```

```bash
systemctl status docker --no-pager 2>/dev/null || true
```

```text
No output. Docker daemon status could not be confirmed from this shell.
```

```bash
command -v nvidia-smi || echo "nvidia-smi not found"
```

```text
/usr/bin/nvidia-smi
```

```bash
nvidia-smi -L || true
```

```text
GPU 0: NVIDIA H200 (UUID: GPU-94a6df43-79d1-eeb5-609d-4bb624da86ec)
```

```bash
command -v nvidia-ctk || echo "nvidia-ctk not found"
```

```text
nvidia-ctk not found
```

```bash
test -f /.dockerenv && echo "probably inside a Docker container"
```

```text
probably inside a Docker container
```

```bash
cat /proc/1/comm
```

```text
docker-init
```

```bash
grep -E "docker|kubepods|containerd|lxc" /proc/1/cgroup || true
```

```text
No output.
```

Diagnosis:
- The current shell can see `nvidia-smi` and the NVIDIA H200 device.
- The current shell cannot resolve the `docker` executable.
- The current shell cannot see `/var/run/docker.sock`.
- `/.dockerenv` exists and PID 1 is `docker-init`, so this shell is probably already inside a Docker container or restricted sandbox rather than the actual Docker host terminal.
- Docker socket mounting or Docker-in-Docker must not be configured ad hoc. This requires explicit security review and administrator approval.

Required access:
- Run the host prerequisite recovery commands from the actual GPU host terminal, or ask an administrator to provide Docker CLI/daemon access and NVIDIA Container Toolkit on the real host.
- Do not retry Stage 3/4 from this shell while `docker CLI not found` remains true.

### Stage 3/4 Blocker Clarification

Stage 3/4 did not fail inside the BCE-Net container.
The BCE-Net container was never created because the host shell could not resolve the `docker` executable.
Therefore, no BCE-Net runtime, pretrained weight, Python package, CUDA extension, or DCNv2 diagnosis has been performed yet.
The only confirmed blocker is missing host-level Docker CLI and missing NVIDIA Container Toolkit.

Korean summary:
- Stage 3/4는 컨테이너 내부 실행 실패가 아니라, 컨테이너 생성 전 host shell에서 `docker` 실행 파일을 찾지 못해 blocked 되었다.
- 따라서 BCE-Net repo, pretrained weight, Python/PyTorch/CUDA runtime, DCNv2 build에 대한 진단은 아직 수행되지 않았다.
- 다음 조치는 Stage 1 host prerequisite 복구이며, Docker Engine/CLI, Docker Compose v2, NVIDIA Container Toolkit 설치 및 GPU container smoke test 통과 후 Stage 2 image build부터 재개한다.

Not related to this blocker:
- Missing real orthoimage data is not the cause of Stage 3/4 blockage.
- BCE-Net source code has not been diagnosed yet.
- Pretrained weights have not been inspected yet.
- DCNv2 has not been built or tested yet.
- Python/PyTorch/CUDA runtime inside the BCE-Net container has not been inspected yet.

### Docker Engine, CLI, and Compose v2 Recovery Plan

The following commands are for an actual Ubuntu GPU host with administrator approval only.
They were not executed in this run.

```bash
# Docker official repository setup
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update

sudo apt install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

sudo systemctl enable --now docker
sudo systemctl status docker --no-pager

docker version
docker compose version
```

Security note:
- Adding a user to the `docker` group grants Docker daemon access, which is effectively high privilege on the host.
- Do not run this automatically.
- Optional, security review required:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

### NVIDIA Container Toolkit Recovery Plan

The following commands require administrator approval.
They were not executed in this run.

```bash
# NVIDIA Container Toolkit repository setup
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

nvidia-ctk --version
```

Configuration note:
- `sudo nvidia-ctk runtime configure --runtime=docker` modifies Docker daemon configuration.
- It must only be run after explicit user or administrator approval.

### Stage 1 Completion Gate: GPU Container Smoke Test

After Docker and NVIDIA Container Toolkit are installed and configured, the following command must pass:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Pass criterion:
- The command prints GPU information from inside the CUDA container.
- This confirms that Docker, NVIDIA Container Toolkit, host driver access, and GPU runtime wiring are ready.

If it fails, classify the failure as one or more of:
- Docker CLI/daemon problem
- NVIDIA Container Toolkit missing or not configured
- Docker runtime configuration problem
- Host driver problem
- Image pull/network problem
- Permission problem

### Docker Compose GPU Setting

Current `docker-compose.yml` includes:

```yaml
environment:
  NVIDIA_VISIBLE_DEVICES: all
  NVIDIA_DRIVER_CAPABILITIES: compute,utility
gpus: all
```

Decision:
- Keep `gpus: all` for now.
- Do not change Compose GPU syntax until `docker run --rm --gpus all ... nvidia-smi` passes.
- If `docker run --gpus all` passes but Compose GPU access fails, evaluate this alternative Compose syntax:

```yaml
services:
  bcenet:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Resume Order After Host Prerequisites Pass

Run from the actual GPU host after Stage 1 is fixed:

```bash
cd /home/work/BCE-Net

docker version
docker compose version

docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

docker compose build bcenet
docker compose run --rm bcenet bash
```

If Compose container entry fails after the image is built, use this fallback:

```bash
docker run --rm --gpus all -it \
  --ipc=host \
  --shm-size=16g \
  -v "$PWD":/workspace/BCE-Net \
  -v "$(realpath ../models/BCE-Net)":/workspace/models/BCE-Net:ro \
  -v "$(realpath ../data)":/workspace/data \
  -v "$(realpath ../results)":/workspace/results \
  -w /workspace/BCE-Net \
  bcenet-poc:dev \
  bash
```

### Stage 3 Container-Only Checklist

Only after entering the BCE-Net container:

```bash
pwd
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
ls -la /workspace/models/BCE-Net
ls -la /workspace/data || true
ls -la /workspace/results || true
```

Then create:
- `scripts/inspect_env.py`
- `scripts/inspect_bcenet_weights.py`

Then run inside the container:

```bash
python scripts/inspect_env.py
python scripts/inspect_bcenet_weights.py --weights-dir /workspace/models/BCE-Net
```

Expected Stage 3 output:

```text
/workspace/results/bcenet_weight_inspection.json
```

### Stage 4 Container-Only Checklist

Only after Stage 3 succeeds:

```bash
grep -R "DCN\|dcn\|DCNv2" -n Testmodel utils dataset *.py || true

cd /workspace/BCE-Net/DCNv2
python setup.py build develop
cd /workspace/BCE-Net
```

Then create:
- `scripts/smoke_dcnv2.py`

Then run:

```bash
python scripts/smoke_dcnv2.py
```

If DCNv2 build fails, do not bypass it.
Capture:

```bash
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
nvcc --version || true
gcc --version || true
g++ --version || true
```

Record the core 50 failure log lines and classify as one or more of:
- PyTorch/CUDA mismatch
- GCC version mismatch
- DCNv2 source compatibility
- GPU driver/container runtime mismatch
- H200/Hopper architecture compatibility issue

### H200 Compatibility Risk

H200 compatibility risk:
The primary Docker base image `pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel` may be too old for NVIDIA H200/Hopper hardware.
If DCNv2 build or runtime fails with architecture-related errors such as `no kernel image is available for execution on the device`, `unsupported gpu architecture`, or `compute_90` issues, classify it as CUDA/PyTorch/DCNv2 compatibility risk rather than BCE-Net data/model failure.
Alternative base images to evaluate:
- `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel`
- `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel`

These may require BCE-Net/DCNv2 compatibility fixes.

### Explicit Non-Goals and Prohibited Actions During Recovery

Do not:
- Run host Python.
- Run `python setup.py build develop` on the host.
- Try to build DCNv2 without Docker.
- Try BCE-Net inference without Docker.
- Attempt real PoC evaluation before real data is uploaded.
- Mount Docker socket ad hoc.
- Configure Docker-in-Docker ad hoc.
- Run sudo installation or configuration commands without approval.
- Upload raw geospatial data externally.
- Upload pretrained weights externally.
- Mark candidates as confirmed errors.
- Expand scope to road, depiction-error, Aux Head, MapRepair, or DragOSM design.

## Stage 1B: Managed Container Mode Decision

Date: 2026-06-22 18:57 UTC

Current shell is already inside a managed GPU container.

Evidence:
- `/.dockerenv` exists.
- PID 1 is `docker-init`.
- `docker` CLI is not available.
- `/var/run/docker.sock` is not visible.
- NVIDIA H200 is visible through `nvidia-smi`.
- PyTorch CUDA is available.
- `nvcc` and `gcc/g++` are available.

Decision:
- Do not use nested Docker, Docker Compose, or Docker socket mount in this environment.
- Treat the current managed container as the PoC development environment.
- Keep Dockerfile/docker-compose files only for future reproducibility on a real Docker host.
- Reinterpret "do not use host Python" as "do not use actual host Python"; current container Python may be used.

Current focus:
- Inspect current environment.
- Inspect BCE-Net pretrained weight.
- Do not build DCNv2 yet.
- Do not run inference yet.

Recorded current state:
- Repo root: `/home/work/BCE-Net`
- Pretrained weight path: `../models/BCE-Net/checkpoint-best-whu.pth`
- SIBU pretrained weight: not present at this stage
- Results path: `../results`
- Results write access: available
- Python version: `3.12.3`
- PyTorch version: `2.10.0a0+b558c986e8.nv25.11`
- `torch.cuda.is_available`: `True`
- `torch.version.cuda`: `13.0`
- GPU name: `NVIDIA H200`
- GPU capability: `[9, 0]`
- `nvcc`: `/usr/local/cuda/bin/nvcc`, CUDA compiler release `13.0`, `V13.0.88`
- `gcc/g++`: `13.3.0`
- Available Python packages: `numpy`, `pandas`, `cv2`, `PIL`, `torch`, `shapely`, `fiona`, `pyproj`, `tqdm`, `skimage`, `scipy`
- Missing or failing Python packages: `rasterio` missing, `geopandas` import failure, `pyogrio` missing
- Python/PyTorch/CUDA/package details are recorded in `../results/managed_container_env_inspection.json`
- Weight structure details are recorded in `../results/bcenet_weight_inspection.json`

Interpretation:
- BCE-Net `newly constructed` output will be interpreted later as a digital mapping missing-building candidate.
- BCE-Net `removed` output will be interpreted later as a digital mapping excess-building candidate.
- These outputs are reviewer-facing candidate objects, not confirmed errors.
- Existing/building outputs remain reference outputs and are not treated as depiction errors in this stage.

## Stage 3M: Managed Container Environment and Weight Inspection

Date: 2026-06-22 18:58 UTC

Scope:
- Use the current managed GPU container directly.
- Create and run environment inspection script.
- Create and run pretrained weight inspection script.
- Do not run Docker commands.
- Do not build DCNv2 yet.
- Do not run BCE-Net inference yet.

Commands executed:

```bash
mkdir -p scripts ../results ../results/logs
python scripts/inspect_env.py 2>&1 | tee ../results/logs/inspect_env.log
python scripts/inspect_bcenet_weights.py \
  --weights-dir ../models/BCE-Net \
  --out-json ../results/bcenet_weight_inspection.json \
  2>&1 | tee ../results/logs/inspect_bcenet_weights.log
```

Created scripts:
- `scripts/inspect_env.py`
- `scripts/inspect_bcenet_weights.py`

JSON outputs:
- `../results/managed_container_env_inspection.json`
- `../results/bcenet_weight_inspection.json`

Logs:
- `../results/logs/inspect_env.log`
- `../results/logs/inspect_bcenet_weights.log`

Environment inspection result:
- Working directory: `/home/work/BCE-Net`
- User: `work`
- Hostname: `main1`
- OS: Ubuntu 24.04.3 LTS
- Container evidence: `/.dockerenv` exists, PID 1 is `docker-init`
- Git commit: `d55e1a19062481d1c6f302ff3113f9bb2979ac51`
- Python executable: `/usr/bin/python`
- Python version: `3.12.3`
- PyTorch version: `2.10.0a0+b558c986e8.nv25.11`
- `torch.cuda.is_available`: `True`
- `torch.version.cuda`: `13.0`
- GPU name: `NVIDIA H200`
- GPU capability: `[9, 0]`
- Torch arch list includes `sm_90`
- `nvcc`: CUDA compiler release `13.0`, `V13.0.88`
- `gcc`: `13.3.0`
- `g++`: `13.3.0`
- `make`: GNU Make `4.3`
- BCE-Net key files and directories exist.
- `../models/BCE-Net` exists.
- `../results` write check passed.

Package inspection result:
- Available: `numpy`, `pandas`, `cv2`, `PIL`, `torch`, `shapely`, `fiona`, `pyproj`, `tqdm`, `skimage`, `scipy`
- Missing/failing: `rasterio`, `geopandas`, `pyogrio`
- `geopandas` failure: cannot import `_NDFrameIndexer` from `pandas.core.indexing`

Weight inspection result:
- Selected weight: `/home/work/models/BCE-Net/checkpoint-best-whu.pth`
- File size: `127721607` bytes
- Modified time UTC: `2026-06-22T06:48:18.891335+00:00`
- Checkpoint type: `dict['state_dict']`
- Top-level keys: `state_dict`
- Has state_dict: `True`
- State dict key count: `323`
- First key: `module.resnet_features.conv1.weight`
- First tensor shape: `[64, 3, 7, 7]`

Warning:
- Only WHU pretrained weight is available. Domain gap risk exists for Korea PoC.

Stage 3M status: passed

Next recommended step:
- Before DCNv2 build, decide whether to repair the geospatial Python stack in the managed container or defer `rasterio/geopandas/pyogrio` until actual geospatial data arrives.
- Then proceed to DCNv2 import usage analysis and DCNv2 build smoke test in the current managed container.

## Stage 4M: DCNv2 Source Analysis and Build/Import Smoke Test

Date: 2026-06-22

Mode:
- Managed Container Mode.
- Do not use docker/docker compose/docker run.
- Use the current managed GPU container directly.

Scope:
- Analyze BCE-Net DCNv2 usage.
- Build original DCNv2 extension in the current container.
- Run DCNv2 import/minimal forward smoke test.
- Do not load BCE-Net model yet.
- Do not run BCE-Net inference yet.
- Do not install geospatial packages in this stage.

Environment:
- Python: 3.12.3
- PyTorch: 2.10.0a0+b558c986e8.nv25.11
- CUDA: 13.0
- GPU: NVIDIA H200
- GPU capability: [9, 0]
- nvcc: CUDA 13.0
- gcc/g++: 13.3.0

Build strategy:
- Use `TORCH_CUDA_ARCH_LIST="9.0"` for H200.
- Capture full build logs.
- If build fails, classify as DCNv2/PyTorch/CUDA/Python/GCC compatibility issue, not data or model failure.

Commands executed:

```bash
python scripts/analyze_bcenet_source.py \
  2>&1 | tee ../results/logs/analyze_bcenet_source.log

python scripts/precheck_dcnv2_build.py \
  2>&1 | tee ../results/logs/dcnv2_build_precheck.log

grep -R "DCN\|dcn\|DCNv2\|_ext\|deform" -n \
  Testmodel utils dataset DCNv2 *.py \
  2>&1 | tee ../results/logs/dcnv2_usage_grep.log

cd /home/work/BCE-Net/DCNv2

python setup.py clean --all \
  2>&1 | tee ../../results/logs/dcnv2_clean.log

CUDA_HOME=/usr/local/cuda \
TORCH_CUDA_ARCH_LIST="9.0" \
MAX_JOBS=4 \
python setup.py build_ext --inplace \
  2>&1 | tee ../../results/logs/dcnv2_build_ext_inplace.log

cd /home/work/BCE-Net

python scripts/check_dcnv2_build_outputs.py \
  2>&1 | tee ../results/logs/dcnv2_build_outputs.log

PYTHONPATH=/home/work/BCE-Net:/home/work/BCE-Net/DCNv2:$PYTHONPATH \
python scripts/smoke_dcnv2.py \
  2>&1 | tee ../results/logs/dcnv2_smoke_test.log
```

Source analysis:
- JSON: `../results/bcenet_source_analysis.json`
- Markdown: `../results/bcenet_source_analysis.md`
- Log: `../results/logs/analyze_bcenet_source.log`
- Pattern match count: `573`
- BCE-Net model files use `from DCNv2.dcn_v2 import DCN`.
- `DCNv2/dcn_v2.py` imports backend extension as top-level `_ext`.
- Preferred smoke import path: `from DCNv2.dcn_v2 import DCN, DCNv2`.

Build precheck:
- JSON: `../results/dcnv2_build_precheck.json`
- Python: `3.12.3`
- PyTorch: `2.10.0a0+b558c986e8.nv25.11`
- `torch.cuda.is_available`: `True`
- `torch.version.cuda`: `13.0`
- GPU: `NVIDIA H200`
- GPU capability: `[9, 0]`
- Torch arch list includes `sm_90`
- `nvcc`: CUDA `13.0`, `V13.0.88`
- `gcc/g++`: `13.3.0`
- Existing Linux `.so` before build: none
- Existing Windows artifact: `DCNv2/_ext.cp38-win_amd64.pyd`, ignored for Linux/Python 3.12

Build command:

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=4 python setup.py build_ext --inplace
```

Build result: failed

Generated `.so` files:
- None

Build output inspection:
- JSON: `../results/dcnv2_build_outputs.json`
- Python extension suffix: `.cpython-312-x86_64-linux-gnu.so`
- `DCNv2/build` exists after failed build attempt
- No matching `.so` was generated

Core build failure summary:
- CPU files fail on missing legacy Torch header:
  - `fatal error: TH/TH.h: No such file or directory`
  - Affected examples: `src/cpu/dcn_v2_im2col_cpu.cpp`, `src/cpu/dcn_v2_cpu.cpp`, `src/cpu/dcn_v2_psroi_pooling_cpu.cpp`
- CUDA file fails on missing legacy Torch CUDA header:
  - `fatal error: THC/THC.h: No such file or directory`
  - Affected example: `src/cuda/dcn_v2_cuda.cu`
- `TORCH_CUDA_ARCH_LIST="9.0"` was applied; nvcc command included `-gencode=arch=compute_90,code=sm_90`
- The failure occurred before link/import because compilation stopped.

Failure classification:
- PyTorch 2.10 API compatibility
- Python 3.12 compatibility risk
- CUDA 13.0 / nvcc compatibility risk
- old DCNv2 source compatibility
- include/header issue

Not classified as:
- Missing real data
- BCE-Net checkpoint failure
- Korea PoC inference failure
- H200 arch flag failure, because `sm_90` was generated and the observed blocker is missing `TH/TH.h` and `THC/THC.h`

Import smoke result:
- JSON: `../results/dcnv2_smoke_test.json`
- Log: `../results/logs/dcnv2_smoke_test.log`
- `from DCNv2.dcn_v2 import DCN, DCNv2`: failed, `ModuleNotFoundError: No module named '_ext'`
- `from dcn_v2 import DCN, DCNv2`: failed, `ModuleNotFoundError: No module named '_ext'`
- `import DCNv2`: passed only for empty package `__init__.py`
- `import _ext`: failed, `ModuleNotFoundError: No module named '_ext'`

Minimal forward smoke result:
- Not attempted.
- Reason: `DCN` class import failed because `_ext` was not built.
- CPU fallback was not used as success.

Patch applied:
- None.
- Original DCNv2 source was not modified in this stage.

Stage 4M result: blocked by DCNv2 compatibility issue
- DCNv2 build/import did not pass.
- This is not caused by missing real data.
- This is not a BCE-Net checkpoint failure.
- Failure class: PyTorch 2.10 / Python 3.12 / CUDA 13.0 compatibility with old DCNv2 source, specifically missing legacy `TH/TH.h` and `THC/THC.h` headers.
- Next action: analyze minimal compatibility patch or request a compatible PyTorch/CUDA/DCNv2 environment.

## Stage 4M-Patch: DCNv2 Compatibility Patch Analysis

Date: 2026-06-22

Mode:
- Managed Container Mode.
- Current environment is kept.
- No Docker, no sudo, no environment switch.

Reason:
- Stage 4M failed because original DCNv2 includes legacy `TH/TH.h` and `THC/THC.h`.
- `_ext` was not built.
- `from DCNv2.dcn_v2 import DCN` failed.
- This is not caused by real data, checkpoint, geospatial packages, or H200 arch flag.

Goal:
- Inspect legacy DCNv2 API usage.
- Apply minimal PyTorch 2.x/CUDA compatibility patch if feasible.
- Rebuild DCNv2 with `TORCH_CUDA_ARCH_LIST="9.0"`.
- Re-run DCNv2 smoke test.

Commands executed:

```bash
python scripts/analyze_dcnv2_legacy_api.py \
  2>&1 | tee ../results/logs/analyze_dcnv2_legacy_api.log

python scripts/summarize_dcnv2_build_failure.py \
  --log ../results/logs/dcnv2_build_ext_inplace.log \
  2>&1 | tee ../results/logs/summarize_dcnv2_build_failure.log

git status --short > ../results/patches/git_status_before_dcnv2_patch.txt
git diff > ../results/patches/git_diff_before_dcnv2_patch.diff
find DCNv2 -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.cuh" -o -name "setup.py" -o -name "dcn_v2.py" \) \
  | sort > ../results/patches/dcnv2_source_files_before_patch.txt

python scripts/propose_dcnv2_patch_plan.py \
  2>&1 | tee ../results/logs/propose_dcnv2_patch_plan.log

cd /home/work/BCE-Net/DCNv2

python setup.py clean --all \
  2>&1 | tee ../../results/logs/dcnv2_clean_after_patch.log

CUDA_HOME=/usr/local/cuda \
TORCH_CUDA_ARCH_LIST="9.0" \
MAX_JOBS=4 \
python setup.py build_ext --inplace \
  2>&1 | tee ../../results/logs/dcnv2_build_after_patch.log

cd /home/work/BCE-Net

python scripts/check_dcnv2_build_outputs.py \
  --out-json ../results/dcnv2_build_outputs_after_patch.json \
  2>&1 | tee ../results/logs/dcnv2_build_outputs_after_patch.log

PYTHONPATH=/home/work/BCE-Net:/home/work/BCE-Net/DCNv2:$PYTHONPATH \
python scripts/smoke_dcnv2.py \
  --out-json ../results/dcnv2_smoke_test_after_patch.json \
  2>&1 | tee ../results/logs/dcnv2_smoke_after_patch.log
```

Legacy API analysis:
- JSON: `../results/dcnv2_legacy_api_analysis.json`
- Markdown: `../results/dcnv2_legacy_api_analysis.md`
- Log: `../results/logs/analyze_dcnv2_legacy_api.log`
- Files scanned: `14`
- Match count: `187`
- Key legacy issues:
  - Removed headers: `TH/TH.h`, `THC/THC.h`
  - Removed/legacy CUDA helpers: `THCState`, `THCudaCheck`, `THCCeilDiv`
  - Legacy tensor pointer API: `.data<scalar_t>()`
  - Deprecated dispatch input: `tensor.type()`

Build failure summary:
- JSON: `../results/dcnv2_build_failure_summary.json`
- Markdown: `../results/dcnv2_build_failure_summary.md`
- Initial failure class: removed `TH/TH.h` and `THC/THC.h` headers in old DCNv2 source.

Patch plan:
- JSON: `../results/dcnv2_patch_plan.json`
- Markdown: `../results/dcnv2_patch_plan.md`
- Patch size: `medium`
- Recommendation: `apply_minimal_patch`

Pre-patch records:
- Git status: `../results/patches/git_status_before_dcnv2_patch.txt`
- Git diff before source patch: `../results/patches/git_diff_before_dcnv2_patch.diff`
- DCNv2 source file list: `../results/patches/dcnv2_source_files_before_patch.txt`

Patch applied:
- Minimal compatibility patch applied to DCNv2 source.
- Git diff after patch: `../results/patches/git_diff_after_dcnv2_minimal_patch.diff`
- Modified source files:
  - `DCNv2/src/cpu/dcn_v2_cpu.cpp`
  - `DCNv2/src/cpu/dcn_v2_im2col_cpu.cpp`
  - `DCNv2/src/cpu/dcn_v2_psroi_pooling_cpu.cpp`
  - `DCNv2/src/cuda/dcn_v2_cuda.cu`
  - `DCNv2/src/cuda/dcn_v2_im2col_cuda.cu`
  - `DCNv2/src/cuda/dcn_v2_im2col_cuda.h`
  - `DCNv2/src/cuda/dcn_v2_psroi_pooling_cuda.cu`
- Patch actions:
  - Removed legacy `TH/TH.h` and `THC/THC.h` includes.
  - Replaced `THArgCheck` with `AT_ASSERTM`.
  - Replaced active `.data<scalar_t>()` calls with `.data_ptr<scalar_t>()`.
  - Replaced `THCudaCheck` with `C10_CUDA_CHECK`.
  - Replaced `THCCeilDiv(out_size, 512L)` with explicit integer ceiling division.
  - Replaced failing `AT_DISPATCH_FLOATING_TYPES(tensor.type(), ...)` calls with `tensor.scalar_type()`.
  - Used `at::cuda::getCurrentCUDAStream()` for current CUDA stream access.

Build after patch:
- Command: `CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=4 python setup.py build_ext --inplace`
- Build log: `../results/logs/dcnv2_build_after_patch.log`
- First post-patch attempt log: `../results/logs/dcnv2_build_after_patch_attempt1.log`
- First post-patch attempt found the next compatibility issue: `AT_DISPATCH_FLOATING_TYPES(input.type(), ...)` no longer accepts `DeprecatedTypeProperties` under the current PyTorch API.
- Final build result: `pass`
- `TORCH_CUDA_ARCH_LIST="9.0"` was used; nvcc generated `compute_90/sm_90` code.

Generated `.so` files:
- `/home/work/BCE-Net/DCNv2/_ext.cpython-312-x86_64-linux-gnu.so`
- `/home/work/BCE-Net/DCNv2/build/lib.linux-x86_64-cpython-312/_ext.cpython-312-x86_64-linux-gnu.so`
- Build output JSON: `../results/dcnv2_build_outputs_after_patch.json`
- Python extension suffix: `.cpython-312-x86_64-linux-gnu.so`
- ABI suffix match: `true`

Import smoke result:
- JSON: `../results/dcnv2_smoke_test_after_patch.json`
- Log: `../results/logs/dcnv2_smoke_after_patch.log`
- `from DCNv2.dcn_v2 import DCN, DCNv2`: passed
- `from dcn_v2 import DCN, DCNv2`: passed
- `import DCNv2`: passed
- `import _ext`: passed

Minimal forward smoke result:
- Status: `forward_passed`
- Device: CUDA on `NVIDIA H200`
- Input shape: `[1, 3, 16, 16]`
- Output shape: `[1, 4, 16, 16]`
- Output dtype: `torch.float32`
- CPU fallback was not used.

Stage 4M-Patch status: pass
- Current managed container environment was kept.
- No Docker commands were used.
- No sudo commands were used.
- No BCE-Net model load, checkpoint load, BCE-Net forward, real data inference, or geospatial package installation was performed.
- The Stage 4M blocker was DCNv2/PyTorch/CUDA source compatibility, not data, checkpoint, or geospatial package availability.

Next stage:
- Stage 5M: BCE-Net WHU model load smoke test.

## Stage 5M: BCE-Net WHU Model Load Smoke Test

Date: 2026-06-22

Mode:
- Managed Container Mode.
- No Docker, no sudo, no external downloads.

Prerequisite:
- Stage 4M-Patch passed.
- DCNv2 `_ext` backend was built successfully.
- DCNv2 import and minimal CUDA forward smoke test passed.

Scope:
- Import BCE-Net WHU model class.
- Instantiate model without external downloads.
- Load `/home/work/models/BCE-Net/checkpoint-best-whu.pth`.
- Check state_dict compatibility.
- Move model to CUDA if loading succeeds.
- Do not run BCE-Net forward yet.
- Do not run inference yet.

Commands executed:

```bash
python scripts/analyze_bcenet_model_load_source.py \
  2>&1 | tee ../results/logs/analyze_bcenet_model_load_source.log

PYTHONPATH=/home/work/BCE-Net:/home/work/BCE-Net/DCNv2:$PYTHONPATH \
python scripts/smoke_bcenet_model_load.py \
  --weights /home/work/models/BCE-Net/checkpoint-best-whu.pth \
  --out-json ../results/bcenet_model_load_smoke.json \
  --device cuda:0 \
  --model-source whu \
  2>&1 | tee ../results/logs/bcenet_model_load_smoke.log
```

Model source analysis:
- JSON: `../results/bcenet_model_load_source_analysis.json`
- Markdown: `../results/bcenet_model_load_source_analysis.md`
- Log: `../results/logs/analyze_bcenet_model_load_source.log`
- Original WHU model class: `Testmodel.CDResWHU.Baseline34`
- Original test script constructor: `Baseline34(pretrained=True).cuda()`
- Original test script wraps model with `torch.nn.DataParallel(net)` before checkpoint load.
- Original checkpoint load: `net.load_state_dict(torch.load(trained_model)['state_dict'])`
- Original forward call recorded only: `predicts_b, predicts_mov, predicts_new, _, _ = net.forward(inputs, labels_o)`

Model load smoke:
- JSON: `../results/bcenet_model_load_smoke.json`
- Log: `../results/logs/bcenet_model_load_smoke.log`
- DCNv2 import: passed
- Model import: `from Testmodel.CDResWHU import Baseline34` passed
- Constructor used: `Baseline34(pretrained=False)`
- External download prevention:
  - Active `Baseline34` accepts `pretrained=False` but internally calls `resnet34(pretrained=True)`.
  - Smoke script monkeypatched `Testmodel.CDResWHU.resnet34` to force `pretrained=False`.
  - `load_state_dict_from_url` was guarded against external download.
  - Download guard triggered calls: `0`
  - Forced `resnet34(pretrained=False)` calls observed: `3`

Checkpoint:
- Path: `/home/work/models/BCE-Net/checkpoint-best-whu.pth`
- Load method: `torch.load(weights_only=True)`
- Structure: `dict['state_dict']`
- State dict key count: `323`
- All original checkpoint keys have `module.` prefix: `true`
- First key: `module.resnet_features.conv1.weight`

Strict load attempts:
- `model original state_dict strict=True`: failed as expected because checkpoint keys include `module.` prefix and the unwrapped model keys do not.
- `model module-prefix-stripped state_dict strict=True`: passed
- Missing keys after selected strict load: `0`
- Unexpected keys after selected strict load: `0`
- Shape mismatch keys after selected strict load: `0`
- `strict=False` was not needed and was not used as a pass condition.

DataParallel/prefix handling:
- Original checkpoint was saved from a DataParallel-wrapped model.
- For Stage 5M smoke loading, `module.` prefix removal was sufficient.
- DataParallel wrapping was not required for the selected strict=True load.

Model CUDA move:
- Result: passed
- Device: `cuda:0`
- Eval mode: `true`

Model size and DCNv2 modules:
- Parameter count: `31,877,932`
- Trainable parameter count: `31,877,932`
- DCN module count: `1`
- DCN module path: `dcn`

Forward:
- BCE-Net forward was not executed.
- Dummy tensor forward is reserved for Stage 6M.

Stage 5M result: pass
- BCE-Net WHU model class was imported.
- WHU checkpoint was loaded with strict=True after `module.` prefix removal.
- Model was moved to CUDA and set to eval mode.
- This result is not an inference result and does not use real ortho/vector data.

Next stage:
- Stage 6M: BCE-Net dummy tensor forward smoke test.

## Stage 6M: BCE-Net Dummy Tensor Forward Smoke Test

Date: 2026-06-22

Mode:
- Managed Container Mode.
- No Docker, no sudo, no external downloads.

Prerequisites:
- Stage 4M-Patch passed.
- DCNv2 `_ext` backend was built successfully.
- DCNv2 import and minimal CUDA forward smoke test passed.
- Stage 5M passed.
- `Testmodel.CDResWHU.Baseline34` was imported.
- WHU checkpoint was loaded with strict=True after stripping `module.` prefix.
- Model was moved to CUDA and set to eval mode.

Scope:
- Run BCE-Net forward with dummy tensors only.
- Confirm output tuple structure and output tensor shapes.
- Confirm `predicts_b`, `predicts_mov`, `predicts_new` can be produced.
- Do not run real inference.
- Do not run geospatial preprocessing.

Commands executed:

```bash
python scripts/analyze_bcenet_forward_source.py \
  2>&1 | tee ../results/logs/analyze_bcenet_forward_source.log

PYTHONPATH=/home/work/BCE-Net:/home/work/BCE-Net/DCNv2:$PYTHONPATH \
python scripts/smoke_bcenet_forward.py \
  --weights /home/work/models/BCE-Net/checkpoint-best-whu.pth \
  --out-json ../results/bcenet_forward_smoke.json \
  --device cuda:0 \
  --height 512 \
  --width 512 \
  --batch-size 1 \
  --input-mode random \
  --old-mask-mode synthetic_rect \
  2>&1 | tee ../results/logs/bcenet_forward_smoke.log
```

Forward source analysis:
- JSON: `../results/bcenet_forward_source_analysis.json`
- Markdown: `../results/bcenet_forward_source_analysis.md`
- Log: `../results/logs/analyze_bcenet_forward_source.log`
- Original WHU forward call: `predicts_b, predicts_mov, predicts_new, _, _ = net.forward(inputs, labels_o)`
- Input tensor contract: `inputs` is `[B, 3, H, W]` float tensor on CUDA.
- `labels_o` role: historical/old building footprint mask.
- `labels_o` shape evidence:
  - Test script indexes `labels_o[index]` as a 2D mask.
  - Model forward uses `torch.unsqueeze(labelso, dim=1)`.
  - Likely runtime shape is `[B, H, W]`.
- Output mapping:
  - `outputs[0]`: `predicts_b`
  - `outputs[1]`: `predicts_mov`
  - `outputs[2]`: `predicts_new`
  - `outputs[3]`: `feat_all`
  - `outputs[4]`: `feat_mov`
- Sigmoid post-processing in original script:
  - `torch.sigmoid(predicts_new)`
  - `torch.sigmoid(predicts_mov)`
  - `torch.sigmoid(predicts_b)`
- Thresholding at `0.5` is recorded only and was not executed in Stage 6M.

Forward smoke:
- JSON: `../results/bcenet_forward_smoke.json`
- Log: `../results/logs/bcenet_forward_smoke.log`
- DCNv2 import: passed
- Model load: passed with Stage 5M strict=True path after removing `module.` prefix
- Model class: `Testmodel.CDResWHU.Baseline34`
- Checkpoint path: `/home/work/models/BCE-Net/checkpoint-best-whu.pth`
- Input shape: `[1, 3, 512, 512]`
- First requested `labels_o` candidate `[1, 1, 512, 512]`: failed as diagnostic because the model internally applies `torch.unsqueeze(labelso, dim=1)`, producing a 5D tensor for `conv2d`.
- Selected `labels_o` shape: `[1, 512, 512]`
- Forward call used: `model.forward(inputs, labels_o)`
- Size fallback used: `false`
- Output type: `tuple`
- Output count: `5`

Output summaries:
- `predicts_b`: shape `[1, 1, 512, 512]`, dtype `torch.float32`, device `cuda:0`, min `-3.612304210662842`, max `3.0757033824920654`, mean `1.0722819566726685`, NaN `false`, Inf `false`
- `predicts_mov`: shape `[1, 1, 512, 512]`, dtype `torch.float32`, device `cuda:0`, min `-6.99284553527832`, max `12.718271255493164`, mean `-2.5365514755249023`, NaN `false`, Inf `false`
- `predicts_new`: shape `[1, 1, 512, 512]`, dtype `torch.float32`, device `cuda:0`, min `-23.219348907470703`, max `-2.2819161415100098`, mean `-9.022745132446289`, NaN `false`, Inf `false`
- `feat_all`: shape `[1, 1, 512, 512]`, dtype `torch.float32`, device `cuda:0`, NaN `false`, Inf `false`
- `feat_mov`: shape `[1, 1, 512, 512]`, dtype `torch.float32`, device `cuda:0`, NaN `false`, Inf `false`

Sigmoid summaries:
- `sigmoid_predicts_b`: min `0.026280289515852928`, max `0.9558793306350708`, mean `0.7042233347892761`, NaN `false`, Inf `false`
- `sigmoid_predicts_mov`: min `0.0009175865561701357`, max `0.9999970197677612`, mean `0.2768421173095703`, NaN `false`, Inf `false`
- `sigmoid_predicts_new`: min `8.240715054785852e-11`, max `0.09263177216053009`, mean `0.008416125550866127`, NaN `false`, Inf `false`

Timing and CUDA memory:
- Forward timing: `263.47711589187384` ms
- CUDA memory before selected forward:
  - allocated: `134048768` bytes
  - reserved: `157286400` bytes
- CUDA memory after selected forward:
  - allocated: `172846080` bytes
  - reserved: `1268776960` bytes
  - max allocated: `894266368` bytes
  - max reserved: `1268776960` bytes

Failure classification:
- None for selected forward.
- The diagnostic `[B, 1, H, W]` failure is a `labels_o` shape mismatch, not a model/data/geospatial failure.

Stage 6M result: pass
- BCE-Net WHU dummy forward succeeded.
- `predicts_b`, `predicts_mov`, and `predicts_new` outputs were produced.
- Output tensors had no NaN/Inf.
- Sigmoid post-processing was applicable.
- Real orthoimage inference was not executed.
- Real vector data was not used.
- No candidate files or vectorization outputs were produced.

Next stage:
- Stage 7M: geospatial stack repair before synthetic and real data tiling.

## Stage 7M: Geospatial Stack Repair

Date: 2026-06-22

Mode:
- Managed Container Mode.
- No Docker, no sudo.
- Do not modify global Python packages unless explicitly approved.
- Use a local venv `.venv-bcenet-geo` with `--system-site-packages`.

Prerequisites:
- Stage 4M-Patch passed.
- Stage 5M passed.
- Stage 6M passed.
- BCE-Net model forward works with dummy 512x512 tensors.

Scope:
- Repair rasterio/geopandas/pyogrio environment.
- Run geospatial read/write/rasterize/polygonize smoke tests.
- Confirm BCE-Net/DCNv2 still works inside the venv.
- Do not run real inference.
- Do not process real geospatial data yet.

Commands executed:

```bash
python scripts/inspect_geospatial_env.py \
  --out-json ../results/geospatial_env_inspection_before_venv.json \
  2>&1 | tee ../results/logs/inspect_geospatial_env_before_venv.log

python -m venv --system-site-packages .venv-bcenet-geo

source .venv-bcenet-geo/bin/activate
python -V
python -m pip --version

PYTHONPATH=/home/work/BCE-Net:/home/work/BCE-Net/DCNv2:$PYTHONPATH \
python - <<'PY' 2>&1 | tee ../results/logs/venv_torch_dcnv2_check.log
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
from DCNv2.dcn_v2 import DCN, DCNv2
print("DCNv2 import OK")
PY

python -m pip install --upgrade pip setuptools wheel

python -m pip install --only-binary=:all: \
  "rasterio>=1.4,<2" \
  "geopandas>=1.0,<2" \
  "pyogrio>=0.10,<1" \
  2>&1 | tee ../results/logs/install_geospatial_venv.log

python scripts/inspect_geospatial_env.py \
  --out-json ../results/geospatial_env_inspection_after_venv.json \
  2>&1 | tee ../results/logs/inspect_geospatial_env_after_venv.log

python scripts/smoke_geospatial_stack.py \
  --out-dir ../results/geospatial_smoke \
  --out-json ../results/geospatial_smoke_result.json \
  2>&1 | tee ../results/logs/smoke_geospatial_stack.log

PYTHONPATH=/home/work/BCE-Net:/home/work/BCE-Net/DCNv2:$PYTHONPATH \
python scripts/smoke_bcenet_forward.py \
  --weights /home/work/models/BCE-Net/checkpoint-best-whu.pth \
  --out-json ../results/bcenet_forward_smoke_after_geo_venv.json \
  --device cuda:0 \
  --height 512 \
  --width 512 \
  --batch-size 1 \
  --input-mode random \
  --old-mask-mode synthetic_rect \
  2>&1 | tee ../results/logs/bcenet_forward_smoke_after_geo_venv.log
```

Pre-venv geospatial inspection:
- JSON: `../results/geospatial_env_inspection_before_venv.json`
- Log: `../results/logs/inspect_geospatial_env_before_venv.log`
- Python executable: `/usr/bin/python`
- Status: `partial`
- Failed imports: `rasterio`, `geopandas`, `pyogrio`
- `geopandas` failure: pandas compatibility issue, `ImportError: cannot import name '_NDFrameIndexer' from pandas.core.indexing`
- Available before venv:
  - `numpy`: `1.26.4`
  - `pandas`: `2.3.3`
  - `shapely`: `1.7.0`
  - `fiona`: `1.10.1`, GDAL `3.9.2`
  - `pyproj`: `3.7.2`, PROJ `9.5.1`
- System command GDAL:
  - `gdalinfo --version`: GDAL `3.8.4`
  - `ogrinfo --version`: GDAL `3.8.4`

Venv:
- Path: `/home/work/BCE-Net/.venv-bcenet-geo`
- Python executable: `/home/work/BCE-Net/.venv-bcenet-geo/bin/python`
- Created with `--system-site-packages`
- Torch/DCNv2 venv check log: `../results/logs/venv_torch_dcnv2_check.log`
- Torch in venv: `2.10.0a0+b558c986e8.nv25.11`
- CUDA in venv: available, CUDA `13.0`
- GPU in venv: `NVIDIA H200`
- DCNv2 import in venv: passed

Install result:
- Install log: `../results/logs/install_geospatial_venv.log`
- Command used binary wheels with `--only-binary=:all:`
- Result: success
- Installed into venv:
  - `rasterio`: `1.5.0`
  - `geopandas`: `1.1.3`
  - `pyogrio`: `0.12.1`
  - `shapely`: `2.1.2`
  - `numpy`: `2.5.0`
  - `affine`: `2.4.0`
- System/global packages were not uninstalled or upgraded.
- Resolver warnings recorded:
  - `astropy`, `catboost`, and `numba` declare older NumPy constraints.
  - These warnings are scoped to the venv and did not break Stage 7M smoke tests.

Post-venv geospatial inspection:
- JSON: `../results/geospatial_env_inspection_after_venv.json`
- Log: `../results/logs/inspect_geospatial_env_after_venv.log`
- Status: `pass`
- Failed imports: none
- Package versions:
  - `numpy`: `2.5.0`
  - `pandas`: `2.3.3`
  - `rasterio`: `1.5.0`, GDAL `3.12.1`, PROJ `9.7.1`, GEOS `3.14.1`
  - `geopandas`: `1.1.3`
  - `pyogrio`: `0.12.1`, GDAL `[3, 11, 4]`
  - `shapely`: `2.1.2`, GEOS `3.13.1`
  - `fiona`: `1.10.1`, GDAL `3.9.2`
  - `pyproj`: `3.7.2`, PROJ `9.5.1`
  - `scipy`: `1.16.3`
  - `skimage`: `0.25.0`
  - `cv2`: `4.11.0`
- System command GDAL remains:
  - `gdalinfo --version`: GDAL `3.8.4`
  - `ogrinfo --version`: GDAL `3.8.4`

Geospatial smoke test:
- JSON: `../results/geospatial_smoke_result.json`
- Log: `../results/logs/smoke_geospatial_stack.log`
- Status: `pass`
- CRS used: `EPSG:5186`
- Synthetic GeoTIFF:
  - Path: `../results/geospatial_smoke/synthetic_ortho.tif`
  - Shape: `[3, 128, 128]`
  - dtype: `uint8`
  - CRS preserved: `EPSG:5186`
  - Transform preserved: `[1.0, 0.0, 200000.0, 0.0, -1.0, 600000.0]`
- Synthetic GeoPackage:
  - Path: `../results/geospatial_smoke/synthetic_buildings.gpkg`
  - Write engine: `pyogrio`
  - Feature count: `3`
  - CRS preserved: `EPSG:5186`
  - Geometry validity: `true`
- Rasterize:
  - Path: `../results/geospatial_smoke/synthetic_old_footprint.tif`
  - Result: `pass`
  - Positive pixel count: `2000`
- Polygonize:
  - Path: `../results/geospatial_smoke/synthetic_polygonized_mask.gpkg`
  - Result: `pass`
  - Feature count: `3`
  - CRS preserved: `EPSG:5186`
  - Geometry validity: `true`

BCE-Net forward after venv:
- JSON: `../results/bcenet_forward_smoke_after_geo_venv.json`
- Log: `../results/logs/bcenet_forward_smoke_after_geo_venv.log`
- Result: `pass`
- Python executable: `/home/work/BCE-Net/.venv-bcenet-geo/bin/python`
- DCNv2 import: passed
- WHU checkpoint strict load: passed after removing `module.` prefix
- Input shape: `[1, 3, 512, 512]`
- `labels_o` shape: `[1, 512, 512]`
- Output count: `5`
- `predicts_b`, `predicts_mov`, `predicts_new`: produced, no NaN/Inf
- Forward timing: `187.46877368539572` ms

Failure classification:
- None for Stage 7M pass.

Stage 7M result: pass
- Local geospatial venv `.venv-bcenet-geo` was created with system-site-packages.
- `rasterio`, `geopandas`, and `pyogrio` import succeeded.
- Synthetic GeoTIFF and GeoPackage read/write smoke tests passed.
- Rasterize and polygonize smoke tests passed.
- BCE-Net forward smoke still passed inside the venv.
- Real data inference was not executed.
- Real geospatial data was not processed.

Next stage:
- Stage 8M: synthetic BCE-Net data pipeline.

## Stage 8M: Synthetic BCE-Net Data Pipeline

Date: 2026-06-22

Mode:
- Managed Container Mode.
- Use `.venv-bcenet-geo`.
- No Docker, no sudo.
- No real geospatial data.

Prerequisites:
- Stage 4M-Patch passed.
- Stage 5M passed.
- Stage 6M passed.
- Stage 7M passed.
- BCE-Net model forward works with `[B,3,512,512]` image input and `[B,512,512]` historical footprint input.

Scope:
- Generate synthetic current ortho and old building vector.
- Convert old building vector into BCE-Net historical footprint raster.
- Create 512x512 tiles.
- Create Korea synthetic dataloader.
- Run BCE-Net inference smoke on synthetic tiles.
- Save probability/mask outputs.
- Do not perform real data inference.
- Do not evaluate model accuracy.
- Do not vectorize final candidates in this stage.

Created files:
- `scripts/create_synthetic_bcenet_scene.py`
- `scripts/prepare_korea_bcenet_tiles.py`
- `dataset/cd_dataload_korea_512.py`
- `test_model_korea.py`
- `scripts/check_synthetic_bcenet_outputs.py`

Commands executed:

```bash
source /home/work/BCE-Net/.venv-bcenet-geo/bin/activate

python scripts/create_synthetic_bcenet_scene.py \
  --out-dir ../results/dev_synthetic/raw \
  --width 1024 \
  --height 1024 \
  --crs EPSG:5186 \
  --pixel-size 0.5 \
  --id-col BLDG_ID \
  --seed 42 \
  2>&1 | tee ../results/logs/create_synthetic_bcenet_scene.log

python scripts/prepare_korea_bcenet_tiles.py \
  --ortho ../results/dev_synthetic/raw/synthetic_current_ortho.tif \
  --buildings ../results/dev_synthetic/raw/synthetic_old_buildings.gpkg \
  --out-dir ../results/dev_synthetic/korea_poc \
  --id-col BLDG_ID \
  --tile-size 512 \
  --overlap 64 \
  --min-valid-ratio 0.1 \
  --image-format tif \
  --mask-format tif \
  --create-dummy-labels true \
  2>&1 | tee ../results/logs/prepare_korea_bcenet_tiles_synthetic.log

PYTHONPATH=/home/work/BCE-Net:/home/work/BCE-Net/DCNv2:$PYTHONPATH \
python test_model_korea.py \
  --csv ../results/dev_synthetic/korea_poc/dataset/test_korea.csv \
  --weights /home/work/models/BCE-Net/checkpoint-best-whu.pth \
  --out-dir ../results/dev_synthetic/res-korea \
  --device cuda:0 \
  --batch-size 1 \
  --num-workers 0 \
  --threshold-new 0.5 \
  --threshold-removed 0.5 \
  --threshold-building 0.5 \
  --save-prob true \
  --save-mask true \
  --has-gt false \
  2>&1 | tee ../results/logs/test_model_korea_synthetic.log

python scripts/check_synthetic_bcenet_outputs.py \
  --tile-root ../results/dev_synthetic/korea_poc \
  --inference-dir ../results/dev_synthetic/res-korea \
  --out-json ../results/dev_synthetic/res-korea/output_check.json \
  2>&1 | tee ../results/logs/check_synthetic_bcenet_outputs.log
```

Synthetic raw scene:
- Manifest: `../results/dev_synthetic/raw/synthetic_scene_manifest.json`
- Current ortho: `../results/dev_synthetic/raw/synthetic_current_ortho.tif`
- Old building vector: `../results/dev_synthetic/raw/synthetic_old_buildings.gpkg`
- Reference change vector: `../results/dev_synthetic/raw/synthetic_reference_changes.gpkg`
- CRS: `EPSG:5186`
- Size: `1024 x 1024`
- Pixel size: `0.5`
- Old building count: `5`
- Reference change count: `4`
- Roles:
  - existing-like old/current buildings: `3`
  - removed-like old-only buildings: `2`
  - newly-constructed-like current-only buildings: `2`
- Reference change vector is smoke-test reference context only, not PoC performance ground truth.

Tile preparation:
- Summary: `../results/dev_synthetic/korea_poc/metadata/prepare_summary.json`
- Tile index: `../results/dev_synthetic/korea_poc/metadata/tile_index.geojson`
- Dataset CSV: `../results/dev_synthetic/korea_poc/dataset/test_korea.csv`
- Tile size: `512`
- Overlap: `64`
- Stride: `448`
- Generated tile count: `9`
- Old footprint value convention: `0/1 uint8` on disk.
- Dataloader convention: `labels_o` item shape is `[H,W]` float32, so batch shape is `[B,H,W]`.
- Dummy label rasters are placeholders for loader compatibility, not metric ground truth.

Korea dataloader:
- Path: `dataset/cd_dataload_korea_512.py`
- Return order:
  - `inputs`
  - `labels_o`
  - `labels_n`
  - `labels_m`
  - `labels_b`
  - `labels`
  - `tile_id`
- `labels_o` item shape observed during inference: `[512, 512]`
- `labels_o` batch shape observed during inference: `[1, 512, 512]`

Synthetic inference:
- Script: `test_model_korea.py`
- Summary: `../results/dev_synthetic/res-korea/summary.json`
- Log: `../results/logs/test_model_korea_synthetic.log`
- Weight: `/home/work/models/BCE-Net/checkpoint-best-whu.pth`
- Model: `Testmodel.CDResWHU.Baseline34(pretrained=False)`
- Checkpoint structure: `dict['state_dict']`
- Strict load: passed after removing `module.` prefix
- DCNv2 import: passed
- Tile count: `9`
- Successful tiles: `9`
- Failed tiles: `0`
- Metrics: skipped because `--has-gt false`
- Real data inference: not executed
- Candidate vectorization: not executed
- Output mapping:
  - `predicts_b`: `outputs[0]`
  - `predicts_mov`: `outputs[1]`
  - `predicts_new`: `outputs[2]`
- Probability output directories:
  - `../results/dev_synthetic/res-korea/prob/building`
  - `../results/dev_synthetic/res-korea/prob/removed`
  - `../results/dev_synthetic/res-korea/prob/new`
- Mask output directories:
  - `../results/dev_synthetic/res-korea/mask/building`
  - `../results/dev_synthetic/res-korea/mask/removed_raw`
  - `../results/dev_synthetic/res-korea/mask/new`
- Preview output directory:
  - `../results/dev_synthetic/res-korea/preview`
- Probability stats:
  - `predicts_b`/building min-max: `0.007824080064892769` to `0.999997615814209`
  - `predicts_mov`/removed min-max: `2.423543992335908e-05` to `0.7839024662971497`
  - `predicts_new`/new min-max: `1.7823546355574625e-28` to `0.9999996423721313`
- CUDA memory after inference:
  - allocated: `172846080` bytes
  - reserved: `1272971264` bytes
  - max allocated: `900033536` bytes
  - max reserved: `1272971264` bytes

Output check:
- JSON: `../results/dev_synthetic/res-korea/output_check.json`
- Log: `../results/logs/check_synthetic_bcenet_outputs.log`
- Status: `pass`
- Directory counts:
  - `prob/building`: `9`
  - `prob/removed`: `9`
  - `prob/new`: `9`
  - `mask/building`: `9`
  - `mask/removed_raw`: `9`
  - `mask/new`: `9`
  - `preview`: `9`
- Probability rasters: `float32`, shape `512 x 512`, CRS/transform preserved, values in `[0,1]`
- Mask rasters: `uint8`, shape `512 x 512`, CRS/transform preserved
- Preview PNGs: `9`
- Failure count: `0`

Notes:
- NumPy/rasterio emitted a deprecation warning while reading arrays under NumPy `2.5.0`.
- The warning did not fail output creation or validation.
- Synthetic probability patterns are not interpreted as Korea PoC model performance.

Failure classification:
- None for Stage 8M pass.

Stage 8M result: pass
- Synthetic current ortho and old building vector were generated.
- Old building vector was rasterized into historical footprint tiles.
- Korea synthetic dataloader returned `labels_o` as `[H,W]`, producing batched `[B,H,W]`.
- BCE-Net inference smoke passed on synthetic tiles.
- `predicts_b`, `predicts_mov`, and `predicts_new` probability/mask outputs were saved.
- No real data inference or accuracy evaluation was performed.

Next stage:
- Stage 8V: synthetic candidate vectorization smoke test, or Stage 9 if real data is uploaded.

## Stage 8V: Synthetic Candidate Vectorization Smoke Test

Date: 2026-06-22

Mode:
- Managed Container Mode.
- Use `.venv-bcenet-geo`.
- No Docker, no sudo.
- No real geospatial data.

Prerequisites:
- Stage 8M passed.
- Synthetic BCE-Net inference outputs exist.
- Probability and mask GeoTIFFs preserve CRS/transform.
- Old ID rasters exist.

Scope:
- Convert synthetic BCE-Net pixel/probability outputs into object-level candidate vector layers.
- `prob/new` is interpreted as building missing candidate evidence.
- `prob/removed` is interpreted as building excess candidate evidence.
- Results are candidate layers only, not confirmed errors.
- Do not evaluate real model accuracy.
- Do not process real data yet.

Created files:
- `scripts/vectorize_bcenet_candidates.py`
- `scripts/create_controlled_candidate_prob_maps.py`
- `scripts/check_candidate_vector_outputs.py`
- `scripts/make_candidate_vector_preview.py`

Commands executed:

```bash
source /home/work/BCE-Net/.venv-bcenet-geo/bin/activate

python scripts/vectorize_bcenet_candidates.py \
  --tile-index ../results/dev_synthetic/korea_poc/metadata/tile_index.geojson \
  --inference-dir ../results/dev_synthetic/res-korea \
  --old-buildings ../results/dev_synthetic/raw/synthetic_old_buildings.gpkg \
  --old-id-dir ../results/dev_synthetic/korea_poc/tiles/old_id \
  --out ../results/dev_synthetic/res-korea/vector/bcenet_building_candidates.gpkg \
  --threshold-new 0.5 \
  --threshold-removed 0.5 \
  --threshold-stat p90 \
  --min-area-m2 10 \
  --merge-missing true \
  --allow-empty true \
  --out-summary ../results/dev_synthetic/res-korea/vector/vectorization_summary.json \
  2>&1 | tee ../results/logs/vectorize_bcenet_candidates_synthetic.log

python scripts/create_controlled_candidate_prob_maps.py \
  --tile-index ../results/dev_synthetic/korea_poc/metadata/tile_index.geojson \
  --old-buildings ../results/dev_synthetic/raw/synthetic_old_buildings.gpkg \
  --old-id-dir ../results/dev_synthetic/korea_poc/tiles/old_id \
  --reference-changes ../results/dev_synthetic/raw/synthetic_reference_changes.gpkg \
  --out-dir ../results/dev_synthetic/res-korea-controlled \
  --id-col BLDG_ID \
  2>&1 | tee ../results/logs/create_controlled_candidate_prob_maps.log

python scripts/vectorize_bcenet_candidates.py \
  --tile-index ../results/dev_synthetic/korea_poc/metadata/tile_index.geojson \
  --inference-dir ../results/dev_synthetic/res-korea-controlled \
  --old-buildings ../results/dev_synthetic/raw/synthetic_old_buildings.gpkg \
  --old-id-dir ../results/dev_synthetic/korea_poc/tiles/old_id \
  --out ../results/dev_synthetic/res-korea-controlled/vector/bcenet_building_candidates_controlled.gpkg \
  --threshold-new 0.5 \
  --threshold-removed 0.5 \
  --threshold-stat p90 \
  --min-area-m2 10 \
  --merge-missing true \
  --allow-empty false \
  --out-summary ../results/dev_synthetic/res-korea-controlled/vector/vectorization_summary_controlled.json \
  2>&1 | tee ../results/logs/vectorize_bcenet_candidates_controlled.log

python scripts/check_candidate_vector_outputs.py \
  --gpkg ../results/dev_synthetic/res-korea/vector/bcenet_building_candidates.gpkg \
  --summary-json ../results/dev_synthetic/res-korea/vector/vectorization_summary.json \
  --out-json ../results/dev_synthetic/res-korea/vector/vector_output_check.json \
  --require-non-empty false \
  2>&1 | tee ../results/logs/check_candidate_vector_outputs_model.log

python scripts/check_candidate_vector_outputs.py \
  --gpkg ../results/dev_synthetic/res-korea-controlled/vector/bcenet_building_candidates_controlled.gpkg \
  --summary-json ../results/dev_synthetic/res-korea-controlled/vector/vectorization_summary_controlled.json \
  --out-json ../results/dev_synthetic/res-korea-controlled/vector/vector_output_check_controlled.json \
  --require-non-empty true \
  2>&1 | tee ../results/logs/check_candidate_vector_outputs_controlled.log

python scripts/make_candidate_vector_preview.py \
  --ortho ../results/dev_synthetic/raw/synthetic_current_ortho.tif \
  --old-buildings ../results/dev_synthetic/raw/synthetic_old_buildings.gpkg \
  --candidates ../results/dev_synthetic/res-korea-controlled/vector/bcenet_building_candidates_controlled.gpkg \
  --out-dir ../results/dev_synthetic/res-korea-controlled/vector_preview \
  --max-previews 5 \
  2>&1 | tee ../results/logs/make_candidate_vector_preview.log
```

Model-output vectorization:
- Input inference dir: `../results/dev_synthetic/res-korea`
- Old ID rasters: `../results/dev_synthetic/korea_poc/tiles/old_id`
- Output GeoPackage: `../results/dev_synthetic/res-korea/vector/bcenet_building_candidates.gpkg`
- Summary: `../results/dev_synthetic/res-korea/vector/vectorization_summary.json`
- Check JSON: `../results/dev_synthetic/res-korea/vector/vector_output_check.json`
- Layers:
  - `building_missing_candidates`
  - `building_excess_candidates`
- Missing candidates: `2`
- Excess candidates: `0`
- Total candidates: `2`
- Empty layer handling:
  - `building_excess_candidates` is an empty but schema-valid layer.
- Model-output candidate count is not interpreted as model performance.

Controlled probability smoke:
- Output dir: `../results/dev_synthetic/res-korea-controlled`
- Summary: `../results/dev_synthetic/res-korea-controlled/controlled_probability_summary.json`
- Controlled missing references: `2`
- Controlled excess old building IDs: `103`, `104`
- Probability values:
  - candidate region: `0.95`
  - background: `0.05`
  - old building reference: `0.80`

Controlled vectorization:
- Output GeoPackage: `../results/dev_synthetic/res-korea-controlled/vector/bcenet_building_candidates_controlled.gpkg`
- Summary: `../results/dev_synthetic/res-korea-controlled/vector/vectorization_summary_controlled.json`
- Check JSON: `../results/dev_synthetic/res-korea-controlled/vector/vector_output_check_controlled.json`
- Missing candidates: `2`
- Excess candidates: `2`
- Total candidates: `4`
- Both candidate layers are non-empty.

Candidate schema:
- Layers:
  - `building_missing_candidates`
  - `building_excess_candidates`
- Common fields:
  - `candidate_id`
  - `candidate_type`
  - `model_output`
  - `confidence_mean`
  - `confidence_p90`
  - `confidence_max`
  - `area_m2`
  - `source_tile`
  - `review_status`
  - `review_comment`
  - `is_synthetic`
- Excess-only field:
  - `old_building_id`
- Allowed `candidate_type` values used:
  - `MISSING`
  - `EXCESS`
- `review_status`: `UNREVIEWED`
- CRS: `EPSG:5186`
- Geometry validity: passed for model-output and controlled candidate layers.
- Results are candidate layers, not confirmed errors.

Preview:
- Summary: `../results/dev_synthetic/res-korea-controlled/vector_preview/candidate_vector_preview_summary.json`
- Overview PNG: `../results/dev_synthetic/res-korea-controlled/vector_preview/candidate_vector_preview_overview.png`
- Detail preview PNGs:
  - `candidate_preview_01_MISSING.png`
  - `candidate_preview_02_MISSING.png`
  - `candidate_preview_03_EXCESS.png`
  - `candidate_preview_04_EXCESS.png`
- Preview is a smoke-test quicklook, not a reviewer UI.

Notes:
- NumPy/rasterio emitted deprecation warnings while reading arrays under NumPy `2.5.0`.
- These warnings did not fail vectorization, schema checks, or preview generation.
- A first model-output vectorization attempt exposed an empty excess-layer handling bug; the script was minimally patched so empty candidate layers retain schema and can be written.

Failure classification:
- None for final Stage 8V pass.

Stage 8V result: pass
- Synthetic BCE-Net model outputs were vectorized into candidate GeoPackage layers.
- Controlled probability smoke test produced non-empty missing/excess candidate layers.
- Candidate layers use `MISSING` and `EXCESS` types only.
- All candidates are marked as `UNREVIEWED`.
- Results are object-level candidates, not confirmed errors.

Next stage:
- Stage 9: real data upload and inference preparation, when real ortho/image and existing building vector are available.

## Managed-Container Bootstrap Automation

Date: 2026-07-15

Session model:
- A new cloud session may create a fresh provider-managed container.
- The user cannot choose the packages installed while that container is created.
- BCE-Net source may enter the container through either `git clone` or a volume bind mount.
- The goal is one repeatable post-create command, not a custom base image.

Bootstrap entry points:

```bash
make setup
make verify
./scripts/run_in_env.sh python test_model_korea.py --help
```

Implementation:
- `scripts/setup_env.sh`: prerequisite checks, venv creation, pinned wheel installation, GPU-architecture detection, DCNv2 build, and verification.
- `scripts/verify_env.py`: package imports, CUDA visibility, DCNv2 CUDA forward, geospatial smoke, and checkpoint-path validation.
- `scripts/run_in_env.sh`: command execution without relying on interactive shell activation.
- `requirements-managed.txt`: managed-container geospatial package pins.
- `Makefile`: human-facing `setup`, `verify`, `env`, and `shell` targets.
- `.vscode/settings.json`: repository venv selection for VS Code.
- `ENVIRONMENT.md`: clone, volume binding, persistence, override, and troubleshooting boundaries.

Base-container prerequisites:
- Python 3.12-compatible runtime.
- CUDA-enabled PyTorch and torchvision.
- Visible NVIDIA GPU.
- CUDA toolkit with `nvcc`.
- `g++` and Python development headers.
- Initial pip wheel download access.

The bootstrap intentionally does not attempt to install or configure a host
NVIDIA driver, container GPU runtime, or missing system CUDA toolkit. Those are
properties of the provider-managed base container.

Environment created and verified:
- Venv: `/home/work/BCE-Net/.venv-bcenet-geo`
- Python: `3.12.3`
- PyTorch: `2.10.0a0+b558c986e8.nv25.11`
- CUDA: `13.0`
- GPU: NVIDIA H200, compute capability `9.0`
- DCNv2 extension: `_ext.cpython-312-x86_64-linux-gnu.so`, built for `sm_90`
- NumPy: `2.2.6`
- rasterio: `1.5.0`
- geopandas: `1.1.3`
- pyogrio: `0.12.1`

Verification results:
- `make setup`: pass
- Idempotent second `make setup`: pass; compatible DCNv2 build skipped
- `make verify`: pass
- DCNv2 16x16 CUDA forward: pass
- GeoTIFF and GeoPackage read/write: pass
- Rasterize and polygonize: pass
- WHU checkpoint strict load: pass
- BCE-Net 512x512 CUDA forward: pass
- BCE-Net forward time observed: approximately `201 ms`
- BCE-Net output finiteness: all checked outputs contained no NaN/Inf

Persistence behavior:
- Fresh clone/fresh volume: run `make setup`; packages and DCNv2 are rebuilt.
- Persistent bind-mounted repository: venv and extension may be reused.
- New terminal in the same container: no reinstall; use the runner or VS Code interpreter.
- Changed base Python/PyTorch/CUDA ABI: rerun `make setup`; incompatible DCNv2 is force-rebuilt.

Checkpoint path:
- Default: `/home/work/models/BCE-Net/checkpoint-best-whu.pth`
- Override: `BCENET_WEIGHTS=/mounted/path/checkpoint.pth make setup`

Generated environment state is excluded from Git:
- `.venv-bcenet-geo/`
- `.setup-logs/`
- `DCNv2/_ext*.so`
- `DCNv2/build/`
- weights, data, and results

Bootstrap result: pass
- A fresh compatible cloud container can be configured with one repository command.
- A provider image without CUDA-enabled PyTorch, `nvcc`, or a C++ compiler fails early with a prerequisite error.
