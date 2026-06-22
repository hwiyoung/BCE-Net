# BCE-Net PoC Docker Environment

This directory defines the development container for the BCE-Net Korea PoC.
Run commands from the BCE-Net repository root.

## Prerequisites

The host must have:
- Docker Engine and Docker Compose v2
- NVIDIA driver
- NVIDIA Container Toolkit configured for Docker

Stage 1 currently records Docker and `nvidia-ctk` as unavailable on this host.
Do not run container development until those host prerequisites are satisfied.

## Build

```bash
docker compose build bcenet
```

## Start A Shell

```bash
docker compose run --rm bcenet bash
```

Inside the container, the repository is mounted at `/workspace/BCE-Net`.

## Mounts

| Host path | Container path | Mode | Purpose |
| --- | --- | --- | --- |
| `.` | `/workspace/BCE-Net` | read/write | BCE-Net source |
| `../models/BCE-Net` | `/workspace/models/BCE-Net` | read-only | pretrained weights |
| `../data` | `/workspace/data` | read/write | input geospatial data |
| `../results` | `/workspace/results` | read/write | development and inference outputs |

`/workspace/data` may be empty at this stage.

## Docker Run Fallback

Use this if `docker compose` does not support `gpus: all` in the local environment:

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

## Later Container-Only Work

All Python execution, DCNv2 build, model checks, and smoke tests must be run
inside this container after the host prerequisites are fixed.
