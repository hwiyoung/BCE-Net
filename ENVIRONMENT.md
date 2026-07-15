# BCE-Net Environment

The managed GPU container provides Python, PyTorch, CUDA, `nvcc`, and the GPU
driver. The repository-local `.venv-bcenet-geo` adds the compatible geospatial
stack, while `DCNv2/_ext*.so` is built for the current PyTorch/CUDA ABI.

## Session model

In this project, a new cloud session may mean a completely new container. The
cloud provider chooses the base image, and the repository is then made visible
through either `git clone` or a volume bind mount. The bootstrap does not
control container creation; it configures the supplied container after the
source directory is available.

The supplied base container must already provide:

- a visible NVIDIA GPU and compatible driver;
- Python with CUDA-enabled PyTorch and torchvision;
- the CUDA toolkit including `nvcc`;
- `g++` and Python development headers; and
- network access for the initial wheel download.

`setup_env.sh` checks these prerequisites before installing repository-local
packages. It cannot install a host GPU driver or repair a cloud image that does
not expose CUDA or `nvcc`.

## First setup or repair in a container

```bash
make setup
```

The command is idempotent. It reuses an existing venv, skips the DCNv2 build
when the extension imports successfully, and finishes with GPU/geospatial smoke
checks. Logs are written under `.setup-logs/`.

`make setup` is the convenience entry point. If the base image does not include
`make`, call the underlying script directly:

```bash
./scripts/setup_env.sh
```

The DCNv2 build targets the attached GPU architecture only. Set
`BCENET_CUDA_ARCH_LIST` explicitly if a portable multi-architecture binary is
required.

## Clone and volume-binding workflows

Both source-delivery methods use the same command:

```bash
cd /path/to/BCE-Net
make setup
```

- Fresh clone or fresh volume: the venv and DCNv2 extension are created again.
- Persistent bind-mounted repository: `.venv-bcenet-geo` and the compiled
  extension remain on the volume and can be reused if the base ABI is still
  compatible.
- Changed Python/PyTorch/CUDA base image: rerun `make setup`; the import check
  forces a DCNv2 rebuild when the old extension is incompatible.

The default checkpoint path is
`/home/work/models/BCE-Net/checkpoint-best-whu.pth`. Override it when the model
volume is mounted elsewhere:

```bash
BCENET_WEIGHTS=/models/checkpoint-best-whu.pth make setup
```

## New terminal in the same container

VS Code automatically selects `.venv-bcenet-geo/bin/python`. For commands that
must work independently of shell activation, use the wrapper:

```bash
./scripts/run_in_env.sh python test_model_korea.py --help
make verify
```

Interactive activation is optional:

```bash
source .venv-bcenet-geo/bin/activate
```

As long as the container or bound workspace persists, a new terminal does not
need another installation. A completely new container with a fresh filesystem
must run `make setup` again. This replaces manual installation with one
repeatable command; it does not make an ephemeral container retain packages.

## What persists

- Persists with the workspace: `.venv-bcenet-geo`, DCNv2 `.so`, source, data,
  and results.
- Resets in a new shell: the temporary `source .../activate` state.
- May change with a new managed-container image: system PyTorch, CUDA, compiler,
  and system Python packages. `make setup` detects an incompatible DCNv2 import
  and rebuilds it.

Generated venvs, compiled extensions, logs, weights, data, and results are
ignored by Git. Commit the bootstrap files, not the generated environment.
