#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${BCENET_VENV:-$REPO_ROOT/.venv-bcenet-geo}"
PYTHON_BIN="${BCENET_BOOTSTRAP_PYTHON:-python}"
REQUIREMENTS="$REPO_ROOT/requirements-managed.txt"
LOG_DIR="$REPO_ROOT/.setup-logs"
WEIGHTS_PATH="${BCENET_WEIGHTS:-/home/work/models/BCE-Net/checkpoint-best-whu.pth}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MAX_JOBS="${MAX_JOBS:-4}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/DCNv2${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOG_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "[0/5] Checking cloud-container prerequisites"
for tool in g++ nvcc; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: Required base-container tool is missing: $tool" >&2
    echo "The cloud image must provide Python, CUDA/nvcc, a C++ compiler, and CUDA-enabled PyTorch." >&2
    exit 1
  fi
done

if ! "$PYTHON_BIN" - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

if not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled PyTorch cannot access a GPU")
if CUDA_HOME is None:
    raise SystemExit("PyTorch cannot locate the CUDA toolkit")
print(f"      torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY
then
  echo "ERROR: The cloud base image does not satisfy BCE-Net GPU prerequisites." >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[1/5] Creating venv: $VENV_DIR"
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
else
  echo "[1/5] Reusing venv: $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

echo "[2/5] Installing pinned geospatial packages"
"$VENV_PYTHON" -m pip install \
  "pip==25.3" "setuptools==80.9.0" "wheel==0.45.1" \
  >"$LOG_DIR/pip-bootstrap.log" 2>&1
"$VENV_PYTHON" -m pip install --only-binary=:all: -r "$REQUIREMENTS" \
  >"$LOG_DIR/pip-managed.log" 2>&1

if [[ -n "${BCENET_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST="$BCENET_CUDA_ARCH_LIST"
else
  # Managed images often export a broad multi-GPU architecture list. Building
  # all of it is slow and unnecessary, so default to the attached GPU only.
  TORCH_CUDA_ARCH_LIST="$($VENV_PYTHON - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; cannot build DCNv2")
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"
fi
export TORCH_CUDA_ARCH_LIST

echo "[3/5] Checking DCNv2 extension (CUDA arch $TORCH_CUDA_ARCH_LIST)"
if ! "$VENV_PYTHON" -c "from DCNv2.dcn_v2 import DCN" >/dev/null 2>&1; then
  echo "      Building DCNv2 for the current PyTorch/CUDA runtime"
  (
    cd "$REPO_ROOT/DCNv2"
    "$VENV_PYTHON" setup.py build_ext --inplace --force
  ) >"$LOG_DIR/dcnv2-build.log" 2>&1 || {
    echo "ERROR: DCNv2 build failed. See $LOG_DIR/dcnv2-build.log" >&2
    tail -80 "$LOG_DIR/dcnv2-build.log" >&2
    exit 1
  }
else
  echo "      Existing DCNv2 extension is compatible; build skipped"
fi

echo "[4/5] Verifying imports, GPU, DCNv2, and geospatial stack"
"$VENV_PYTHON" "$REPO_ROOT/scripts/verify_env.py"

cat >"$VENV_DIR/.bcenet-environment" <<EOF
repo=$REPO_ROOT
python=$($VENV_PYTHON -V 2>&1)
torch=$($VENV_PYTHON -c 'import torch; print(torch.__version__)')
torch_cuda=$($VENV_PYTHON -c 'import torch; print(torch.version.cuda)')
cuda_arch=$TORCH_CUDA_ARCH_LIST
weights=$WEIGHTS_PATH
EOF

echo "[5/5] Environment ready"
echo "Run commands without activation:"
echo "  ./scripts/run_in_env.sh python test_model_korea.py --help"
echo "Or activate it in this shell:"
echo "  source .venv-bcenet-geo/bin/activate"
