#!/usr/bin/env bash
# Create a conda environment and install the dependencies needed by this baseline.
#
# Usage:
#   bash install.sh                                  # default env "nnunet", Python 3.10
#   ENV_NAME=myenv PYTHON_VERSION=3.11 bash install.sh
#   TORCH_CUDA=cu118 bash install.sh                 # change PyTorch CUDA wheel
#   ANACONDA_BASE=/opt/anaconda3 bash install.sh     # override Anaconda root
#
# Notes:
#   - The env is created at ${ANACONDA_BASE}/envs/${ENV_NAME} to avoid the
#     pitfall where a custom `envs_dirs` in ~/.condarc places envs elsewhere.
#   - PyTorch is installed from the official PyTorch wheel index so the CUDA
#     build matches your driver.
#   - nnUNetv2 is installed via pip.

set -euo pipefail

ENV_NAME="${ENV_NAME:-nnunet}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"   # one of: cu118 / cu121 / cu124 / cpu
ANACONDA_BASE="${ANACONDA_BASE:-/usr/data/env/anaconda3}"
ENV_PATH="${ANACONDA_BASE}/envs/${ENV_NAME}"

# --- 0. Sanity checks ---
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi
if [[ ! -d "${ANACONDA_BASE}" ]]; then
    echo "ERROR: Anaconda root '${ANACONDA_BASE}' does not exist. Override with ANACONDA_BASE=..."
    exit 1
fi

# --- 1. Create the environment ---
if [[ -d "${ENV_PATH}" ]]; then
    echo "Env already exists at ${ENV_PATH} (skipping creation)."
    echo "To rebuild: conda env remove -p ${ENV_PATH}"
else
    echo "Creating conda env: ${ENV_PATH} (Python ${PYTHON_VERSION})"
    conda create -p "${ENV_PATH}" "python=${PYTHON_VERSION}" -y
fi

# --- 2. Activate it ---
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"
echo "Activated $(python --version) at $(which python)"

# --- 3. PyTorch ---
if [[ "${TORCH_CUDA}" == "cpu" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
    TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
fi
echo "Installing PyTorch (${TORCH_CUDA}) from ${TORCH_INDEX}"
pip install --upgrade pip
pip install torch torchvision --index-url "${TORCH_INDEX}"

# --- 4. Training & preprocessing deps ---
echo "Installing nnUNetv2 and helpers"
pip install \
    nnunetv2 \
    SimpleITK \
    numpy \
    tqdm \
    scipy

# --- 5. Self-check ---
python - <<'PY'
import torch, nnunetv2, SimpleITK as sitk
print(f"torch     {torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"nnunetv2  {getattr(nnunetv2, '__version__', 'unknown')}")
print(f"SimpleITK {sitk.Version_VersionString()}")
PY

cat <<EOF

Install complete.

Next: add the following to your ~/.bashrc or ~/.zshrc (edit paths as needed):

    export nnUNet_raw=/path/to/nnUNet_raw
    export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
    export nnUNet_results=/path/to/nnUNet_results

Every session:

    conda activate ${ENV_NAME}

EOF
