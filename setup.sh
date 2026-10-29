#!/usr/bin/env bash
# One-time server setup: Python 3.11 venv with CUDA 12.4 torch, then the CPU test suite.
set -eu
cd "$(dirname "$0")"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -x .venv/bin/python ] || uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
.venv/bin/python -c "import torch,timm;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
MASKEDKD_FAKE_DATA=1 .venv/bin/python -m pytest -q tests
