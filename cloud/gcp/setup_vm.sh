#!/bin/bash
# Runs ON the VM once: uv, CPU-only torch, the openvitals wheel copied alongside.
set -euo pipefail
cd "$HOME/ov"
command -v tmux >/dev/null || (sudo apt-get -qq update && sudo apt-get -qq install -y tmux >/dev/null)
[ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH="$HOME/.local/bin:$PATH"
uv venv -q --python 3.11 .venv
# CPU wheels: the default PyPI torch on Linux pulls ~3 GB of CUDA libraries we don't need.
uv pip install -q --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -q --python .venv/bin/python --reinstall-package openvitals "$(ls openvitals-*.whl)[seq]"
.venv/bin/openvitals featurize --help | grep -q max-nights && echo "openvitals ready: $(.venv/bin/python -c 'import torch;print("torch",torch.__version__)')"
