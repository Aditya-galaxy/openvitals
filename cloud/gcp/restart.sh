#!/bin/bash
# Runs ON the VM: stop the current run, install the newest wheel, start run.sh again.
# A script, not an ssh one-liner: "pkill -f <pattern>" also matches the ssh session's
# own command line when the pattern appears in it, killing the session itself.
# Usage: BUCKET=gs://... MAX_NIGHTS=1 WORKERS=8 bash restart.sh
set -uo pipefail
cd "$HOME/ov"
export PATH="$HOME/.local/bin:$PATH"
tmux kill-session -t run 2>/dev/null
pkill -f "bin/openvitals featurize"; pkill -f "bin/openvitals bench"; pkill -f "gsutil -q -m rsync"
sleep 3
echo "old processes left: $(pgrep -fc 'bin/openvitals (featurize|bench)')"
uv pip install -q --python .venv/bin/python --reinstall-package openvitals "$(ls -t openvitals-*.whl | head -1)[seq]" || exit 1
.venv/bin/python -c "import openvitals.featstore as f; assert hasattr(f, 'clear_claims'); print('package ok')" || exit 1
tmux new -d -s run "BUCKET=$BUCKET MAX_NIGHTS=${MAX_NIGHTS:-1} WORKERS=${WORKERS:-8} bash run.sh"
sleep 60
echo "featurize processes: $(pgrep -fc 'bin/openvitals featurize')"
echo "claims: $(ls feat/bidsleep/.claims 2>/dev/null | tr '\n' ' ')"
echo "done: $(ls feat/bidsleep/*.pkl 2>/dev/null | wc -l)"
tail -1 logs/run.log
