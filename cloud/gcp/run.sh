#!/bin/bash
# Runs ON the VM: featurize BIDSleep (streamed, resumable), then 5-seed benchmarks.
# Every phase is resumable: re-running skips finished subjects and finished reports.
# Usage: BUCKET=gs://... MAX_NIGHTS=1 SEEDS="0 1 2 3 4" bash run.sh
# Optional: MODELS=a,b  SKIP_FEATURIZE=1  JOBS=n  POWEROFF=0
set -uo pipefail
cd "$HOME/ov"
: "${BUCKET:?set BUCKET}" "${MAX_NIGHTS:=1}" "${SEEDS:=0 1 2 3 4}" "${JOBS:=$(nproc)}" "${POWEROFF:=1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OV=.venv/bin/openvitals
mkdir -p logs results/parts
log() { echo "$(date -u +%FT%TZ) $*" | tee -a logs/run.log; }

# Sync features/results/logs to the bucket every 10 min, so a lost VM loses at most 10 min.
( while sleep 600; do gsutil -q -m rsync -r feat "$BUCKET/feat"; gsutil -q -m rsync -r results "$BUCKET/results"; gsutil -q -m rsync -r logs "$BUCKET/logs"; done ) &
SYNC=$!
sync_now() { gsutil -q -m rsync -r feat "$BUCKET/feat"; gsutil -q -m rsync -r results "$BUCKET/results"; gsutil -q -m rsync -r logs "$BUCKET/logs"; }

if [ "${SKIP_FEATURIZE:-0}" = 1 ]; then rc=0; else
  log "featurize bidsleep (max_nights=$MAX_NIGHTS)"
  $OV featurize bidsleep --data-dir data --features feat --download --evict --workers 4 --max-nights "$MAX_NIGHTS" >> logs/featurize.log 2>&1
  rc=$?; log "featurize exit $rc"; sync_now
fi
[ $rc -eq 0 ] || { log "featurize failed; not benchmarking"; kill $SYNC; exit $rc; }

W=walch=features:feat; B=bidsleep=features:feat
ALL=${MODELS:-gbdt_hr,gbdt_motion_hr,tcn_hr,tcn_motion_hr,gru_hr,gru_motion_hr}
: > logs/jobs.txt
for s in $SEEDS; do
  echo "bidsleep|kfold-4class|$B --split kfold|$s" >> logs/jobs.txt
  echo "cross|bidsleep_to_walch-4class|$B --test $W --split cross|$s" >> logs/jobs.txt
  echo "cross|walch_to_bidsleep-4class|$W --test $B --split cross|$s" >> logs/jobs.txt
done
log "benchmarks: $(wc -l < logs/jobs.txt) jobs on $JOBS cores"
export OV ALL
# xargs passes only a line number (long substituted commands break BSD xargs).
seq 1 "$(wc -l < logs/jobs.txt)" | xargs -P "$JOBS" -n 1 bash -c '
  IFS="|" read -r group stem args seed <<< "$(sed -n "${0}p" logs/jobs.txt)"
  out=results/parts/$group; [ -f "$out/$stem-s$seed.json" ] && exit 0
  $OV bench $args --scheme 4class --models $ALL --seeds $seed --tag s$seed --out $out \
    > logs/bench-$group-$stem-s$seed.log 2>&1 && echo "done $stem s$seed" || echo "FAILED $stem s$seed"
' | tee -a logs/run.log

log "merging"
for pair in "bidsleep kfold-4class gbdt_motion_hr" "cross bidsleep_to_walch-4class gbdt_motion_hr" "cross walch_to_bidsleep-4class gbdt_motion_hr"; do
  set -- $pair
  $OV merge results/parts/$1/$2-s*.json --reference $3 --out results/v04/$1 >> logs/merge.log 2>&1
  $OV merge results/parts/$1/$2-s*.json --reference tcn_hr --tag vs_tcn_hr --out results/v04/$1 >> logs/merge.log 2>&1
done
log "all done"; kill $SYNC; sync_now
touch logs/ALL_DONE; gsutil -q cp logs/ALL_DONE "$BUCKET/logs/ALL_DONE"
[ "$POWEROFF" = 1 ] && sudo poweroff
