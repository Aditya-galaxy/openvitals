#!/bin/bash
# Run ON the DREAMT VM, by the data-use-agreement signer:  bash dreamt_fetch.sh
#
# PhysioNet serves restricted files to wget (its documented tool) but refuses curl-type
# clients, and throttles each connection, so this runs many wget downloads in parallel
# (one file each) instead of one multi-connection ZIP download.
#
#   Phase 1: all of data_64Hz/   (100 x ~141 MB = 14.1 GB): wrist E4 signals + sleep stages
#   Phase 2: first N100 subjects of data_100Hz/ (~1.1 GB each): adds ECG, for HRV checks
#
# Your login goes into ~/.wgetrc-physionet (mode 600, readable only by you) because the
# download runs for days; it is deleted when the download finishes. Re-running resumes:
# finished files are verified against SHA256SUMS.txt and skipped.
#
# DREAMT is restricted data (PhysioNet Restricted Health Data Use Agreement 1.5.0):
# it stays on this VM; don't copy it elsewhere or share access.
set -uo pipefail
BASE=https://physionet.org/files/dreamt/2.2.0
OUT="$HOME/dreamt"
CFG="$HOME/.wgetrc-physionet"
JOBS=${JOBS:-12}
N100=${N100:-20}
mkdir -p "$OUT" && cd "$OUT" || exit 1
command -v tmux >/dev/null || { sudo apt-get -qq update && sudo apt-get -qq install -y tmux >/dev/null 2>&1; }

if [ ! -s "$CFG" ]; then
  read -r -p "PhysioNet username: " U
  read -r -s -p "PhysioNet password (not shown): " P; echo
  (umask 077; printf 'user = %s\npassword = %s\n' "$U" "$P" > "$CFG")
  unset P
fi

echo "== checking access"
if ! wget -q --config="$CFG" -O SHA256SUMS.txt "$BASE/SHA256SUMS.txt"; then
  echo "access check failed: wrong username/password, or no DREAMT access on this account."
  rm -f "$CFG"; exit 1
fi
echo "ok: $(wc -l < SHA256SUMS.txt) files listed"

# queue: all data_64Hz, then the first N100 subjects' data_100Hz
{ grep ' data_64Hz/' SHA256SUMS.txt | sort -k2
  grep ' data_100Hz/' SHA256SUMS.txt | sort -k2 | head -n "$N100"
  grep -E ' (participant_info.csv|Metadata.txt)$' SHA256SUMS.txt; } > queue.txt
echo "queue: $(wc -l < queue.txt) files ($JOBS in parallel)"

cat > fetch_one.sh <<'ONE'
#!/bin/bash
# fetch_one.sh <sha256> <path>: download one file with wget (resuming), verify, log.
sum="$1"; rel="$2"; out="$HOME/dreamt/$rel"
mkdir -p "$(dirname "$out")"
if [ -f "$out" ] && echo "$sum  $out" | sha256sum -c --status; then exit 0; fi
for try in 1 2 3 4 5 6; do
  if wget -q -c --config="$HOME/.wgetrc-physionet" --timeout=120 --tries=3 -O "$out" \
    "https://physionet.org/files/dreamt/2.2.0/$rel"; then
    if echo "$sum  $out" | sha256sum -c --status; then
      echo "$(date -u +%FT%TZ) ok $rel $(( $(stat -c %s "$out") / 1000000 )) MB" >> "$HOME/dreamt/progress.log"; exit 0
    fi
    rm -f "$out"  # complete but corrupt (e.g. after a crash): start over, don't resume onto it
  fi
  sleep $(( try * 30 ))
done
echo "$(date -u +%FT%TZ) FAILED $rel" >> "$HOME/dreamt/progress.log"
ONE
chmod +x fetch_one.sh

tmux kill-session -t dl 2>/dev/null
tmux new -d -s dl "cd '$OUT' && awk '{print \$1, \$2}' queue.txt | xargs -P $JOBS -n 2 ./fetch_one.sh; \
  echo \"\$(date -u +%FT%TZ) ALL_DONE\" >> progress.log; rm -f '$CFG'"
echo "started in background (tmux session 'dl'). You can close this window."
echo "progress: grep -c ' ok ' ~/dreamt/progress.log   (of $(wc -l < queue.txt))"
