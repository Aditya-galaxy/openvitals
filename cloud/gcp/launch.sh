#!/bin/bash
# Runs LOCALLY: bucket + VM, copy the wheel, Walch features and scripts, install.
# Starting the run itself is a separate step (see README in this folder).
set -euo pipefail
PROJECT=${PROJECT:-kronagent}; ZONE=${ZONE:-us-east1-b}; VM=${VM:-openvitals-run}
MACHINE=${MACHINE:-e2-standard-8}; BUCKET=${BUCKET:-gs://kronagent-openvitals}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); TMP=$(mktemp -d)
gcloud storage buckets describe "$BUCKET" --project "$PROJECT" >/dev/null 2>&1 || \
  gcloud storage buckets create "$BUCKET" --project "$PROJECT" --location us-east1 --uniform-bucket-level-access
gcloud compute instances describe "$VM" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1 || \
  gcloud compute instances create "$VM" --project "$PROJECT" --zone "$ZONE" --machine-type "$MACHINE" \
    --image-family debian-12 --image-project debian-cloud --boot-disk-size 60GB --boot-disk-type pd-balanced \
    --scopes storage-rw --labels app=openvitals
(cd "$ROOT" && uv build -q --wheel --out-dir "$TMP")
# No macOS metadata: otherwise tar adds '._*' AppleDouble stubs next to every file.
COPYFILE_DISABLE=1 tar --no-xattrs -C "$ROOT" -czf "$TMP/feat-walch.tgz" feat/walch
for i in 1 2 3 4 5 6; do gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --quiet --command "mkdir -p ov" && break; sleep 10; done
gcloud compute scp --zone "$ZONE" --project "$PROJECT" --quiet "$TMP"/openvitals-*.whl "$TMP/feat-walch.tgz" \
  "$ROOT/cloud/gcp/run.sh" "$ROOT/cloud/gcp/setup_vm.sh" "$VM:ov/"
gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --quiet --command \
  "cd ov && tar xzf feat-walch.tgz && bash setup_vm.sh"
rm -rf "$TMP"
