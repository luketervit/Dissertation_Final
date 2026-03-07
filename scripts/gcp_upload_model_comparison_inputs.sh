#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <instance-name> <zone> [remote-project-dir]"
  exit 1
fi

INSTANCE="$1"
ZONE="$2"
REMOTE_DIR="${3:-~/Dissertation_Final}"

echo "Uploading model-comparison inputs to ${INSTANCE}:${REMOTE_DIR} (${ZONE})"

gcloud compute scp --recurse \
  batch_simulations_reconstructed \
  sim \
  scripts \
  config \
  requirements.txt \
  "${INSTANCE}:${REMOTE_DIR}" \
  --zone="${ZONE}"

echo "Upload complete."
