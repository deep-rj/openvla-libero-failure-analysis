#!/bin/bash
# Copies outputs/ to $OUTPUTS_ARCHIVE_DIR/<pod id>/ (e.g. a RunPod global volume)
# so raw traces survive pod teardown. No-op when OUTPUTS_ARCHIVE_DIR is unset.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${OUTPUTS_ARCHIVE_DIR:-}" ]; then
  echo "OUTPUTS_ARCHIVE_DIR not set; skipping archive."
  exit 0
fi
# Not created here: a missing root means the volume isn't mounted.
if [ ! -d "$OUTPUTS_ARCHIVE_DIR" ]; then
  echo "ERROR: $OUTPUTS_ARCHIVE_DIR not found; outputs not archived." >&2
  exit 1
fi

dest="$OUTPUTS_ARCHIVE_DIR/${RUNPOD_POD_ID:-$(hostname)}"
mkdir -p "$dest"
# Plain cp -ru: object-storage mounts reject the chmod/chown/utime that rsync -a or cp -a issue.
cp -ru "$REPO_DIR/outputs/." "$dest/"
echo "Archived outputs/ to $dest"
