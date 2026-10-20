#!/bin/bash
# Copies a run directory (default: outputs/) to an immutable snapshot
# $OUTPUTS_ARCHIVE_DIR/<UTC time>-<git sha>-<pod id>/ (e.g. a network or object-storage volume) so
# raw traces survive pod teardown, then writes and verifies MANIFEST.sha256 beside it.
# No-op when OUTPUTS_ARCHIVE_DIR is unset.
#   archive_outputs.sh [run_dir]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="${1:-$REPO_DIR/outputs}"

if [ -z "${OUTPUTS_ARCHIVE_DIR:-}" ]; then
  echo "OUTPUTS_ARCHIVE_DIR not set; skipping archive."
  exit 0
fi
# Not created here: a missing root means the volume isn't mounted.
if [ ! -d "$OUTPUTS_ARCHIVE_DIR" ]; then
  echo "ERROR: $OUTPUTS_ARCHIVE_DIR not found; outputs not archived." >&2
  exit 1
fi
if [ ! -d "$src" ]; then
  echo "ERROR: $src is not a directory." >&2
  exit 1
fi

sha="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo nogit)"
dest="$OUTPUTS_ARCHIVE_DIR/$(date -u +%Y%m%dT%H%M%SZ)-$sha-${RUNPOD_POD_ID:-$(hostname)}"
# No -p: an existing snapshot is never overwritten.
mkdir "$dest"
# Plain cp: object-storage mounts reject the chmod/chown/utime that rsync -a or cp -a issue.
cp -r "$src/." "$dest/"
(
  cd "$dest"
  find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
  sha256sum --quiet -c MANIFEST.sha256
)
echo "Archived $src to $dest"
