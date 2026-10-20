#!/bin/bash
# Pulls archived run directories from a remote host to outputs/<run>/ with rsync over SSH.
# The remote root is the directory archive_outputs.sh wrote to ($OUTPUTS_ARCHIVE_DIR there).
#
#   pull_outputs.sh -H <host> -p <port> -r <remote_root> --list
#   pull_outputs.sh -H <host> -p <port> -r <remote_root> <run> [--only GLOB]... [--no-video] [--dry-run]
#
# Host, port, remote root and user also come from PULL_HOST, PULL_PORT, REMOTE_ROOT, PULL_USER.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host="${PULL_HOST:-}"
port="${PULL_PORT:-22}"
remote_root="${REMOTE_ROOT:-}"
user="${PULL_USER:-root}"
list=0
dry_run=0
no_video=0
only=()
run=""

usage() { sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    -H|--host) host="${2:?missing value}"; shift 2 ;;
    -p|--port) port="${2:?missing value}"; shift 2 ;;
    -r|--remote-root) remote_root="${2:?missing value}"; shift 2 ;;
    --list) list=1; shift ;;
    --only) only+=("${2:?missing value}"); shift 2 ;;
    --no-video) no_video=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage ;;
    -*) echo "Unknown option: $1" >&2; usage ;;
    *) [ -z "$run" ] || usage; run="$1"; shift ;;
  esac
done

[ -n "$host" ] || { echo "ERROR: pass -H <host> or set PULL_HOST." >&2; usage; }
[ -n "$remote_root" ] || { echo "ERROR: pass -r <remote_root> or set REMOTE_ROOT." >&2; usage; }

ssh_cmd=(ssh -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
target="$user@$host"

if ! "${ssh_cmd[@]}" "$target" "test -d '$remote_root'"; then
  echo "ERROR: $remote_root not found on $host; is the archive volume mounted?" >&2
  exit 1
fi

if [ "$list" -eq 1 ]; then
  "${ssh_cmd[@]}" "$target" "cd '$remote_root' && du -sh -- */ 2>/dev/null"
  exit 0
fi

[ -n "$run" ] || { echo "ERROR: give a run directory, or --list." >&2; usage; }

if ! "${ssh_cmd[@]}" "$target" "command -v rsync >/dev/null"; then
  echo "ERROR: rsync missing on the pod; run: apt-get update && apt-get install -y rsync" >&2
  exit 1
fi

filters=()
[ "$no_video" -eq 0 ] || filters+=(--exclude '*.mp4')
if [ "${#only[@]}" -gt 0 ]; then
  for pattern in "${only[@]}"; do filters+=(--include "$pattern"); done
  filters+=(--include '*/' --exclude '*')
fi

dest="$REPO_DIR/outputs/$run"
rsync_args=(-rltvh --partial --progress --prune-empty-dirs)
if [ "$dry_run" -eq 1 ]; then rsync_args+=(--dry-run); else mkdir -p "$dest"; fi
rsync "${rsync_args[@]}" "${filters[@]}" \
  -e "ssh -p $port -o StrictHostKeyChecking=accept-new" \
  "$target:$remote_root/$run/" "$dest/"
[ "$dry_run" -eq 1 ] || echo "Pulled $run to $dest"
