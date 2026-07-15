#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${1:-/home/spco/diff_bitnet/dre_bcm}
cd "$ROOT_DIR"

if [ ! -d pids ]; then
  echo "no pids directory under $ROOT_DIR"
  exit 0
fi

for pidfile in pids/*.pid; do
  [ -e "$pidfile" ] || continue
  name=$(basename "$pidfile" .pid)
  pid=$(cat "$pidfile")
  echo "=== $name ==="
  ps -p "$pid" -o pid,etime,pcpu,pmem,stat,cmd || true
  log_path="logs/${name}.log"
  if [ -f "$log_path" ]; then
    echo "--- log tail ---"
    tail -n 20 "$log_path" || true
  fi
  outdir="results/matrix_fit/${name}"
  if [ -d "$outdir" ]; then
    echo "--- output files ---"
    find "$outdir" -type f | wc -l
  fi
  echo
done
