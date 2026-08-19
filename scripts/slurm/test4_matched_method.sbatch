#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

: "${METHOD:?Set METHOD to one Test 4 method}"
ROOT=/mnt/beegfs/home/Winston/qafd-graphkv
GRAPHKV_PY=/mnt/beegfs/home/Winston/miniconda3/envs/graphkv/bin/python
QAFD_PY=/mnt/beegfs/home/Winston/miniconda3/envs/qafd-rag/bin/python
PORT=${PORT:-8791}
OUT_ROOT=${OUT_ROOT:-$ROOT/artifacts/results/test4_250_winner_matched_methods}
OUT_DIR=$OUT_ROOT/$METHOD

mkdir -p "$OUT_DIR"
cd "$ROOT"
PYTHONPATH="$ROOT" "$GRAPHKV_PY" -m src.eval.matched_all_methods_server \
  --model ldsjmdy/Tulu3-Block-FT --port "$PORT" > "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 180); do
  if "$GRAPHKV_PY" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', $PORT)); s.close()" >/dev/null 2>&1; then
    READY=1
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "Matched server exited" >&2; exit 1; }
  sleep 2
done
[[ "$READY" -eq 1 ]] || { echo "Matched server readiness timeout" >&2; exit 1; }

PYTHONPATH="$ROOT" "$QAFD_PY" -u -m src.eval.test4_benchmark \
  --results outputs/hotpotqa/gpt-4o-mini_nvidia-nv-embed-v2/results_hotpotqa.json \
  --questions third_party/QAFD-RAG/data/multihop/hotpotqa.json \
  --graph third_party/QAFD-RAG/kg/multihop/gpt-4o-mini_nvidia-nv-embed-v2_hotpotqa/graph.pickle \
  --output-dir "$OUT_DIR" --port "$PORT" --recursive-port "$PORT" \
  --k 20 --limit 250 --method "$METHOD" --matched-prompt --max-new-tokens 256
