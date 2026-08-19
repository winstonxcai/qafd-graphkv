#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

: "${METHOD:?Set METHOD to sequential or qafd_graphkv}"
: "${PORT:?Set a unique server port}"
: "${OUT_DIR:?Set the result directory}"
: "${STRATEGY_NAME:?Set the strategy name}"

ROOT=/mnt/beegfs/home/Winston/qafd-graphkv
GRAPHKV_PY=/mnt/beegfs/home/Winston/miniconda3/envs/graphkv/bin/python
QAFD_PY=/mnt/beegfs/home/Winston/miniconda3/envs/qafd-rag/bin/python
POOL_K=${POOL_K:-20}
HOPS=${HOPS:-0}
CENTER_RULE=${CENTER_RULE:-best_edge}
MAX_NEIGHBORS=${MAX_NEIGHBORS:-4}
MAX_STARS=${MAX_STARS:-1}
PROMPT_STYLE=${PROMPT_STYLE:-concise}
LIMIT=${LIMIT:-250}

mkdir -p "$OUT_DIR"
cd "$ROOT"
PYTHONPATH="$ROOT" "$GRAPHKV_PY" -m src.eval.qafd_graph_server \
  --model ldsjmdy/Tulu3-Block-FT \
  --port "$PORT" > "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 180); do
  if "$GRAPHKV_PY" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', $PORT)); s.close()" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Generation server exited before becoming ready" >&2
    exit 1
  fi
  sleep 2
done
[[ "$READY" -eq 1 ]] || { echo "Server readiness timeout" >&2; exit 1; }

PYTHONPATH="$ROOT" "$QAFD_PY" -u -m src.eval.qafd_graph_pair_benchmark \
  --results outputs/hotpotqa/gpt-4o-mini_nvidia-nv-embed-v2/results_hotpotqa.json \
  --questions third_party/QAFD-RAG/data/multihop/hotpotqa.json \
  --graph third_party/QAFD-RAG/kg/multihop/gpt-4o-mini_nvidia-nv-embed-v2_hotpotqa/graph.pickle \
  --output-dir "$OUT_DIR" \
  --method "$METHOD" \
  --strategy-name "$STRATEGY_NAME" \
  --port "$PORT" \
  --pool-k "$POOL_K" \
  --hops "$HOPS" \
  --center-rule "$CENTER_RULE" \
  --max-neighbors "$MAX_NEIGHBORS" \
  --max-stars "$MAX_STARS" \
  --prompt-style "$PROMPT_STYLE" \
  --limit "$LIMIT"
