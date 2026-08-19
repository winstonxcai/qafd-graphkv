#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

: "${METHOD:?Set METHOD to a Test 4 benchmark method}"
: "${PORT:?Set PORT to a unique per-job server port}"

ROOT=/mnt/beegfs/home/Winston/qafd-graphkv
GRAPHKV_PY=/mnt/beegfs/home/Winston/miniconda3/envs/graphkv/bin/python
QAFD_PY=/mnt/beegfs/home/Winston/miniconda3/envs/qafd-rag/bin/python
OUT_ROOT=${OUT_ROOT:-$ROOT/artifacts/results/test4_250_k15_parallel}
LIMIT=${LIMIT:-250}
K=${K:-15}
PROMPT_STYLE=${PROMPT_STYLE:-default}
OUT_DIR=$OUT_ROOT/$METHOD

mkdir -p "$OUT_DIR"

if [[ "$METHOD" == "qafd_recursive_h1_t2" ]]; then
  cd "$ROOT"
  PYTHONPATH="$ROOT" "$GRAPHKV_PY" -m src.eval.recursive_server \
    --model ldsjmdy/Tulu3-Block-FT \
    --port "$PORT" > "$OUT_DIR/server.log" 2>&1 &
else
  cd "$ROOT/third_party/GraphKV"
  "$GRAPHKV_PY" server/generate_server.py \
    --model ldsjmdy/Tulu3-Block-FT \
    --port "$PORT" \
    --dtype bfloat16 > "$OUT_DIR/server.log" 2>&1 &
fi

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

if [[ "$READY" -ne 1 ]]; then
  echo "Generation server did not become ready within 6 minutes" >&2
  exit 1
fi

cd "$ROOT"
EXTRA_ARGS=(--prompt-style "$PROMPT_STYLE")
if [[ -n "${STRATEGY_NAME:-}" ]]; then
  EXTRA_ARGS+=(--strategy-name "$STRATEGY_NAME")
fi
if [[ "${INCLUDE_GRAPH_LINKS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--include-graph-links)
fi
if [[ "${GRAPH_SELECTION:-none}" != "none" ]]; then
  EXTRA_ARGS+=(
    --graph-selection "$GRAPH_SELECTION"
    --pool-k "${POOL_K:-20}"
    --context-k "${CONTEXT_K:?Set CONTEXT_K for graph selection}"
  )
fi
if [[ -n "${SEQUENTIAL_ORDER:-}" ]]; then
  EXTRA_ARGS+=(--sequential-order "$SEQUENTIAL_ORDER")
fi
PYTHONPATH="$ROOT" "$QAFD_PY" -u -m src.eval.test4_benchmark \
  --results outputs/hotpotqa/gpt-4o-mini_nvidia-nv-embed-v2/results_hotpotqa.json \
  --questions third_party/QAFD-RAG/data/multihop/hotpotqa.json \
  --graph third_party/QAFD-RAG/kg/multihop/gpt-4o-mini_nvidia-nv-embed-v2_hotpotqa/graph.pickle \
  --output-dir "$OUT_DIR" \
  --port "$PORT" \
  --recursive-port "$PORT" \
  --k "$K" \
  --limit "$LIMIT" \
  --method "$METHOD" \
  "${EXTRA_ARGS[@]}"
