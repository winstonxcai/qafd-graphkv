#!/bin/bash
#SBATCH --job-name=csa-grid
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-7

set -euo pipefail

ROOT=/mnt/beegfs/home/Winston/qafd-graphkv
GRAPHKV_PY=/mnt/beegfs/home/Winston/miniconda3/envs/graphkv/bin/python
QAFD_PY=/mnt/beegfs/home/Winston/miniconda3/envs/qafd-rag/bin/python
RETRIEVAL=${RETRIEVAL:-$ROOT/artifacts/qafd/hotpotqa_csa_qafd_500_949.json}
GRAPH=$ROOT/third_party/QAFD-RAG/kg/multihop/gpt-4o-mini_nvidia-nv-embed-v2_hotpotqa/graph.pickle
OUT_ROOT=${OUT_ROOT:-$ROOT/artifacts/results/csa_development/grid}
WORKER_COUNT=${WORKER_COUNT:-8}
PORT=$((9000 + SLURM_ARRAY_TASK_ID))
LOG_DIR=$OUT_ROOT/worker_logs

mkdir -p "$LOG_DIR"
cd "$ROOT"
PYTHONPATH="$ROOT" "$GRAPHKV_PY" -u -m src.eval.csa_server \
  --model ldsjmdy/Tulu3-Block-FT --port "$PORT" --attn-implementation sdpa \
  > "$LOG_DIR/server_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 180); do
  if "$GRAPHKV_PY" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', $PORT)); s.close()" >/dev/null 2>&1; then
    READY=1
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$LOG_DIR/server_${SLURM_ARRAY_TASK_ID}.log"; exit 1; }
  sleep 2
done
[[ "$READY" -eq 1 ]] || { echo "CSA server readiness timeout" >&2; exit 1; }

PYTHONPATH="$ROOT" "$QAFD_PY" -u -m src.eval.run_csa_grid \
  --retrieval "$RETRIEVAL" --graph "$GRAPH" --output-dir "$OUT_ROOT" \
  --port "$PORT" --worker-index "$SLURM_ARRAY_TASK_ID" \
  --worker-count "$WORKER_COUNT" --start 500 --limit 200 --max-new-tokens 256 \
  > "$LOG_DIR/grid_${SLURM_ARRAY_TASK_ID}.log" 2>&1
