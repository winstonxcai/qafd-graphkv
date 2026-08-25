#!/bin/bash
#SBATCH --job-name=csa-controls
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-1

set -euo pipefail

ROOT=/mnt/beegfs/home/Winston/qafd-graphkv
GRAPHKV_PY=/mnt/beegfs/home/Winston/miniconda3/envs/graphkv/bin/python
QAFD_PY=/mnt/beegfs/home/Winston/miniconda3/envs/qafd-rag/bin/python
RETRIEVAL=${RETRIEVAL:-$ROOT/artifacts/qafd/hotpotqa_csa_qafd_500_949.json}
GRAPH=$ROOT/third_party/QAFD-RAG/kg/multihop/gpt-4o-mini_nvidia-nv-embed-v2_hotpotqa/graph.pickle
OUT_ROOT=${OUT_ROOT:-$ROOT/artifacts/results/csa_development/grid}
PORT=$((9100 + SLURM_ARRAY_TASK_ID))

if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
  NAME=dense_vanilla_sdpa
  IMPLEMENTATION=sdpa
else
  NAME=dense_vanilla_flash2
  IMPLEMENTATION=flash_attention_2
fi
mkdir -p "$OUT_ROOT/$NAME"
cd "$ROOT"
PYTHONPATH="$ROOT" "$GRAPHKV_PY" -u -m src.eval.csa_server \
  --model ldsjmdy/Tulu3-Block-FT --port "$PORT" \
  --attn-implementation "$IMPLEMENTATION" > "$OUT_ROOT/$NAME/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 180); do
  if "$GRAPHKV_PY" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', $PORT)); s.close()" >/dev/null 2>&1; then
    READY=1
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$OUT_ROOT/$NAME/server.log"; exit 1; }
  sleep 2
done
[[ "$READY" -eq 1 ]] || { echo "CSA control server readiness timeout" >&2; exit 1; }

PYTHONPATH="$ROOT" "$QAFD_PY" -u -m src.eval.csa_benchmark \
  --retrieval "$RETRIEVAL" --graph "$GRAPH" \
  --output-dir "$OUT_ROOT/$NAME" --strategy-name "$NAME" --port "$PORT" \
  --start 500 --limit 200 --mode vanilla --beta 0 --top-b 4 \
  --pooling normalized_token_mean --backend gathered_sdpa \
  --max-new-tokens 256 --warmup
