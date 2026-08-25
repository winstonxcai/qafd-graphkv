#!/bin/bash
#SBATCH --job-name=csa-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

ROOT=/mnt/beegfs/home/Winston/qafd-graphkv
GRAPHKV_PY=/mnt/beegfs/home/Winston/miniconda3/envs/graphkv/bin/python
QAFD_PY=/mnt/beegfs/home/Winston/miniconda3/envs/qafd-rag/bin/python
RETRIEVAL=${RETRIEVAL:-$ROOT/artifacts/qafd/hotpotqa_csa_qafd_500_949.json}
GRAPH=$ROOT/third_party/QAFD-RAG/kg/multihop/gpt-4o-mini_nvidia-nv-embed-v2_hotpotqa/graph.pickle
OUT=$ROOT/artifacts/results/csa_development/smoke_10
PORT=${PORT:-8980}

mkdir -p "$OUT"
cd "$ROOT"
PYTHONPATH="$ROOT" "$GRAPHKV_PY" -m pytest -q \
  tests/test_csa_prompt.py tests/test_csa_routing.py tests/test_csa_attention.py \
  tests/test_csa_graph.py > "$OUT/tests.log" 2>&1

PYTHONPATH="$ROOT" "$GRAPHKV_PY" -u -m src.eval.csa_server \
  --model ldsjmdy/Tulu3-Block-FT --port "$PORT" --attn-implementation sdpa \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 180); do
  if "$GRAPHKV_PY" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', $PORT)); s.close()" >/dev/null 2>&1; then
    READY=1
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$OUT/server.log"; exit 1; }
  sleep 2
done
[[ "$READY" -eq 1 ]] || { echo "CSA server readiness timeout" >&2; exit 1; }

PYTHONPATH="$ROOT" "$QAFD_PY" -u -m src.eval.csa_smoke \
  --retrieval "$RETRIEVAL" --graph "$GRAPH" --port "$PORT" --real-prompts 10 \
  > "$OUT/equivalence.json"

PYTHONPATH="$ROOT" "$QAFD_PY" -u -m src.eval.csa_benchmark \
  --retrieval "$RETRIEVAL" --graph "$GRAPH" \
  --output-dir "$OUT/default_b2_beta0p5" \
  --strategy-name default_b2_beta0p5 --port "$PORT" \
  --start 500 --limit 10 --beta 0.5 --top-b 2 \
  --pooling normalized_token_mean --backend gathered_sdpa \
  --max-new-tokens 256 --warmup
