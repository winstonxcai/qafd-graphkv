#!/bin/bash
#SBATCH --job-name=csa-plain
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

ROOT=/mnt/beegfs/home/Winston/qafd-graphkv
GRAPHKV_PY=/mnt/beegfs/home/Winston/miniconda3/envs/graphkv/bin/python
QAFD_PY=/mnt/beegfs/home/Winston/miniconda3/envs/qafd-rag/bin/python
RETRIEVAL=${RETRIEVAL:-$ROOT/artifacts/qafd/hotpotqa_csa_qafd_500_949.json}
GRAPH=$ROOT/third_party/QAFD-RAG/kg/multihop/gpt-4o-mini_nvidia-nv-embed-v2_hotpotqa/graph.pickle
GRID=${GRID:-$ROOT/artifacts/results/csa_development/grid}
REPORT=${REPORT:-$ROOT/artifacts/results/csa_development}
PORT=${PORT:-9200}

mkdir -p "$REPORT"
cd "$ROOT"
PYTHONPATH="$ROOT" "$QAFD_PY" -m src.eval.csa_results \
  --input-dir "$GRID" --output-dir "$REPORT" --retrieval "$RETRIEVAL"

read -r BETA TOP_B < <("$QAFD_PY" - "$REPORT/csa_normalized_winner.json" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
print(row["beta"], row["top_b"])
PY
)

PYTHONPATH="$ROOT" "$GRAPHKV_PY" -u -m src.eval.csa_server \
  --model ldsjmdy/Tulu3-Block-FT --port "$PORT" --attn-implementation sdpa \
  > "$REPORT/plain_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 180); do
  if "$GRAPHKV_PY" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', $PORT)); s.close()" >/dev/null 2>&1; then
    READY=1
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$REPORT/plain_server.log"; exit 1; }
  sleep 2
done
[[ "$READY" -eq 1 ]] || { echo "CSA server readiness timeout" >&2; exit 1; }

PYTHONPATH="$ROOT" "$QAFD_PY" -u -m src.eval.run_csa_grid \
  --retrieval "$RETRIEVAL" --graph "$GRAPH" --output-dir "$GRID" \
  --port "$PORT" --start 500 --limit 200 --max-new-tokens 256 \
  --plain-beta "$BETA" --plain-b "$TOP_B"

PYTHONPATH="$ROOT" "$QAFD_PY" -m src.eval.csa_results \
  --input-dir "$GRID" --output-dir "$REPORT" --retrieval "$RETRIEVAL" \
  --include-plain
