#!/bin/bash
#SBATCH --job-name=graphkv-morehop
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/home/Winston/old_data/qafd-graphkv/slurm-morehop-%A_%a.out

set -euo pipefail

ROOT=${ROOT:-/mnt/beegfs/home/Winston/old_data/qafd-graphkv}
GRAPHKV_PY=${GRAPHKV_PY:-$ROOT/../miniconda3/envs/graphkv/bin/python}
HOP=${HOP:-${SLURM_ARRAY_TASK_ID:-1}}
PORT=${PORT:-$((8790 + HOP))}
METHODS=${METHODS:-sequential,block_rag,graphkv_top1,graphkv_top3,graphkv_full}
DATASET=${DATASET:-$ROOT/artifacts/results/morehop_faithful_reproduction/input/with_human_verification_ascend.jsonl}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/artifacts/results/morehop_faithful_reproduction}
OUT_DIR=${OUT_DIR:-$RESULT_ROOT/hop_${HOP}}
QID_MANIFEST=${QID_MANIFEST:-}

cd "$ROOT"
mkdir -p "$OUT_DIR"

export HF_HOME="$ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
MODEL_PATH=${MODEL_PATH:-$HUGGINGFACE_HUB_CACHE/models--ldsjmdy--Tulu3-Block-FT/snapshots/62eecbdfa4e12821a487e91ed2ef847a85426efe}
test -r "$MODEL_PATH/config.json"

{
  echo "ROOT=$ROOT"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "HOP=$HOP"
  ls -ld "$MODEL_PATH" "$MODEL_PATH/config.json"
} > "$OUT_DIR/server.log"

PYTHONPATH="$ROOT:$ROOT/third_party/GraphKV" "$GRAPHKV_PY" -u -m src.eval.matched_all_methods_server \
  --model "$MODEL_PATH" --port "$PORT" >> "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 240); do
  if "$GRAPHKV_PY" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', $PORT)); s.close()" >/dev/null 2>&1; then
    READY=1
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$OUT_DIR/server.log"; exit 1; }
  sleep 2
done
[[ "$READY" -eq 1 ]] || { tail -100 "$OUT_DIR/server.log"; exit 1; }

EXTRA_ARGS=()
if [[ -n "$QID_MANIFEST" ]]; then
  EXTRA_ARGS+=(--qid-manifest "$QID_MANIFEST")
fi
PYTHONPATH="$ROOT:$ROOT/third_party/GraphKV" "$GRAPHKV_PY" -u -m src.eval.graphkv_faithful_morehop \
  --dataset "$DATASET" --output-dir "$OUT_DIR" --port "$PORT" --hop "$HOP" --methods "$METHODS" "${EXTRA_ARGS[@]}"
