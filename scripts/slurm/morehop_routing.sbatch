#!/bin/bash
#SBATCH --job-name=morehop-routing
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
ROOT=${ROOT:-/mnt/beegfs/home/Winston/old_data/qafd-graphkv}
GRAPHKV_PY=${GRAPHKV_PY:-$ROOT/../miniconda3/envs/graphkv/bin/python}
DATASET=${DATASET:-$ROOT/artifacts/results/morehop_faithful_reproduction/input/with_human_verification_ascend.jsonl}
QID_MANIFEST=${QID_MANIFEST:?set QID_MANIFEST to the frozen existing manifest}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/artifacts/results/morehop_routing}
MODEL_PATH=${MODEL_PATH:-$ROOT/.cache/huggingface/hub/models--ldsjmdy--Tulu3-Block-FT/snapshots/62eecbdfa4e12821a487e91ed2ef847a85426efe}
cd "$ROOT"
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HUGGINGFACE_HUB_CACHE}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$OUTPUT_DIR"
EXTRA_ARGS=()
if [[ "${PARITY_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--parity-only --methods full,released_last1,released_last3)
else
  EXTRA_ARGS+=(--parity-report "${PARITY_REPORT:?set PARITY_REPORT}" --methods "${METHODS:-query_bm25_k1}")
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
PYTHONPATH="$ROOT:$ROOT/third_party/GraphKV" "$GRAPHKV_PY" -u -m src.eval.morehop_routing_runner \
  --dataset "$DATASET" --qid-manifest "$QID_MANIFEST" --output-dir "$OUTPUT_DIR" --model "$MODEL_PATH" "${EXTRA_ARGS[@]}"
