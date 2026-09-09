#!/bin/bash
#SBATCH --job-name=graphkv-parity
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfs/home/Winston/old_data/qafd-graphkv/slurm-morehop-parity-%j.out

set -euo pipefail

ROOT=${ROOT:-/mnt/beegfs/home/Winston/old_data/qafd-graphkv}
GRAPHKV_PY=${GRAPHKV_PY:-$ROOT/../miniconda3/envs/graphkv/bin/python}
DATASET=${DATASET:-$ROOT/artifacts/results/morehop_faithful_reproduction/input/with_human_verification_ascend.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/artifacts/results/morehop_engine_parity}

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

export HF_HOME="$ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
MODEL_PATH=${MODEL_PATH:-$HUGGINGFACE_HUB_CACHE/models--ldsjmdy--Tulu3-Block-FT/snapshots/62eecbdfa4e12821a487e91ed2ef847a85426efe}
test -r "$MODEL_PATH/config.json"

PYTHONPATH="$ROOT:$ROOT/third_party/GraphKV" "$GRAPHKV_PY" -u -m src.eval.graphkv_engine_parity \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL_PATH" \
  --teacher-forced-steps 8 \
  --max-new-tokens 256
