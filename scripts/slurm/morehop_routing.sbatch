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
MODEL_PATH=${MODEL_PATH:-ldsjmdy/Tulu3-Block-FT}
cd "$ROOT"
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HUGGINGFACE_HUB_CACHE}
PYTHONPATH="$ROOT:$ROOT/third_party/GraphKV" "$GRAPHKV_PY" -u -m src.eval.morehop_routing_runner \
  --dataset "$DATASET" --qid-manifest "$QID_MANIFEST" --output-dir "$OUTPUT_DIR" --model "$MODEL_PATH"
