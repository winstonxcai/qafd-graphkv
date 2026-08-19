#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

ROOT=/mnt/beegfs/home/Winston/qafd-graphkv
PY=/mnt/beegfs/home/Winston/miniconda3/envs/qafd-rag/bin/python
OUT=${OUT:-$ROOT/artifacts/results/test4_holdout_500_qafd_retrieval.json}

cd "$ROOT"
PYTHONPATH="$ROOT" "$PY" -u -m src.qafd_bridge.run_qafd_topology_input \
  --qafd-root "$ROOT/third_party/QAFD-RAG" \
  --dataset hotpotqa \
  --questions 500 \
  --retrieval-top-k 20 \
  --save-results "$OUT"
