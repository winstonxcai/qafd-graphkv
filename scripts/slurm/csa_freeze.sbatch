#!/bin/bash
#SBATCH --job-name=csa-freeze
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT=/mnt/beegfs/home/Winston/qafd-graphkv
PY=/mnt/beegfs/home/Winston/miniconda3/envs/qafd-rag/bin/python
RETRIEVAL=${RETRIEVAL:-$ROOT/artifacts/qafd/hotpotqa_csa_qafd_500_949.json}
GRID=${GRID:-$ROOT/artifacts/results/csa_development/grid}
REPORT=${REPORT:-$ROOT/artifacts/results/csa_development}

cd "$ROOT"
PYTHONPATH="$ROOT" "$PY" -m src.eval.csa_results \
  --input-dir "$GRID" --output-dir "$REPORT" --retrieval "$RETRIEVAL" \
  --include-plain --freeze
