#!/bin/bash
#SBATCH --job-name=csa-qafd-500-949
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

ROOT=/mnt/beegfs/home/Winston/qafd-graphkv
PY=/mnt/beegfs/home/Winston/miniconda3/envs/qafd-rag/bin/python
OUT=${OUT:-$ROOT/artifacts/qafd/hotpotqa_csa_qafd_500_949.json}

cd "$ROOT"
PYTHONPATH="$ROOT" "$PY" -u -m src.qafd_bridge.run_qafd_slice \
  --qafd-root "$ROOT/third_party/QAFD-RAG" \
  --dataset hotpotqa \
  --start 500 \
  --questions 450 \
  --retrieval-top-k 5 \
  --entity-hops 1 \
  --save-results "$OUT"

"$PY" - "$OUT" <<'PY'
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
assert [row["qid"] for row in payload["per_query"]] == list(range(500, 950))
print(json.dumps({"artifact": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "questions": 450}))
PY
