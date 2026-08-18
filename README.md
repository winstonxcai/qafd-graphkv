Yes. Given intermittent shared A800 access, I would **design the project so GPU time is only needed for two bounded stages**:

[
\text{QAFD retrieval/embedding}
\rightarrow
\boxed{\text{serialize passage graph}}
]

then separately:

[
\boxed{\text{serialized passage graph}}
\rightarrow
\text{Graph-KV experiments}.
]

Do **not** make every Graph-KV run rerun QAFD.

My initial target would be **1 A800 GPU, Tulu3-Block-FT 8B, HotpotQA, (k=5), 3 passages for unit tests, then 5 for real tests**. Graph-KV's official implementation uses the 8B Tulu3-Block-FT model and its README's standard RAG server command uses a single GPU. ([[GitHub](https://github.com/Graph-COM/GraphKV)][1])

## 1. Hardware I would actually request

Start with:

```text
GPU:      1 × A800
CPU:      8–16 cores
RAM:      64 GB
Disk:     ≥100 GB scratch
```

Don't request the whole node initially.

First thing after allocation:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

so you know whether your particular A800 allocation has enough headroom. The model is 8B parameters, so BF16 weights alone are roughly (16) GB; the real additional concern is Graph-KV's intermediate KV copies and concatenations. ([[Hugging Face](https://huggingface.co/ldsjmdy/Tulu3-Block-FT)][2])

For the first smoke test I would use:

[
k=3,\qquad
L_P\le512\text{ tokens},\qquad
T\le2.
]

Then move to:

[
k=5,\qquad
T\in{0,1,2,3}.
]

If your A800 has ample memory, increase passage length afterward. **Don't start by optimizing for maximum context.**

If eventually you get 4–8 A800s, I would still put **one 8B model per GPU and shard questions across GPUs**, rather than model-parallelizing one 8B model. That's much friendlier to a shared cluster.

---

# 2. Repository organization

I would create a new repo around the two upstream projects rather than merging their codebases:

```text
qafd-graphkv/
│
├── third_party/
│   ├── QAFD-RAG/
│   └── GraphKV/
│
├── src/
│   ├── qafd_bridge/
│   │   ├── export_retrieval.py
│   │   └── export_trace.py
│   │
│   ├── graph/
│   │   ├── connectivity.py
│   │   └── stats.py
│   │
│   ├── recursive_kv/
│   │   ├── cache_ops.py
│   │   ├── prefill.py
│   │   └── propagate.py
│   │
│   └── eval/
│       ├── hotpot.py
│       └── metrics.py
│
├── artifacts/
│   ├── qafd/
│   ├── connectivity/
│   └── results/
│
├── configs/
│   ├── smoke.yaml
│   └── hotpot.yaml
│
└── scripts/
    ├── gpu_smoke.sh
    └── slurm/
```

Clone upstream as submodules:

```bash
mkdir qafd-graphkv
cd qafd-graphkv
git init

git submodule add \
  https://github.com/Tarzanagh/QAFD-RAG \
  third_party/QAFD-RAG

git submodule add \
  https://github.com/Graph-COM/GraphKV \
  third_party/GraphKV

git add .
git commit -m "Initialize QAFD GraphKV prototype"
```

Both are currently the authors' official repositories. ([[GitHub](https://github.com/Graph-COM/GraphKV)][1])

**I would avoid modifying GraphKV or QAFD directly as much as possible.** Put our new logic in `src/`.

---

# 3. Keep two separate environments

This matters. Don't try to force QAFD and Graph-KV into one environment initially.

### QAFD

```bash
conda create -n qafd python=3.10 -y
conda activate qafd

cd third_party/QAFD-RAG
pip install -r requirements.txt
```

That's the setup QAFD currently documents. ([[GitHub](https://github.com/Tarzanagh/QAFD-RAG)][3])

Download the **prebuilt multihop KG** rather than rebuilding it:

```bash
huggingface-cli download tarzanagh/QAFD-RAG \
  --include "kg/multihop/*" \
  --local-dir .
```

The authors explicitly recommend the prebuilt KGs because rebuilding costs hours plus LLM/API work. Their multihop setup uses NV-Embed-v2 and says it requires a GPU with at least 16 GB VRAM. ([[GitHub](https://github.com/Tarzanagh/QAFD-RAG)][3])

### Graph-KV

```bash
conda create -n graphkv python=3.10.16 -y
conda activate graphkv

conda install \
  pytorch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  pytorch-cuda=12.1 \
  -c pytorch -c nvidia

pip install transformers==4.50.0
pip install accelerate pandas
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install fire flask-cors
```

Those PyTorch/Transformers/FlashAttention versions match Graph-KV's published environment. ([[GitHub](https://github.com/Graph-COM/GraphKV)][1])

I'd also put Hugging Face models somewhere persistent:

```bash
export HF_HOME=/path/to/shared/scratch/$USER/hf-cache
mkdir -p "$HF_HOME"
```

so a new allocation does not download the model again.

---

# 4. Test 0 — confirm Graph-KV works untouched

**Do this before writing any new method.**

Start the official model on one A800:

```bash
conda activate graphkv
cd third_party/GraphKV

CUDA_VISIBLE_DEVICES=0 \
python server/generate_server.py \
  --model ldsjmdy/Tulu3-Block-FT \
  --port 8771 \
  --dtype bfloat16
```

That is the authors' published single-GPU command. ([[GitHub](https://github.com/Graph-COM/GraphKV)][1])

Graph-KV already exposes:

```text
vanilla
gapemp          # Graph-KV
block
gapemp_appr
```

and HotpotQA is task `hqa`. ([[GitHub](https://github.com/Graph-COM/GraphKV)][1])

So before touching anything, run perhaps **10 HotpotQA examples** under:

```text
vanilla
gapemp
```

Record:

```text
answer
wall time
TTFT
peak GPU memory
```

At this stage I do **not care about statistical significance**. The goal is:

> Can I reproduce the repo and correctly manipulate its caches on my A800?

---

# 5. Test 1 — synthetic recursive-prefill test

This is the most important implementation test.

**Do this before QAFD integration.**

Hard-code:

[
P_0-P_1-P_2.
]

Create two test cases, A and B:

```text
A:
P0 = same
P1 = same
P2 = "The secret value is RED."

B:
P0 = same
P1 = same
P2 = "The secret value is BLUE."
```

Initial caches:

[
KV_i^{(0)}=LLM(P_i).
]

Round 1:

[
KV_0^{(1)}
==========

LLM(P_0\mid KV_1^{(0)})
]

[
KV_1^{(1)}
==========

LLM(P_1\mid KV_0^{(0)},KV_2^{(0)})
]

[
KV_2^{(1)}
==========

LLM(P_2\mid KV_1^{(0)}).
]

Because (P_0) cannot see (P_2) in round 1:

[
KV_{0,A}^{(1)}
\approx
KV_{0,B}^{(1)}.
]

But in round 2:

[
KV_0^{(2)}
==========

LLM(P_0\mid KV_1^{(1)}),
]

and (KV_1^{(1)}) now contains information from (P_2).

Therefore:

[
\boxed{
KV_{0,A}^{(2)}
\ne
KV_{0,B}^{(2)}.
}
]

This directly verifies our claim:

[
\boxed{
T\text{ recursive prefills}
\Longrightarrow
T\text{ passage hops of information propagation}.
}
]

This is a much better first test than jumping directly to QA accuracy.

---

# 6. Don't rewrite Graph-KV's cache machinery

Graph-KV already has almost exactly the low-level operations you need.

Its current `pcw.py` does:

1. independent context prefill,
2. obtain `past_key_values`,
3. remove/reapply RoPE positioning,
4. flatten/stack/concatenate caches,
5. run a second context pass with the first caches supplied as `past_key_values`. ([[GitHub](https://github.com/Graph-COM/GraphKV/blob/main/pcw.py)][4])

In particular, reuse their functions around:

```python
apply_pkv_rerotary_position_embeddings(...)
apply_pkv_rotary_position_embeddings(...)
flatten_pkv(...)
stack_pkv(...)
concact_pkv(...)
cut_pkv(...)
```

Their current `gapemp` effectively does:

[
KV^{(0)}=LLM(P)
]

followed by another model invocation using the prior cache:

[
KV^{(1)}=LLM(P\mid KV^{(0)}).
]

([[GitHub](https://github.com/Graph-COM/GraphKV/blob/main/pcw.py)][4])

So your `propagate.py` should conceptually be only:

```python
kv = independent_prefill(passages)

for t in range(T):
    next_kv = []

    for i, passage in enumerate(passages):
        source_kv = [
            kv[j]
            for j in neighbors[i]
        ]

        next_kv.append(
            graphkv_prefill(
                passage,
                source_kv,
                round_id=t + 1,
            )
        )

    kv = next_kv
```

The difficult part is **RoPE/cache positioning**, not the recursion loop. Reuse Graph-KV's implementation rather than reconstructing it.

Also inspect `gapemp_graph` in `pcw_parallel.py`: the repository already has a graph-specific mode taking a center node and neighbor nodes for its Arxiv graph experiment. That's likely the closest starting point for arbitrary QAFD adjacency. ([[GitHub](https://github.com/Graph-COM/GraphKV/blob/main/pcw_parallel.py)][5])

---

# 7. Test 2 — extract QAFD structure, without Graph-KV

This requires **zero recursive LLM work**.

Run QAFD on, say:

[
n=20
]

HotpotQA questions using its prebuilt KG:

```bash
conda activate qafd
cd third_party/QAFD-RAG

python benchmarks/run.py \
  --task multihop \
  --dataset hotpotqa \
  --questions 20 \
  --skip_qa
```

QAFD supports HotpotQA directly and exposes retrieval-only execution. ([[GitHub](https://github.com/Tarzanagh/QAFD-RAG)][3])

### One small QAFD modification

Currently:

```python
node_scores = qafd.run(...)
```

produces a score for **every graph node**, but the wrapper then extracts passage scores and returns only:

```python
sorted_ids, sorted_scores
```

to the retriever. ([[GitHub](https://github.com/Tarzanagh/QAFD-RAG/blob/main/src/passage_entity/graph_adapter.py)][6])

So make the smallest possible patch:

```python
def run_igraph_qafd(..., return_trace=False):
    ...
    node_scores = qafd.run(...)

    ...

    if return_trace:
        return sorted_ids, sorted_scores, node_scores

    return sorted_ids, sorted_scores
```

This does **not change QAFD at all**.

It merely preserves information QAFD currently throws away.

The retriever's `_graph_search` already calls this function after constructing the entity/passages seeds, so this is the natural trace point. ([[GitHub](https://github.com/Tarzanagh/QAFD-RAG/blob/main/src/passage_entity/retriever.py)][7])

---

# 8. Serialize QAFD results

For every question, write one JSONL row:

```json
{
  "qid": "...",
  "question": "...",
  "answer": "...",

  "passages": [
    {
      "pid": "...",
      "text": "...",
      "qafd_score": 0.18
    }
  ],

  "edges": [
    {
      "src": 0,
      "dst": 1,
      "entity_hops": 1,
      "path": ["entity-A", "entity-B"]
    }
  ],

  "diameter": 2
}
```

Then the Graph-KV environment **never imports QAFD**.

Your interface is simply:

[
\boxed{
\text{QAFD}
\rightarrow
\texttt{question.jsonl}
\rightarrow
\text{Graph-KV}.
}
]

That's valuable given your unstable A800 access.

---

# 9. Test 3 — determine whether the QAFD passage graph is even useful

Before expensive LLM experiments, answer this empirical question:

> For QAFD's top-5 passages, what does the induced passage graph actually look like?

For each selected pair (P_i,P_j), perform bounded BFS through entity nodes.

Start with:

[
h\in{0,1,2}
]

where:

[
h=0:\quad P_i-e-P_j
]

[
h=1:\quad P_i-e_1-e_2-P_j
]

[
h=2:\quad P_i-e_1-e_2-e_3-P_j.
]

Do **not** calculate (R^h). Just bounded BFS.

For every question log:

```text
# selected passages
# selected/visited entities
# passage-passage edges
connected components
largest component
average degree
diameter
entity-hop length per edge
```

This experiment is CPU-cheap.

### The most important go/no-go result

Suppose (k=5).

Bad outcome 1:

```text
P1   P2   P3   P4   P5
```

Almost everything disconnected.

Then our QAFD structure does not provide enough routing information.

Bad outcome 2:

```text
P1 -- P2
|\  /|\
| \/ | \
| /\ |  ...
```

Nearly complete graph for every question.

Then the QAFD structure isn't providing much useful sparsity.

What we want is something like:

```text
P1 -- P2 -- P3
      |
      P4 -- P5
```

i.e. **sparse, meaningful connectivity with nontrivial diameter**.

I would inspect this on **100–500 queries before spending much GPU time**.

---

# 10. Test 4 — first actual research experiment

Once the graph statistics look sensible, use exactly the **same QAFD top-5 passages** for every method.

Run:

| Method                 | Topology                   |        Rounds |
| ---------------------- | -------------------------- | ------------: |
| Sequential             | full sequence              |             — |
| Original Graph-KV      | original top-(m) heuristic |             1 |
| QAFD-GraphKV           | QAFD graph                 |             1 |
| Recursive QAFD-GraphKV | QAFD graph                 |             2 |
| Adaptive recursive     | QAFD graph                 | (T=\min(D,3)) |

Where:

[
D=\operatorname{diameter}(G_P).
]

I would initially do:

[
n=50.
]

If there's any signal:

[
n=200\rightarrow500.
]

Only run the full benchmark after that.

Measure:

[
\boxed{\text{EM, F1}}
]

and systems metrics:

[
\boxed{
\text{prefill latency,\ TTFT,\ total latency,\ peak VRAM}
}
]

plus:

[
\boxed{
|E_P|,\ D,\ T.
}
]

QAFD's official HotpotQA benchmark itself reports EM/F1, so those are the natural outcome metrics. ([[GitHub](https://github.com/Tarzanagh/QAFD-RAG)][3])

---

# 11. A800 strategy for intermittent availability

This is perhaps the most important practical piece.

Your jobs should be **fully restartable**.

Never write one big:

```text
run_500_questions.py
```

that loses everything when the node disappears.

Instead:

```bash
python run_hotpot.py \
    --start 0 \
    --end 25 \
    --method recursive \
    --rounds 2
```

and append every finished example immediately:

```text
artifacts/results/
  qafd_graphkv_t2/
    shard_000_025.jsonl
    shard_025_050.jsonl
    ...
```

Better still, each result should be keyed by:

```text
qid + method + configuration
```

and skip already completed IDs.

So if your allocation ends after example 17:

[
\text{lost work}\approx1\text{ question}
]

rather than the whole run.

---

# 12. What I would do in the first three development stages

### Stage A — no scarce GPU required

```text
Clone repos
Download data/KG/models
Understand QAFD output
Patch trace output
Run passage-connectivity statistics
Write JSONL bridge
```

The QAFD repo provides prebuilt multihop KGs specifically to avoid rebuilding them. ([[GitHub](https://github.com/Tarzanagh/QAFD-RAG)][3])

### Stage B — first 1×A800 allocation

Do only:

```text
1. Load Tulu3-Block-FT
2. Reproduce vanilla Graph-KV
3. Reproduce gapemp Graph-KV
4. Measure VRAM
5. Run synthetic P0—P1—P2 recursive test
```

I would consider the allocation successful once:

[
KV_{0,A}^{(1)}
\approx KV_{0,B}^{(1)}
]

but:

[
KV_{0,A}^{(2)}
\neq KV_{0,B}^{(2)}.
]

That proves the recursive mechanism itself.

### Stage C — second 1×A800 allocation

Use **precomputed QAFD JSONL** and run:

```text
50 HotpotQA examples
k = 5

vanilla
Graph-KV
QAFD topology T=1
QAFD topology T=2
adaptive T≤3
```

At that point you will know whether there's a research signal.

---

## What I would **not** do initially

I wouldn't rebuild the HotpotQA KG, train anything, modify CUDA kernels, use 8 A800s, test a large model, implement relation-aware QAFD, or optimize throughput.

The smallest meaningful experiment is:

[
\boxed{
\text{HotpotQA}
+
\text{existing QAFD KG}
+
\text{top-5 passages}
+
\text{Tulu3-Block-FT 8B}
+
\text{1 A800}
}
]

and the first genuinely novel result is:

[
\boxed{
\text{Does }T=2\text{ QAFD-topology recursive prefill beat }T=1
\text{ and original Graph-KV?}
}
]

I think **one A800 is enough to establish or reject that hypothesis**; additional GPUs mostly buy you evaluation throughput, not new capability.

[1]: https://github.com/Graph-COM/GraphKV "GitHub - Graph-COM/GraphKV · GitHub"
[2]: https://huggingface.co/ldsjmdy/Tulu3-Block-FT "ldsjmdy/Tulu3-Block-FT · Hugging Face"
[3]: https://github.com/Tarzanagh/QAFD-RAG "GitHub - Tarzanagh/QAFD-RAG: QAFD-RAG is a graph-based retrieval-augmented generation framework that uses query-aware flow diffusion to retrieve contextually relevant subgraphs from a knowledge graph with statistical retrieval guarantees. · GitHub"
[4]: https://github.com/Graph-COM/GraphKV/blob/main/pcw.py "GraphKV/pcw.py at main · Graph-COM/GraphKV · GitHub"
[5]: https://github.com/Graph-COM/GraphKV/blob/main/pcw_parallel.py "GraphKV/pcw_parallel.py at main · Graph-COM/GraphKV · GitHub"
[6]: https://github.com/Tarzanagh/QAFD-RAG/blob/main/src/passage_entity/graph_adapter.py "QAFD-RAG/src/passage_entity/graph_adapter.py at main · Tarzanagh/QAFD-RAG · GitHub"
[7]: https://github.com/Tarzanagh/QAFD-RAG/blob/main/src/passage_entity/retriever.py "QAFD-RAG/src/passage_entity/retriever.py at main · Tarzanagh/QAFD-RAG · GitHub"
