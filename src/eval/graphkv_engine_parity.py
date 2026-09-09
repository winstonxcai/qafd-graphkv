"""Compare project-owned GraphKV Full cache semantics with upstream ``gapemp``.

The upstream submodule remains untouched.  A lightweight model proxy lets the
public function construct its cache normally, captures that cache immediately
before answer decoding, and terminates the public decoding loop after one step.
The captured cache is then compared with an independently implemented
project-owned cache construction on identical token IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


ROOT = Path(__file__).resolve().parents[2]
GRAPHKV_ROOT = ROOT / "third_party" / "GraphKV"
sys.path.insert(0, str(GRAPHKV_ROOT))

from pcw import (  # noqa: E402
    apply_pkv_rerotary_position_embeddings,
    apply_pkv_rotary_position_embeddings,
    concact_pkv,
    cut_pkv,
    flatten_pkv,
    gapemp as upstream_gapemp,
    stack_pkv,
)

from src.eval.graphkv_faithful_morehop import (  # noqa: E402
    load_rows,
    prompt_hash,
    upstream_blocks,
)


CONTEXT_LIMIT = 8192
CONTEXT_RESERVE = 256


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_cache(cache: DynamicCache) -> DynamicCache:
    cloned = DynamicCache()
    cloned.key_cache = [tensor.clone().detach() for tensor in cache.key_cache]
    cloned.value_cache = [tensor.clone().detach() for tensor in cache.value_cache]
    return cloned


def tokenize_inputs(tokenizer, blocks: list[str]) -> dict[str, torch.Tensor]:
    prefix, middle, *contexts, query = blocks
    prefix_ids = tokenizer(
        prefix, return_tensors="pt", add_special_tokens=False
    ).input_ids
    middle_ids = tokenizer(
        middle, return_tensors="pt", add_special_tokens=False
    ).input_ids
    query_ids = tokenizer(
        query, return_tensors="pt", add_special_tokens=False
    ).input_ids
    max_context_length = (
        CONTEXT_LIMIT
        - prefix_ids.shape[1]
        - middle_ids.shape[1]
        - query_ids.shape[1]
        - CONTEXT_RESERVE
    )
    context_ids = tokenizer(
        contexts,
        return_tensors="pt",
        truncation=True,
        max_length=max_context_length,
        padding=True,
        add_special_tokens=False,
    ).input_ids
    return {
        "prefix_ids": prefix_ids,
        "middle_ids": middle_ids,
        "query_ids": query_ids,
        "context_ids": context_ids,
    }


def prepare_project_full_cache(model, tokenizer, emb, blocks: list[str]) -> tuple[DynamicCache, dict]:
    """Project-owned implementation of the upstream Full raw+updated cache."""

    ids = tokenize_inputs(tokenizer, blocks)
    prefix_ids = ids["prefix_ids"]
    context_ids = ids["context_ids"]
    context_mask = (context_ids != tokenizer.pad_token_id).reshape(-1)
    batch_size = context_ids.shape[0]
    flattened_context_length = int(context_mask.sum().item())

    with torch.inference_mode():
        context_outputs = model(context_ids.to(model.device), use_cache=True)
        context_pkv = apply_pkv_rerotary_position_embeddings(
            context_outputs.past_key_values, emb
        )
        context_pkv = flatten_pkv(context_pkv, context_mask)
        raw_context = clone_cache(context_pkv)

        context_pkv = apply_pkv_rotary_position_embeddings(context_pkv, emb)
        context_pkv = stack_pkv(context_pkv, batch_size)
        updated_outputs = model(
            context_ids.to(model.device), past_key_values=context_pkv, use_cache=True
        )
        updated_positions = (
            torch.arange(context_ids.shape[-1], dtype=torch.int64)
            + flattened_context_length
        )
        updated_pkv = cut_pkv(updated_outputs.past_key_values, updated_positions)
        updated_pkv = apply_pkv_rerotary_position_embeddings(
            updated_pkv, emb, updated_positions
        )
        updated_pkv = flatten_pkv(updated_pkv, context_mask)
        context_pkv = concact_pkv(raw_context, updated_pkv)

        prefix_outputs = model(prefix_ids.to(model.device), use_cache=True)
        prefix_pkv = apply_pkv_rerotary_position_embeddings(
            prefix_outputs.past_key_values, emb
        )
        final_cache = apply_pkv_rotary_position_embeddings(
            concact_pkv(prefix_pkv, context_pkv), emb
        )

    metadata = {
        "prefix_token_count": int(prefix_ids.shape[1]),
        "middle_token_count": int(ids["middle_ids"].shape[1]),
        "query_token_count": int(ids["query_ids"].shape[1]),
        "context_batch_size": int(context_ids.shape[0]),
        "padded_context_length": int(context_ids.shape[1]),
        "flattened_context_length": flattened_context_length,
        "final_cache_length": int(final_cache.key_cache[0].shape[-2]),
        "input_ids_sha256": {
            name: hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
            for name, tensor in ids.items()
        },
    }
    return final_cache, metadata


class UpstreamCaptureProxy:
    """Delegate to the real model while capturing upstream's decode boundary."""

    def __init__(self, model, eos_token_id: int):
        self.model = model
        self.device = model.device
        self.eos_token_id = eos_token_id
        self.call_count = 0
        self.decode_input_ids: torch.Tensor | None = None
        self.decode_cache: DynamicCache | None = None
        self.first_logits: torch.Tensor | None = None

    def __call__(self, *args, **kwargs):
        is_decode = self.call_count == 3
        if is_decode:
            input_ids = args[0] if args else kwargs["input_ids"]
            self.decode_input_ids = input_ids.detach().cpu().clone()
            self.decode_cache = clone_cache(kwargs["past_key_values"])
        output = self.model(*args, **kwargs)
        if is_decode:
            self.first_logits = output.logits[:, -1, :].detach().float().cpu()
            forced = output.logits.clone()
            forced[:, -1, :] = torch.finfo(forced.dtype).min
            forced[:, -1, self.eos_token_id] = 0
            output.logits = forced
        self.call_count += 1
        return output


def capture_upstream_cache(model, tokenizer, emb, blocks: list[str]) -> tuple[DynamicCache, torch.Tensor, torch.Tensor]:
    proxy = UpstreamCaptureProxy(model, tokenizer.eos_token_id)
    prefix, middle, *contexts, query = blocks
    upstream_gapemp(
        tokenizer,
        proxy,
        emb,
        prefix,
        middle,
        query,
        contexts,
        "ldsjmdy/Tulu3-Block-FT",
        1,
        1,
        None,
    )
    if (
        proxy.decode_cache is None
        or proxy.decode_input_ids is None
        or proxy.first_logits is None
    ):
        raise RuntimeError("failed to capture upstream GraphKV decode boundary")
    return proxy.decode_cache, proxy.decode_input_ids, proxy.first_logits


def cache_comparison(reference: DynamicCache, candidate: DynamicCache) -> dict[str, Any]:
    reference_shapes = [list(tensor.shape) for tensor in reference.key_cache]
    candidate_shapes = [list(tensor.shape) for tensor in candidate.key_cache]
    shapes_match = reference_shapes == candidate_shapes and [
        list(tensor.shape) for tensor in reference.value_cache
    ] == [list(tensor.shape) for tensor in candidate.value_cache]
    squared_difference = 0.0
    squared_reference = 0.0
    max_absolute_error = 0.0
    element_count = 0
    if shapes_match:
        for reference_tensor, candidate_tensor in zip(
            reference.key_cache + reference.value_cache,
            candidate.key_cache + candidate.value_cache,
            strict=True,
        ):
            difference = candidate_tensor.float() - reference_tensor.float()
            squared_difference += float(torch.sum(difference * difference).item())
            squared_reference += float(
                torch.sum(reference_tensor.float() ** 2).item()
            )
            max_absolute_error = max(
                max_absolute_error, float(torch.max(torch.abs(difference)).item())
            )
            element_count += difference.numel()
    relative_rms = (
        (squared_difference / max(element_count, 1)) ** 0.5
        / max((squared_reference / max(element_count, 1)) ** 0.5, 1e-6)
        if shapes_match
        else float("inf")
    )
    return {
        "shapes_match": shapes_match,
        "reference_key_shapes": reference_shapes,
        "candidate_key_shapes": candidate_shapes,
        "relative_rms_error": relative_rms,
        "max_absolute_error": max_absolute_error,
    }


def relative_logit_rms(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference = reference.float()
    candidate = candidate.float()
    return float(
        torch.sqrt(torch.mean((candidate - reference) ** 2))
        / torch.clamp(torch.sqrt(torch.mean(reference**2)), min=1e-6)
    )


def teacher_forced_comparison(
    model,
    reference_cache: DynamicCache,
    candidate_cache: DynamicCache,
    query_ids: torch.Tensor,
    eos_token_id: int,
    steps: int,
    captured_upstream_first_logits: torch.Tensor | None = None,
) -> dict[str, Any]:
    reference_past = clone_cache(reference_cache)
    candidate_past = clone_cache(candidate_cache)
    current = query_ids.to(model.device)
    records = []
    generated_ids = []
    captured_first_to_candidate_rms = None
    captured_first_to_recomputed_reference_rms = None
    with torch.inference_mode():
        for step in range(steps):
            reference_output = model(
                current, past_key_values=reference_past, use_cache=True
            )
            candidate_output = model(
                current, past_key_values=candidate_past, use_cache=True
            )
            reference_past = reference_output.past_key_values
            candidate_past = candidate_output.past_key_values
            reference_logits = reference_output.logits[:, -1, :]
            candidate_logits = candidate_output.logits[:, -1, :]
            if step == 0 and captured_upstream_first_logits is not None:
                captured_on_device = captured_upstream_first_logits.to(
                    device=reference_logits.device
                )
                captured_first_to_candidate_rms = relative_logit_rms(
                    captured_on_device, candidate_logits
                )
                captured_first_to_recomputed_reference_rms = relative_logit_rms(
                    captured_on_device, reference_logits
                )
            reference_token = int(torch.argmax(reference_logits, dim=-1).item())
            candidate_token = int(torch.argmax(candidate_logits, dim=-1).item())
            records.append(
                {
                    "step": step,
                    "relative_logit_rms_error": relative_logit_rms(
                        reference_logits, candidate_logits
                    ),
                    "reference_top_token": reference_token,
                    "candidate_top_token": candidate_token,
                    "top_token_match": reference_token == candidate_token,
                }
            )
            generated_ids.append(reference_token)
            current = torch.tensor([[reference_token]], device=model.device)
            if reference_token == eos_token_id:
                break
    return {
        "steps": records,
        "reference_prefix_token_ids": generated_ids,
        "captured_upstream_first_to_candidate_rms": captured_first_to_candidate_rms,
        "captured_upstream_first_to_recomputed_reference_rms": captured_first_to_recomputed_reference_rms,
    }


def greedy_generate(model, cache, query_ids, eos_token_id: int, max_new_tokens: int) -> list[int]:
    past = clone_cache(cache)
    current = query_ids.to(model.device)
    generated = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output = model(current, past_key_values=past, use_cache=True)
            past = output.past_key_values
            token = int(torch.argmax(output.logits[:, -1, :], dim=-1).item())
            generated.append(token)
            if token == eos_token_id:
                break
            current = torch.tensor([[token]], device=model.device)
    return generated


def select_two_per_hop(rows: list[dict], seed: str) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["no_of_hops"])].append(row)
    selected = []
    for hop in range(1, 6):
        ranked = sorted(
            grouped[hop],
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['_id']}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ranked[:2])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", default="20260909:morehop-engine-parity")
    parser.add_argument("--teacher-forced-steps", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.dataset)
    selected = select_two_per_hop(rows, args.seed)
    selected_ids = [row["_id"] for row in selected]
    if len(selected_ids) != 10 or len(set(selected_ids)) != 10:
        raise ValueError("parity selection must contain ten unique questions")

    local_files_only = os.path.isabs(args.model) or os.path.isdir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, local_files_only=local_files_only
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        local_files_only=local_files_only,
    )
    model.eval()
    config = AutoConfig.from_pretrained(
        args.model, local_files_only=local_files_only
    )
    emb = LlamaRotaryEmbedding(config=config).to(
        device=model.device, dtype=torch.float32
    )
    emb.eval()

    model_revision = Path(args.model).name if os.path.isdir(args.model) else getattr(config, "_commit_hash", None)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "selection_seed": args.seed,
        "selected_question_ids": selected_ids,
        "selected_hop_counts": {str(hop): 2 for hop in range(1, 6)},
        "model": "ldsjmdy/Tulu3-Block-FT",
        "model_path": str(Path(args.model).resolve()),
        "model_revision": model_revision,
        "torch_dtype": "bfloat16",
        "attention_backend": "flash_attention_2",
        "context_limit": CONTEXT_LIMIT,
        "context_reserve": CONTEXT_RESERVE,
        "teacher_forced_steps": args.teacher_forced_steps,
        "max_new_tokens": args.max_new_tokens,
        "upstream_function": "third_party/GraphKV/pcw.py::gapemp",
        "graphkv_submodule_commit": git_revision(GRAPHKV_ROOT),
        "project_commit": git_revision(ROOT),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    output_path = args.output_dir / "parity.jsonl"
    completed = []
    if output_path.exists():
        completed = [
            json.loads(line)
            for line in output_path.read_text().splitlines()
            if line.strip()
        ]
        if [row["_id"] for row in completed] != selected_ids[: len(completed)]:
            raise ValueError("existing parity output does not match frozen selection")

    with output_path.open("a", encoding="utf-8") as handle:
        for row in selected[len(completed) :]:
            started = time.perf_counter()
            blocks = upstream_blocks(row)
            reference_cache, reference_query_ids, captured_logits = capture_upstream_cache(
                model, tokenizer, emb, blocks
            )
            candidate_cache, token_metadata = prepare_project_full_cache(
                model, tokenizer, emb, blocks
            )
            expected_query_ids = tokenize_inputs(tokenizer, blocks)["query_ids"]
            input_ids_match = torch.equal(reference_query_ids, expected_query_ids)
            cache_metrics = cache_comparison(reference_cache, candidate_cache)
            teacher_forced = teacher_forced_comparison(
                model,
                reference_cache,
                candidate_cache,
                expected_query_ids,
                tokenizer.eos_token_id,
                args.teacher_forced_steps,
                captured_logits,
            )
            reference_tokens = greedy_generate(
                model,
                reference_cache,
                expected_query_ids,
                tokenizer.eos_token_id,
                args.max_new_tokens,
            )
            candidate_tokens = greedy_generate(
                model,
                candidate_cache,
                expected_query_ids,
                tokenizer.eos_token_id,
                args.max_new_tokens,
            )
            max_logit_error = max(
                step["relative_logit_rms_error"]
                for step in teacher_forced["steps"]
            )
            result = {
                "_id": row["_id"],
                "no_of_hops": int(row["no_of_hops"]),
                "prompt_hash": prompt_hash(blocks),
                "input_ids_match": input_ids_match,
                "token_metadata": token_metadata,
                "cache": cache_metrics,
                "teacher_forced": teacher_forced,
                "first_token_match": teacher_forced["steps"][0]["top_token_match"],
                "max_relative_logit_rms_error": max_logit_error,
                "complete_output_match": reference_tokens == candidate_tokens,
                "reference_token_ids": reference_tokens,
                "candidate_token_ids": candidate_tokens,
                "reference_generated": tokenizer.decode(
                    reference_tokens, skip_special_tokens=True
                ),
                "candidate_generated": tokenizer.decode(
                    candidate_tokens, skip_special_tokens=True
                ),
                "latency_seconds": time.perf_counter() - started,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "_id": result["_id"],
                        "hop": result["no_of_hops"],
                        "cache_rms": cache_metrics["relative_rms_error"],
                        "logit_rms": max_logit_error,
                        "first_token_match": result["first_token_match"],
                        "complete_output_match": result["complete_output_match"],
                    }
                ),
                flush=True,
            )

    results = [
        json.loads(line)
        for line in output_path.read_text().splitlines()
        if line.strip()
    ]
    passed = (
        len(results) == 10
        and all(row["input_ids_match"] for row in results)
        and all(row["cache"]["shapes_match"] for row in results)
        and all(row["first_token_match"] for row in results)
        and all(row["complete_output_match"] for row in results)
        and max(row["max_relative_logit_rms_error"] for row in results) <= 0.01
    )
    summary = {
        "status": "PASS" if passed else "FAIL",
        "questions": len(results),
        "input_id_matches": sum(row["input_ids_match"] for row in results),
        "cache_shape_matches": sum(row["cache"]["shapes_match"] for row in results),
        "first_token_matches": sum(row["first_token_match"] for row in results),
        "complete_output_matches": sum(
            row["complete_output_match"] for row in results
        ),
        "maximum_cache_relative_rms_error": max(
            row["cache"]["relative_rms_error"] for row in results
        ),
        "maximum_logit_relative_rms_error": max(
            row["max_relative_logit_rms_error"] for row in results
        ),
        "threshold": 0.01,
        "divergent_question_ids": [
            row["_id"]
            for row in results
            if not row["complete_output_match"]
            or not row["first_token_match"]
            or row["max_relative_logit_rms_error"] > 0.01
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    lines = [
        "# GraphKV Full engine parity",
        "",
        f"**Decision: {summary['status']}**",
        "",
        "The untouched upstream `pcw.gapemp` cache-construction path is compared with the project-owned Full implementation on two deterministic MoreHopQA questions per hop bucket.",
        "",
        "| Check | Result | Gate |",
        "|---|---:|---:|",
        f"| Identical token IDs | {summary['input_id_matches']}/10 | 10/10 |",
        f"| Matching cache shapes | {summary['cache_shape_matches']}/10 | 10/10 |",
        f"| Matching first token | {summary['first_token_matches']}/10 | 10/10 |",
        f"| Matching complete greedy output | {summary['complete_output_matches']}/10 | 10/10 |",
        f"| Maximum relative logit RMS | {summary['maximum_logit_relative_rms_error']:.8g} | ≤0.01 |",
        f"| Maximum relative cache RMS | {summary['maximum_cache_relative_rms_error']:.8g} | diagnostic |",
        "",
        f"Divergent QIDs: `{summary['divergent_question_ids']}`.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
