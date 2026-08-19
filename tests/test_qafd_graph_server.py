import sys
import types

sys.modules.setdefault("torch", types.SimpleNamespace())
sys.modules.setdefault("flask", types.SimpleNamespace(Flask=lambda _: None, request=None))
sys.modules.setdefault(
    "transformers",
    types.SimpleNamespace(AutoConfig=None, AutoModelForCausalLM=None, AutoTokenizer=None),
)
sys.modules.setdefault(
    "transformers.models.llama.modeling_llama",
    types.SimpleNamespace(LlamaRotaryEmbedding=None),
)
sys.modules.setdefault(
    "pcw_parallel",
    types.SimpleNamespace(gapemp_graph=None, gapemp_graph_batch=None),
)

from src.eval.qafd_graph_server import (
    BATCH_CENTER_INSTRUCTION,
    CENTER_INSTRUCTION,
    sequential_batch_prompt,
    sequential_prompt,
)


def test_matched_sequential_prompt_contains_identical_graph_texts():
    prompt = sequential_prompt("prefix", "center", ["n1", "n2"], "query")

    assert prompt == "prefixn1n2" + CENTER_INSTRUCTION + "centerquery"


def test_matched_batch_prompt_preserves_star_groups():
    prompt = sequential_batch_prompt(
        "prefix", ["c1", "c2"], [["n1", "n2"], ["n3"]], "query"
    )

    assert prompt == (
        "prefixn1n2"
        + BATCH_CENTER_INSTRUCTION
        + "c1n3"
        + BATCH_CENTER_INSTRUCTION
        + "c2query"
    )
