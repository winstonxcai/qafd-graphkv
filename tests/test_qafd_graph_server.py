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
sys.modules.setdefault("pcw_parallel", types.SimpleNamespace(gapemp_graph=None))

from src.eval.qafd_graph_server import CENTER_INSTRUCTION, sequential_prompt


def test_matched_sequential_prompt_contains_identical_graph_texts():
    prompt = sequential_prompt("prefix", "center", ["n1", "n2"], "query")

    assert prompt == "prefixn1n2" + CENTER_INSTRUCTION + "centerquery"
