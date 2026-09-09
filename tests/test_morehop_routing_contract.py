import json

import pytest

from src.eval.morehop_routing_runner import _resume, _topology


def test_random_control_preserves_global_source_degree_distribution():
    out = _topology("random_k1_seed42", "question", ["a", "b", "c"])
    assert len({tuple(x) for x in out['neighbors']}) == 1
    assert out == _topology("random_k1_seed42", "question", ["a", "b", "c"])
    assert _topology("reversed_rank_k1", "q", ["a", "b", "c"])['neighbors'] == [[0]] * 3


def test_resume_rejects_mismatch_and_extra_rows(tmp_path):
    p = tmp_path/'rows.jsonl'
    row = {'_id': 'a', 'config_hash': 'c', 'prompt_hash': 'p', 'model_hash': 'm', 'code_hash': 's'}
    p.write_text(json.dumps(row)+'\n')
    assert len(_resume(p, ['a'], 'c', ['p'], 'm', 's')) == 1
    with pytest.raises(ValueError):
        _resume(p, ['a'], 'changed', ['p'], 'm', 's')
    p.write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError):
        _resume(p, ['a'], 'c', ['p'], 'm', 's')
