"""The final search order blends the vector search and the reranker. (No model calls.)

    .venv/bin/python -m pytest -v tests/test_maya_blend.py
"""

from maya.retrieval import blend

# Four candidates in the vector search's order: "a" was closest.
CANDIDATES = [{"id": name} for name in "abcd"]


def order(rerank_scores: dict) -> list[str]:
    return [c["id"] for c, _ in blend(CANDIDATES, rerank_scores)]


def test_both_stages_agree():
    assert order({0: 0.9, 1: 0.6, 2: 0.3, 3: 0.1}) == ["a", "b", "c", "d"]


def test_the_vector_searchs_first_survives_a_low_rerank():
    # The promotion case: "a" is 1st by vector search but the reranker scores it
    # below "d". Alone, the reranker would put "a" 2nd; blended, a = (0.4 + 1.0) / 2
    # = 0.70 beats d = (0.95 + 0.25) / 2 = 0.60.
    assert order({0: 0.4, 1: 0.2, 2: 0.3, 3: 0.95}) == ["a", "d", "b", "c"]


def test_a_strong_rerank_still_lifts_a_late_candidate():
    assert order({0: 0.1, 1: 0.1, 2: 0.1, 3: 1.0})[0] == "d"


def test_a_candidate_the_reranker_skipped_keeps_its_position_score():
    scores = dict((c["id"], s) for c, s in blend(CANDIDATES, {}))
    assert scores == {"a": 0.5, "b": 0.375, "c": 0.25, "d": 0.125}
