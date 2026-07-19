import numpy as np
import pandas as pd

from nogan_synth import check_sum_relationship, find_sum_relationships


def _toy_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    return pd.DataFrame(
        {
            "a": a,
            "b": b,
            "total": a + b,
            "unrelated": rng.normal(size=n),
        }
    )


def test_check_sum_relationship_exact_match():
    df = _toy_df()
    result = check_sum_relationship(df, "total", ["a", "b"])
    assert result["match_frac"] == 1.0
    assert result["resid_max_abs"] < 1e-9


def test_check_sum_relationship_mismatch():
    df = _toy_df()
    result = check_sum_relationship(df, "unrelated", ["a", "b"])
    assert result["match_frac"] < 0.1


def test_find_sum_relationships_detects_known_pair():
    df = _toy_df()
    found = find_sum_relationships(df, max_parts=2)
    assert not found.empty
    top = found.iloc[0]
    assert top["target"] == "total"
    assert set(top["parts"]) == {"a", "b"}
    assert top["match_frac"] == 1.0


def test_find_sum_relationships_respects_min_match_frac():
    df = _toy_df()
    found = find_sum_relationships(df, max_parts=2, min_match_frac=1.01)
    assert found.empty


if __name__ == "__main__":
    test_check_sum_relationship_exact_match()
    test_check_sum_relationship_mismatch()
    test_find_sum_relationships_detects_known_pair()
    test_find_sum_relationships_respects_min_match_frac()
    print("ok")
