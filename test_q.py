#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""Unit tests for qsched's sweep expansion. Run: ./test_q.py  (or via pytest)"""

import pytest

from q import expand_sweep, split_top_level


def test_quoted_value_is_single_variant() -> None:
    assert split_top_level('"96"') == ['"96"']


def test_plain_comma_sweep() -> None:
    assert split_top_level("0.1,0.01") == ["0.1", "0.01"]


def test_list_of_quoted_strings_vs_sweep_of_lists() -> None:
    assert split_top_level('["Day","key"]') == ['["Day","key"]']
    assert split_top_level('["Day","key"],["Day"]') == ['["Day","key"]', '["Day"]']


def test_quoted_comma_not_split() -> None:
    assert split_top_level('"a,b"') == ['"a,b"']
    assert split_top_level("'a,b',c") == ["'a,b'", "c"]


def test_nested_brackets() -> None:
    assert split_top_level("[[a,b],[c]],[[d]]") == ["[[a,b],[c]]", "[[d]]"]


def test_braces_and_parens() -> None:
    assert split_top_level("{a:1,b:2},{a:3}") == ["{a:1,b:2}", "{a:3}"]
    assert split_top_level("f(x,y),g(z)") == ["f(x,y)", "g(z)"]


def test_expand_cartesian_product() -> None:
    argv = ["python", "train.py", "model=a,b", "lr=0.1,0.01", "seed=1"]
    combos = expand_sweep(argv)
    assert len(combos) == 4
    assert ["python", "train.py", "model=a", "lr=0.1", "seed=1"] in combos
    assert ["python", "train.py", "model=b", "lr=0.01", "seed=1"] in combos


def test_expand_user_examples() -> None:
    argv = [
        "uv", "run", "python", "scripts/train_predict.py",
        'features_transform.grouping=["Day","key"],["Day"]',
        'training_window="96"',
    ]
    combos = expand_sweep(argv)
    assert len(combos) == 2
    assert combos[0][-2] == 'features_transform.grouping=["Day","key"]'
    assert combos[1][-2] == 'features_transform.grouping=["Day"]'
    assert all(c[-1] == 'training_window="96"' for c in combos)


def test_flags_and_non_overrides_untouched() -> None:
    argv = ["prog", "--config-name=a,b", "plainword", "+key=x,y"]
    combos = expand_sweep(argv)
    assert len(combos) == 2  # only +key expands
    assert all(c[1] == "--config-name=a,b" for c in combos)


def test_empty_variant_raises() -> None:
    with pytest.raises(ValueError):
        expand_sweep(["key=a,,b"])
    with pytest.raises(ValueError):
        expand_sweep(["key=a,"])


def test_no_expansion_is_single_job() -> None:
    assert expand_sweep(["python", "train.py", "model=a"]) == [["python", "train.py", "model=a"]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))