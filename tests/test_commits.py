import pytest

from mbt import commits
from mbt import trajectory as tf


def version(reasoning: str) -> str:
    return tf.render(tf.Trajectory("p", "Q", reasoning, "A"))


def test_subjects_follow_the_grammar():
    assert commits.parse_subject("logical-check-in-deduction: at line 3 -- checked x != 0").kind \
        == "insertion"
    assert commits.parse_subject("subgoal-when-absent: none -- the opening plans").kind == "none"
    assert commits.parse_subject("fix: shorter sentences").kind == "fix"
    assert commits.parse_subject("before: the teacher trace as generated").kind == "before"


def test_replaying_commits_places_insertions_in_the_original():
    original = "aaaa\nbbbb\ncccc\ndddd"   # 16 non-whitespace characters
    v1 = "aaaa\nbbbb\nINSERTED\ncccc\ndddd"
    v2 = "aaaa\nbbbb\nINSERTED, polished\ncccc\ndddd"
    v3 = v2 + "\nCLOSING"
    events = commits.replay([
        ("s0", "before: x", version(original)),
        ("s1", "checkable-verification: after b -- substituted back", version(v1)),
        ("s2", "fix: polish", version(v2)),
        ("s3", "closing-decision: at the end -- settled", version(v3)),
    ])
    got = commits.insertion_positions(events)
    assert [b for b, _ in got] == ["checkable-verification", "closing-decision"]
    assert got[0][1] == pytest.approx(len("aaaabbbb") / len("aaaabbbbccccdddd"))
    assert got[1][1] == pytest.approx(1.0)
