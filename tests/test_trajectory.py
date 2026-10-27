import pytest

from mbt import trajectory as tf


def test_a_file_parses_back_to_its_sections():
    t = tf.Trajectory("p-1", "Find x.", "Line one.\n\nLine two.", "The answer is 2.")
    assert tf.parse(tf.to_text(t)) == {"PROBLEM": "Find x.", "REASONING": "Line one.\n\nLine two.",
                                       "RESPONSE": "The answer is 2."}


def test_a_trace_holding_a_header_line_is_refused():
    t = tf.Trajectory("p-1", "Find x.", "===== RESPONSE =====", "2")
    with pytest.raises(tf.NotATrajectoryFile):
        tf.to_text(t)


def test_a_missing_section_is_an_error():
    with pytest.raises(tf.NotATrajectoryFile):
        tf.parse("===== PROBLEM =====\nx\n===== REASONING =====\ny\n")
