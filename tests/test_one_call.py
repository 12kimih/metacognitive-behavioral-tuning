import pytest

from mbt import one_call

REASONING = "First paragraph.\n\nSecond paragraph.\n\n\nThird paragraph."


def test_markers_follow_each_paragraph():
    marked, n = one_call.mark(REASONING)
    assert n == 3
    assert "First paragraph.\n[[#1]]" in marked and marked.endswith("Third paragraph.\n[[#3]]")


def test_inserting_and_removing_gives_the_original_back():
    ins = [one_call.Insertion(1, "logical-check-in-deduction", "Check A."),
           one_call.Insertion(3, "closing-decision", "Done.")]
    applied = one_call.apply(REASONING, ins)
    assert "First paragraph.\n\nCheck A.\n\nSecond" in applied.reasoning
    assert applied.reasoning.endswith("Third paragraph.\n\nDone.")
    assert one_call.original_kept(REASONING, applied)
    pos = one_call.positions(REASONING, applied)
    assert pos[0] == pytest.approx(len("Firstparagraph.") / len("".join(REASONING.split())))
    assert pos[1] == pytest.approx(1.0)


def test_an_insert_reply_is_read_strictly():
    reply = ("===== INSERTION =====\nAFTER: [[#2]]\nBEHAVIOUR: catch-and-correct\n"
             "Recomputing gives 4, not 5.\n===== NOTES =====\nnone\n")
    found, notes = one_call.parse_insert_reply(reply, 3)
    assert found == [one_call.Insertion(2, "catch-and-correct", "Recomputing gives 4, not 5.")]
    assert notes == "none"
    with pytest.raises(one_call.UnusableReply):
        one_call.parse_insert_reply(reply.replace("[[#2]]", "[[#9]]"), 3)
    with pytest.raises(one_call.UnusableReply):
        one_call.parse_insert_reply(reply.replace("catch-and-correct", "wait-marker"), 3)


def test_a_rewrite_reply_must_hold_both_sections_and_no_stand_in():
    ok = "===== REASONING =====\nr\n===== RESPONSE =====\na\n===== NOTES =====\nn\n"
    assert one_call.parse_rewrite_reply(ok)["REASONING"] == "r"
    with pytest.raises(one_call.UnusableReply):
        one_call.parse_rewrite_reply("The answer is 3.")
    with pytest.raises(one_call.UnusableReply):
        one_call.parse_rewrite_reply(ok.replace("\nr\n", "\n[rest of the reasoning omitted]\n"),
                                     original={"REASONING": "r", "RESPONSE": "a"})


def test_a_parseable_but_truncated_reply_is_retried():
    from types import SimpleNamespace
    from mbt import methods
    from mbt.client import ChatResult

    class Client:
        endpoint = SimpleNamespace(max_tokens=100)
        calls = 0

        def chat(self, *args, **kwargs):
            self.calls += 1
            return ChatResult(content='parseable', reasoning='', finish_reason=(
                'length' if self.calls == 1 else 'stop'))

    client, attempts = Client(), []
    assert methods._ask(client, 'Q', lambda text: text, attempts) == 'parseable'
    assert client.calls == 2 and attempts[0]['usable'] is False
