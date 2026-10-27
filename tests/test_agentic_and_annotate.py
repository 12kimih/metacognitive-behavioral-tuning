import json

from mbt import agentic, annotate
from mbt import trajectory as tf


def test_the_agent_command_withholds_web_tools_and_reads_the_brief_from_outside_the_repo():
    cmd = agentic.agent_command(agentic.AgentSettings())
    assert "--disallowedTools WebSearch WebFetch" in cmd
    assert "--permission-mode bypassPermissions" in cmd and "--effort max" in cmd
    assert "< /tmp/mbt/brief.md" in cmd and "timeout -k 30 3600" in cmd


def test_collect_replays_a_history():
    texts = {"a": tf.render(tf.Trajectory("p", "Q", "x\ny", "A")),
             "b": tf.render(tf.Trajectory("p", "Q", "x\nNEW\ny", "A"))}
    log = "a\tbefore: the teacher trace as generated\nb\tcheckable-verification: after x -- ok\n"
    got = agentic.collect(log, texts.__getitem__)
    # position is the non-whitespace share before the insertion: "x" of "xy" = 0.5
    assert got["insertions"] == [{"behavior": "checkable-verification", "position": 0.5}]


def test_annotation_replies_are_parsed():
    codes = {c: {"present": 1 if c == "D2" else 0} for c in annotate.UNDESIRABLE_CODES}
    row = annotate.parse("undesirable", json.dumps({"codes": codes, "D1f": 0, "D8c": 1}))
    assert row["D2"] == 1 and row["D1"] == 0 and row["D8c"] == 1
    behaviours = {name: {"present": 1, "count": 2, "thirds": {"early": 1, "late": 1}}
                  for name in annotate.TARGET}
    row = annotate.parse("target", json.dumps({"behaviours": behaviours,
                                               "closing_checks": ["form", "bogus"]}))
    assert row["closing-decision#thirds"] == [1, 0, 1] and row["closing_checks"] == ["form"]
    assert annotate.parse("target", "not json") is None


def test_the_annotation_message_numbers_lines_and_gives_the_thirds():
    text = annotate.bundle("x", "Q", "r1\nr2\nr3", "A")
    msg = annotate.message("target", text)
    assert "L1: === x ===" in msg and "early: L" in msg and "CODEBOOK:" in msg


def test_an_unsuccessful_agent_cannot_supply_training_data():
    row = dict(id='p', problem='Q', reasoning='r', response='a')

    class Backend:
        name = 'test'

        def run_agent(self, **kwargs):
            return dict(exit_code=124, timed_out=True, file_text=kwargs['file_text'])

    got = agentic.run(row, Backend(), agentic.AgentSettings())
    assert got['edited'] is False and 'reasoning' not in got


def test_an_editor_cannot_change_the_problem_and_sft_uses_its_edited_response(tmp_path):
    from argparse import Namespace
    from mbt.cli import cmd_sft

    row = dict(id='p', problem='Q', reasoning='original reasoning', response='old answer')

    class Backend:
        name = 'test'

        def __init__(self, problem):
            self.problem = problem

        def run_agent(self, **kwargs):
            text = tf.to_text(tf.Trajectory('p', self.problem, 'edited reasoning', 'new answer'))
            return dict(exit_code=0, timed_out=False, file_text=text,
                        agent_stream=json.dumps({'type': 'result', 'is_error': False})+'\n')

    assert agentic.run(row, Backend('changed Q'), agentic.AgentSettings())['edited'] is False
    edited = agentic.run(row, Backend('Q'), agentic.AgentSettings())
    assert edited['edited'] is True
    source, output = tmp_path/'edits.jsonl', tmp_path/'sft.jsonl'
    source.write_text(json.dumps(edited)+'\n')
    cmd_sft(Namespace(input=str(source), output=str(output)))
    assistant = json.loads(output.read_text())['messages'][1]['content']
    assert 'edited reasoning' in assistant and assistant.endswith('new answer')
    assert 'old answer' not in assistant


def test_docker_cleanup_runs_when_setup_fails(tmp_path, monkeypatch):
    import pytest
    from mbt.sandbox import docker

    monkeypatch.setattr(docker.shutil, 'which', lambda name: '/usr/bin/docker')
    directory = tmp_path/'sandbox'
    directory.mkdir()
    monkeypatch.setattr(docker.tempfile, 'mkdtemp', lambda **kwargs: str(directory))
    settings = agentic.AgentSettings(api_key_env='MISSING_TEST_KEY')
    monkeypatch.delenv('MISSING_TEST_KEY', raising=False)
    with pytest.raises(SystemExit):
        docker.DockerBackend().run_agent(file_text='trace', brief='brief', settings=settings)
    assert not directory.exists()
