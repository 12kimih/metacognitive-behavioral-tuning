"""Interrupted writes and conversion failures preserve usable output."""

import json
import os
import subprocess
import sys
from argparse import Namespace

import pytest
from mbt import storage
from mbt.cli import cmd_sft, done_ids, run_rows


def test_resume_preserves_fragment_and_keeps_the_next_completed_result(tmp_path):
    output = tmp_path / "out.jsonl"
    fragment = b'{"id": "torn'
    rows = [{"id": "done"}, {"id": "new"}]
    with storage.locked(output):
        storage.bind_output(output, rows, {})
    output.write_bytes(b'{"id":"done"}\n' + fragment)
    calls = []

    def work(row):
        calls.append(row["id"])
        return row

    rows = [{"id": "done"}, {"id": "new"}]
    run_rows(rows, output, work, 1, None, context={})
    run_rows(rows, output, work, 1, None, context={})
    assert done_ids(output) == {"done", "new"}
    assert calls == ["new"]
    assert next((tmp_path / ".recovery").glob("*.tail")).read_bytes() == fragment


def test_complete_final_record_without_newline_is_kept(tmp_path):
    output = tmp_path / "out.jsonl"
    rows = [{"id": "done"}, {"id": "next"}]
    with storage.locked(output):
        storage.bind_output(output, rows, {})
    output.write_text('{"id":"done"}')
    run_rows(rows, output, dict, 1, None, context={})
    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == ["done", "next"]


def test_corrupt_interior_record_fails_before_work(tmp_path):
    output = tmp_path / "out.jsonl"
    original = '{"id":"done"}\ninvalid\n{"id":"last"}\n'
    with storage.locked(output):
        storage.bind_output(output, [{"id": "new"}], {})
    output.write_text(original)
    with pytest.raises(json.JSONDecodeError):
        run_rows([{"id": "new"}], output, lambda row: pytest.fail("must not call"), 1, None, context={})
    assert output.read_text() == original


@pytest.mark.parametrize("alias", ["same", "hardlink"])
def test_sft_refuses_input_output_alias_without_truncation(tmp_path, alias):
    source = tmp_path / "input.jsonl"
    original = '{"id":"one","problem":"Q","reasoning":"R","response":"A"}\n'
    source.write_text(original)
    output = source
    if alias == "hardlink":
        output = tmp_path / "alias.jsonl"
        os.link(source, output)
    with pytest.raises(ValueError, match="different files"):
        cmd_sft(Namespace(input=str(source), output=str(output)))
    assert source.read_text() == original


def test_sft_failure_leaves_previous_output_intact(tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "out.jsonl"
    source.write_text('{"id":"one","reasoning":"R"}\n')
    output.write_text("previous output\n")
    with pytest.raises(KeyError):
        cmd_sft(Namespace(input=str(source), output=str(output)))
    assert output.read_text() == "previous output\n"
    assert not list(tmp_path.glob(".out.jsonl.*"))


def test_second_process_is_refused_before_work(tmp_path):
    output = tmp_path / "out.jsonl"
    script = (
        "from pathlib import Path; from mbt.cli import run_rows; "
        "import sys; run_rows([{'id':'new'}],Path(sys.argv[1]),dict,1,None,context={})"
    )
    with storage.locked(output):
        result = subprocess.run([sys.executable, "-c", script, str(output)],
                                capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "exclusive output lease" in result.stderr
    assert not output.exists()


def test_failed_row_signals_failure_and_preserves_completed_rows(tmp_path):
    output = tmp_path / "out.jsonl"

    def work(row):
        if row["id"] == "bad":
            raise ValueError("offline failure")
        return row

    with pytest.raises(RuntimeError, match="1 rows failed"):
        run_rows([{"id": "good"}, {"id": "bad"}], output, work, 1, None, context={})
    assert done_ids(output) == {"good"}
    run_rows([{"id": "good"}, {"id": "bad"}], output, dict, 1, None, context={})
    assert done_ids(output) == {"good", "bad"}


def test_duplicate_ids_are_refused_before_creating_output(tmp_path):
    output = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="unique"):
        run_rows([{"id": "same"}, {"id": "same"}], output,
                 lambda row: pytest.fail("must not call"), 1, None, context={})
    assert not output.exists()


@pytest.mark.parametrize("change", ["input", "configuration"])
def test_changed_run_identity_refuses_work_and_preserves_output(tmp_path, change):
    output = tmp_path / "out.jsonl"
    rows = [{"id": "one", "problem": "original"}]
    context = {"command": "generate", "model": "original"}
    run_rows(rows, output, dict, 1, None, context=context)
    before = output.read_bytes()
    if change == "input":
        rows[0]["problem"] = "changed"
    else:
        context["model"] = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        run_rows(rows, output, lambda row: pytest.fail("must not call"), 1, None, context=context)
    assert output.read_bytes() == before


def test_legacy_output_is_not_assigned_invented_provenance(tmp_path):
    output = tmp_path / "out.jsonl"
    original = '{"id":"old"}\n'
    output.write_text(original)
    with pytest.raises(ValueError, match="no identity manifest"):
        run_rows([{"id": "old"}], output, dict, 1, None, context={})
    assert output.read_text() == original
    assert not output.with_name(output.name + ".manifest.json").exists()


def test_manifest_does_not_store_context_secrets_and_identity_survives_relocation(tmp_path):
    output = tmp_path / "out.jsonl"
    context = {"command": "insert", "secret": "example-private-credential"}
    rows = [{"id": "one"}]
    run_rows(rows, output, dict, 1, None, context=context)
    manifest = output.with_name(output.name + ".manifest.json")
    before = json.loads(manifest.read_text())
    assert "example-private-credential" not in manifest.read_text()
    moved = tmp_path / "moved.jsonl"
    output.rename(moved)
    manifest.rename(moved.with_name(moved.name + ".manifest.json"))
    run_rows(rows, moved, lambda row: pytest.fail("must not call"), 2, 0, context=context)
    after = json.loads(moved.with_name(moved.name + ".manifest.json").read_text())
    assert after == before


def test_missing_recorded_output_does_not_restart_under_the_old_identity(tmp_path):
    output = tmp_path / "out.jsonl"
    rows = [{"id": "one"}]
    run_rows(rows, output, dict, 1, None, context={})
    output.unlink()
    with pytest.raises(ValueError, match="output file is missing"):
        run_rows(rows, output, dict, 1, None, context={})


@pytest.mark.parametrize("command", ["generate", "rewrite", "insert", "agentic", "annotate"])
def test_cli_binds_every_model_command_before_calls(tmp_path, monkeypatch, command):
    from mbt import agentic, annotate, client, methods, sandbox
    from mbt.cli import main

    calls = []

    class OfflineClient:
        def __init__(self, endpoint, **kwargs):
            self.endpoint = endpoint

    def result(row):
        calls.append(row["id"])
        return {"id": row["id"], "edited": True}

    monkeypatch.setattr(client, "Client", OfflineClient)
    for name in ("generate", "rewrite", "insert"):
        monkeypatch.setattr(methods, name, lambda client, row: result(row))
    monkeypatch.setattr(sandbox, "backend", lambda *a, **k: object())
    monkeypatch.setattr(agentic, "run", lambda row, *a: result(row))
    monkeypatch.setattr(annotate, "annotate", lambda client, scheme, row: result(row))
    source, output = tmp_path / "input.jsonl", tmp_path / "out.jsonl"
    source.write_text('{"id":"one","problem":"Q","reasoning":"R","response":"A"}\n')
    args = [command, "--input", str(source), "--output", str(output), "--workers", "1"]
    if command == "annotate":
        args += ["--scheme", "target"]
    main(args)
    main(args)
    assert calls == ["one"]
    before = output.read_bytes()
    with pytest.raises(ValueError, match="identity changed"):
        main(args + ["--model", "different-model"])
    assert output.read_bytes() == before
    assert calls == ["one"]
