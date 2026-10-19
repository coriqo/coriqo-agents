"""`coriqo-agents prompts push` (MF2-P): hashes files by glob, never sends text, needs only the API key.

    PYTHONPATH="$PWD/sdk/coriqo-agents" python -m pytest sdk/coriqo-agents/tests/test_prompts_push.py -q
"""
from __future__ import annotations

import hashlib
import json

from coriqo_agents import prompts_push as pp


def test_collect_hashes_matching_files_with_relative_posix_paths(tmp_path):
    (tmp_path / "prompts" / "team").mkdir(parents=True)
    (tmp_path / "prompts" / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "prompts" / "team" / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "prompts" / "skip.txt").write_text("nope", encoding="utf-8")
    files = pp.collect(tmp_path, ["prompts/**/*.md"])
    assert [f["path"] for f in files] == ["prompts/a.md", "prompts/team/b.md"]
    assert files[0]["sha256"] == hashlib.sha256(b"alpha").hexdigest() and files[0]["chars"] == 5


def test_config_glob_and_default(tmp_path):
    (tmp_path / pp.CONFIG_NAME).write_text(json.dumps({"glob": "ai/*.txt", "repo": "bank/ai"}))
    assert pp.load_config(None, tmp_path) == {"glob": ["ai/*.txt"], "repo": "bank/ai"}
    assert pp.load_config(None, tmp_path / "missing") == {}


def test_dry_run_sends_nothing_and_never_prints_text(tmp_path, capsys, monkeypatch):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "a.md").write_text("secret instructions", encoding="utf-8")
    monkeypatch.delenv("CORIQO_API_KEY", raising=False)
    rc = pp.main(["--root", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "prompts/a.md" in out and "secret instructions" not in out and "Nothing was sent" in out


def test_no_match_is_an_error_unless_allowed(tmp_path, capsys):
    assert pp.main(["--root", str(tmp_path), "--dry-run"]) == 2
    assert pp.main(["--root", str(tmp_path), "--dry-run", "--allow-empty"]) == 0


def test_missing_settings_is_exit_2_not_a_traceback(tmp_path, monkeypatch, capsys):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "a.md").write_text("x", encoding="utf-8")
    for k in ("CORIQO_API_KEY", "CORIQO_BASE_URL", "CORIQO_TENANT_SLUG"):
        monkeypatch.delenv(k, raising=False)
    assert pp.main(["--root", str(tmp_path)]) == 2
    assert "CORIQO_API_KEY" in capsys.readouterr().err


def test_push_sends_hash_and_context_only(tmp_path, monkeypatch):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "a.md").write_text("body text", encoding="utf-8")
    monkeypatch.setenv("CORIQO_API_KEY", "k")
    monkeypatch.setenv("CORIQO_BASE_URL", "http://coriqo.test")
    monkeypatch.setenv("CORIQO_TENANT_SLUG", "acme")
    monkeypatch.setenv("GITHUB_SHA", "abc1234")
    monkeypatch.setenv("CORIQO_PROMPT_AUTHOR", "dana@bank.test")
    sent = []

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def push_prompt(self, body):
            sent.append(body)
            return {"created_prompt": True, "created_version": True}

    monkeypatch.setattr("coriqo_agents.client.CoriqoAgentClient", FakeClient)
    assert pp.main(["--root", str(tmp_path), "--approver", "omar"]) == 0
    body = sent[0]
    assert body["path"] == "prompts/a.md" and body["commit"] == "abc1234" and body["author"] == "dana@bank.test"
    assert body["approvers"] == ["omar"] and "text" not in body and "body text" not in json.dumps(body)
