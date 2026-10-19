"""The mandate-check pipeline examples must not let a variable inject flags.

`ci/gitlab-mandate-check.yml` once built an argument string and expanded it
unquoted, so CORIQO_AGENT_ID="x --allow-unapproved" switched off the approval
requirement. This runs the real script block from each example under sh and
bash, with hostile values and a stub `coriqo-agents` that records exactly the
arguments it was given, and asserts every value arrives as ONE argument.

Run standalone (the SDK dir has a hyphen, so this is a script, not `-m`):

    python sdk/coriqo-agents/tests/test_mandate_check_ci_snippets.py
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

CI_DIR = Path(__file__).resolve().parent.parent / "ci"


def _gitlab_script() -> str:
    doc = yaml.safe_load((CI_DIR / "gitlab-mandate-check.yml").read_text())
    return doc["coriqo-mandate-check"]["script"][0]


def _github_script() -> str:
    doc = yaml.safe_load((CI_DIR / "github-mandate-check.yml").read_text())
    steps = doc["jobs"]["coriqo-mandate-check"]["steps"]
    return next(s["run"] for s in steps if s.get("name") == "Check the agent's mandate")


def _run(script: str, shell: str, env_vars: dict[str, str]) -> list[str]:
    """Run `script` with a stub coriqo-agents first on PATH; return its argv."""
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "argv.txt")
        stub = os.path.join(tmp, "coriqo-agents")
        with open(stub, "w") as fh:
            fh.write(f'#!/bin/sh\nfor a in "$@"; do printf \'%s\\n\' "$a"; done > "{out}"\n')
        os.chmod(stub, os.stat(stub).st_mode | stat.S_IXUSR)
        env = {"PATH": tmp + os.pathsep + os.environ["PATH"], **env_vars}
        subprocess.run([shell, "-c", script], env=env, check=True, capture_output=True)
        with open(out) as fh:
            return fh.read().split("\n")[:-1]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("  ok  " if cond else " FAIL ") + msg)
        if not cond:
            failures.append(msg)

    hostile = "x --allow-unapproved --mandate-file=/etc/passwd"
    cases = [
        ("gitlab", "sh", _gitlab_script(),
         lambda agent, mfile, allow: {"CORIQO_AGENT_ID": agent, "CORIQO_MANDATE_FILE": mfile, "CORIQO_ALLOW_UNAPPROVED": allow}),
        ("gitlab", "bash", _gitlab_script(),
         lambda agent, mfile, allow: {"CORIQO_AGENT_ID": agent, "CORIQO_MANDATE_FILE": mfile, "CORIQO_ALLOW_UNAPPROVED": allow}),
        ("github", "bash", _github_script(),
         lambda agent, mfile, allow: {"AGENT_ID": agent, "MANDATE_FILE": mfile, "ALLOW_UNAPPROVED": allow}),
    ]
    for name, shell, script, env_for in cases:
        label = f"{name}/{shell}"

        argv = _run(script, shell, env_for(hostile, "", "false"))
        check(argv == ["mandate", "check", f"--agent-id={hostile}"],
              f"{label}: a hostile agent id stays ONE argument (got {argv})")
        check("--allow-unapproved" not in argv, f"{label}: a flag hidden in the agent id is not a flag")

        argv = _run(script, shell, env_for("a1", "my mandate; rm -rf x.txt", "false"))
        check(argv == ["mandate", "check", "--agent-id=a1", "--mandate-file=my mandate; rm -rf x.txt"],
              f"{label}: a mandate path with spaces and a semicolon stays ONE argument (got {argv})")

        argv = _run(script, shell, env_for("--allow-unapproved", "-n", "false"))
        check(argv == ["mandate", "check", "--agent-id=--allow-unapproved", "--mandate-file=-n"],
              f"{label}: values starting with a dash are bound to their flag (got {argv})")

        argv = _run(script, shell, env_for("a1", "", "true"))
        check(argv == ["mandate", "check", "--agent-id=a1", "--allow-unapproved"],
              f"{label}: the documented opt-out still works when set to true (got {argv})")

        argv = _run(script, shell, env_for("a1", "", "TRUE; --allow-unapproved"))
        check("--allow-unapproved" not in argv,
              f"{label}: only the exact value true drops the approval requirement (got {argv})")

    if failures:
        print(f"\n{len(failures)} FAILED")
        return 1
    print("\nAll mandate check pipeline example tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
