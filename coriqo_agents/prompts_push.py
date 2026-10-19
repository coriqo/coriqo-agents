"""`coriqo-agents prompts push`: tell Coriqo's prompt register which prompt files changed.

Run it from your CI after a merge. It hashes each file that matches the globs in your
config (default `prompts/**/*.md`) and posts the path, the sha256 and the commit, author,
pull request and reviewers it can read from the CI environment to
POST /api/v1/prompts/push. The prompt text never leaves your repository, and Coriqo holds
no credential for it. The only secret involved is the Coriqo API key you already use,
read from CORIQO_API_KEY.

Config (`.coriqo-prompts.json` in the repo root, or --config):

    {"glob": ["prompts/**/*.md"], "repo": "bank/ai-prompts"}

Exit 0 pushed (or nothing matched with --allow-empty), 1 Coriqo refused a file, 2 could not
run (missing settings, no matching files, network).
"""
from __future__ import annotations

import glob as _glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_GLOB = ["prompts/**/*.md"]
CONFIG_NAME = ".coriqo-prompts.json"


def load_config(path: str | None, root: Path) -> dict:
    cfg_path = Path(path) if path else root / CONFIG_NAME
    if not cfg_path.is_file():
        if path:
            raise ValueError(f"cannot read {path}")
        return {}
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {cfg_path}: {exc}") from None
    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_path} must hold a JSON object")
    globs = cfg.get("glob", DEFAULT_GLOB)
    if isinstance(globs, str):
        cfg["glob"] = [globs]
    elif not (isinstance(globs, list) and all(isinstance(g, str) for g in globs)):
        raise ValueError(f"'glob' in {cfg_path} must be a pattern or a list of patterns")
    return cfg


def collect(root: Path, patterns: list[str]) -> list[dict]:
    """[{path, sha256, chars}] for every file under `root` matching a pattern, sorted by path.
    The path is relative to `root`, with forward slashes, so it is the same on every machine."""
    seen: dict[str, Path] = {}
    for pattern in patterns:
        for hit in _glob.glob(str(root / pattern), recursive=True):
            p = Path(hit)
            if p.is_file():
                seen[p.relative_to(root).as_posix()] = p
    out = []
    for rel in sorted(seen):
        data = seen[rel].read_bytes()
        out.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(),
                    "chars": len(data.decode("utf-8", errors="replace"))})
    return out


def _git(root: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None


def ci_context(root: Path, env: dict[str, str]) -> dict:
    """What the CI environment and git can tell us. Anything unknown is left out."""
    commit = env.get("GITHUB_SHA") or env.get("CI_COMMIT_SHA") or _git(root, "rev-parse", "HEAD")
    author = env.get("CORIQO_PROMPT_AUTHOR") or _git(root, "log", "-1", "--format=%ae")
    pr_url = env.get("CORIQO_PROMPT_PR_URL")
    if not pr_url and env.get("GITHUB_SERVER_URL") and env.get("GITHUB_REPOSITORY") and env.get("CORIQO_PR_NUMBER"):
        pr_url = f"{env['GITHUB_SERVER_URL']}/{env['GITHUB_REPOSITORY']}/pull/{env['CORIQO_PR_NUMBER']}"
    approvers = [a.strip() for a in env.get("CORIQO_PROMPT_APPROVERS", "").split(",") if a.strip()]
    repo = env.get("GITHUB_REPOSITORY") or env.get("CI_PROJECT_PATH")
    return {k: v for k, v in {"commit": commit, "author": author, "pr_url": pr_url,
                              "approvers": approvers, "repo": repo}.items() if v}


def main(args: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="coriqo-agents prompts push")
    p.add_argument("--glob", action="append", help="files to send (repeatable); default from the config")
    p.add_argument("--config", help=f"config file (default ./{CONFIG_NAME})")
    p.add_argument("--root", default=".", help="repository root (default: current directory)")
    p.add_argument("--repo", help="repository name to show in Coriqo")
    p.add_argument("--pr-url", help="the pull request that made the change")
    p.add_argument("--ticket", help="the ticket or change request")
    p.add_argument("--approver", action="append", help="a reviewer who approved the change (repeatable)")
    p.add_argument("--deployed", action="store_true", help="mark these versions as deployed now")
    p.add_argument("--dry-run", action="store_true", help="print what would be sent, send nothing")
    p.add_argument("--allow-empty", action="store_true", help="exit 0 when no file matches")
    p.add_argument("--base-url", default=os.environ.get("CORIQO_BASE_URL"))
    p.add_argument("--tenant", default=os.environ.get("CORIQO_TENANT_SLUG"))
    ns = p.parse_args(args)

    root = Path(ns.root).resolve()
    try:
        cfg = load_config(ns.config, root)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    patterns = ns.glob or cfg.get("glob") or DEFAULT_GLOB
    files = collect(root, patterns)
    if not files:
        print(f"No file matches {', '.join(patterns)} under {root}. Check the 'glob' in {CONFIG_NAME}.",
              file=sys.stderr)
        return 0 if ns.allow_empty else 2

    ctx = ci_context(root, dict(os.environ))
    common = {**ctx, **{k: v for k, v in {
        "repo": ns.repo or cfg.get("repo"), "pr_url": ns.pr_url, "ticket": ns.ticket,
        "approvers": ns.approver or ctx.get("approvers")}.items() if v}}
    if ns.deployed:
        from datetime import datetime, timezone
        common["deployed_at"] = datetime.now(timezone.utc).isoformat()
    payloads = [{**common, **f} for f in files]

    if ns.dry_run:
        for body in payloads:
            print(f"{body['sha256'][:12]}  {body['path']}")
        print(f"{len(payloads)} file(s) would be sent. Nothing was sent.")
        return 0

    api_key = os.environ.get("CORIQO_API_KEY")
    if not (ns.base_url and api_key and ns.tenant):
        print("need CORIQO_BASE_URL, CORIQO_API_KEY and CORIQO_TENANT_SLUG "
              "(or --base-url and --tenant)", file=sys.stderr)
        return 2

    from .client import CoriqoAgentClient, CoriqoAPIError
    refused = 0
    try:
        with CoriqoAgentClient(ns.base_url, api_key, ns.tenant) as c:
            for body in payloads:
                try:
                    res = c.push_prompt(body)
                except CoriqoAPIError as exc:
                    refused += 1
                    print(f"REFUSED {body['path']}: {exc.detail}", file=sys.stderr)
                    continue
                state = ("new prompt" if res.get("created_prompt") else "new version" if res.get("created_version")
                         else "already recorded")
                print(f"{state:16} {body['path']}  {body['sha256'][:12]}")
    except Exception as exc:  # noqa: BLE001 - the pipeline must see "could not ask", not a traceback
        print(f"could not reach Coriqo: {exc}", file=sys.stderr)
        return 2
    return 1 if refused else 0
