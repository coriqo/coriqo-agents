"""`coriqo-agents` console entry point.

Thin dispatcher, not a second argument parser: `coriqo-agents runner run
...` and `coriqo-agents runner ci ...` hand off to
`coriqo_agents.runner`'s own `main()`, which already knows how to tell the
two apart (see that module's `_CI_SUBCOMMAND` dispatch) — this file only
strips the leading `runner` so `python -m coriqo_agents.runner ci ...` and
`coriqo-agents runner ci ...` run the identical code path.
"""
from __future__ import annotations

import sys


def _receipt_verify(args: list[str]) -> int:
    """Fetch (or load with --file) a receipt and verify it offline.
    Exit 0 verified against a pinned key, 1 a check failed or the receipt is
    incomplete, 2 not yet sealed or usage error, 3 valid but the key was not
    pinned (add --pubkey, or --allow-unpinned to accept the bundle's key)."""
    import argparse
    import json
    import os

    p = argparse.ArgumentParser(prog="coriqo-agents receipt verify")
    p.add_argument("event_hash", nargs="?")
    p.add_argument("--file", help="verify a saved bundle instead of fetching")
    p.add_argument("--pubkey", help="PEM file for a signing key you pinned yourself")
    p.add_argument("--wait", type=float, default=None, help="seconds to wait for sealing")
    p.add_argument("--base-url", default=os.environ.get("CORIQO_BASE_URL"))
    p.add_argument("--api-key", default=os.environ.get("CORIQO_API_KEY"),
                   help="discouraged: visible in `ps`; set CORIQO_API_KEY instead")
    p.add_argument("--allow-unpinned", action="store_true",
                   help="accept the key inside the bundle (proves consistency only)")
    p.add_argument("--tenant", default=os.environ.get("CORIQO_TENANT_SLUG"))
    ns = p.parse_args(args)
    from .receipts import verify_receipt

    if ns.file:
        try:
            with open(ns.file) as fh:
                bundle = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"cannot read {ns.file}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(bundle, dict):
            print(f"{ns.file} is not a receipt: expected a JSON object", file=sys.stderr)
            return 2
    elif ns.event_hash:
        if not (ns.base_url and ns.api_key and ns.tenant):
            print("need --base-url, --api-key, --tenant (or CORIQO_BASE_URL, "
                  "CORIQO_API_KEY, CORIQO_TENANT_SLUG)", file=sys.stderr)
            return 2
        from .client import CoriqoAgentClient
        with CoriqoAgentClient(ns.base_url, ns.api_key, ns.tenant) as c:
            bundle = c.get_receipt(ns.event_hash, wait=ns.wait)
        if bundle is None:
            print("pending: no checkpoint covers this event yet", file=sys.stderr)
            return 2
    else:
        p.error("give an event_hash or --file")
    pem = None
    if ns.pubkey:
        try:
            with open(ns.pubkey) as fh:
                pem = fh.read()
        except OSError as exc:
            print(f"cannot read {ns.pubkey}: {exc}", file=sys.stderr)
            return 2
    res = verify_receipt(bundle, pem, allow_unpinned=ns.allow_unpinned)
    for name, c in res["checks"].items():
        print(f"{name:16} {c['status']:12} {c['reason']}")
    if res["ok"]:
        print("RESULT: verified" + ("" if pem else " (key NOT pinned, --allow-unpinned)"))
        return 0 if (pem or ns.allow_unpinned) else 3
    unpinned_only = (
        pem is None and not any(c["status"] == "fail" for c in res["checks"].values())
        and all(res["checks"][n]["status"] == "pass"
                for n in ("leaf_inclusion", "root_in_sth", "signature")))
    if unpinned_only:
        print("RESULT: valid but not pinned: the key came from the bundle. "
              "Pass --pubkey, or --allow-unpinned to accept it.")
        return 3
    print("RESULT: not verified")
    return 1


def _mandate_check(args: list[str]) -> int:
    """Ask Coriqo whether an agent may be deployed. Put it in a release
    pipeline: the pipeline fails when this does.

    Exit 0 allowed, 1 blocked (including an agent Coriqo has never heard of),
    2 could not ask (missing settings, network, bad key), so a pipeline can
    tell "not approved" from "could not check"."""
    import argparse
    import os

    p = argparse.ArgumentParser(prog="coriqo-agents mandate check")
    p.add_argument("--agent-id", required=True, help="the agent's id in Coriqo")
    p.add_argument("--mandate-file", help="your repository's mandate file; its hash must match Coriqo's")
    p.add_argument("--allow-unapproved", action="store_true",
                   help="do not require a named person's approval (the mandate must still exist)")
    p.add_argument("--base-url", default=os.environ.get("CORIQO_BASE_URL"))
    p.add_argument("--api-key", default=os.environ.get("CORIQO_API_KEY"),
                   help="discouraged: visible in `ps`; set CORIQO_API_KEY instead")
    p.add_argument("--tenant", default=os.environ.get("CORIQO_TENANT_SLUG"))
    ns = p.parse_args(args)

    if not (ns.base_url and ns.api_key and ns.tenant):
        print("need --base-url, --api-key, --tenant (or CORIQO_BASE_URL, "
              "CORIQO_API_KEY, CORIQO_TENANT_SLUG)", file=sys.stderr)
        return 2

    claimed = None
    if ns.mandate_file:
        from .hashing import mandate_sha256
        try:
            with open(ns.mandate_file, encoding="utf-8") as fh:
                claimed = mandate_sha256(fh.read())
        except (OSError, UnicodeDecodeError) as exc:
            print(f"cannot read {ns.mandate_file}: {exc}", file=sys.stderr)
            return 2

    from .client import CoriqoAgentClient, CoriqoAPIError
    try:
        with CoriqoAgentClient(ns.base_url, ns.api_key, ns.tenant) as c:
            res = c.deploy_check(ns.agent_id, mandate_sha256=claimed,
                                 require_approved=not ns.allow_unapproved)
    except CoriqoAPIError as exc:
        if exc.status_code == 404:
            print(f"BLOCKED: Coriqo has no agent with id {ns.agent_id}. "
                  "Register it and get its mandate approved before it ships.")
            return 1
        print(f"could not check: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # network down, DNS, TLS: the pipeline must see "could not ask"
        print(f"could not reach Coriqo: {exc}", file=sys.stderr)
        return 2

    for chk in res.get("checks", []):
        print(f"{chk['id']:14} {'pass' if chk['passed'] else 'FAIL':5} {chk['detail']}")
    for note in res.get("notes", []):
        print(f"note: {note}")
    if res.get("allowed"):
        print(f"ALLOWED: {res['agent']['name']} may be deployed.")
        return 0
    print(f"BLOCKED: {res['agent']['name']} may not be deployed yet. Fix it at {res['fix_url']}")
    return 1


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    if raw_argv[:2] == ["receipt", "verify"]:
        return _receipt_verify(raw_argv[2:])
    if raw_argv[:2] == ["mandate", "check"]:
        return _mandate_check(raw_argv[2:])
    if raw_argv[:2] == ["prompts", "push"]:
        from .prompts_push import main as prompts_push
        return prompts_push(raw_argv[2:])
    if raw_argv[:1] != ["runner"]:
        print(
            "usage: coriqo-agents runner <run|ci|watch|verify|bench|conformance> [options]\n"
            "       coriqo-agents receipt verify <event_hash> | --file bundle.json\n"
            "       coriqo-agents mandate check --agent-id ID [--mandate-file PATH]\n"
            "       coriqo-agents prompts push [--glob PATTERN] [--dry-run]",
            file=sys.stderr,
        )
        return 1
    from .runner.__main__ import main as runner_main
    return runner_main(raw_argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
