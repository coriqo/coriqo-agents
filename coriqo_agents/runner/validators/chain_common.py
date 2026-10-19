"""Shared machinery for the five `chain_*` multi-agent validators.

A chain validator does not read one trace. It reads a **declared roster** of
member agents plus whichever of their traces the runner actually collected,
and it must produce a verdict for the SYSTEM. That is a different shape from
every single-agent validator, and it introduces a failure mode none of
them have: a member that never ran, or whose trace never arrived, contributes
nothing to the evaluation — and "contributes nothing" is exactly how a clean
verdict gets manufactured out of missing evidence.

## Why the roster is the iteration domain, structurally

The obvious implementation iterates the traces that are present, collects
violations, and returns PASS when there are none. That implementation reports
PASS for a three-agent system of which only one agent was observed, and no
amount of review discipline stops the next chain validator from being written
the same way — it is the natural shape of the code.

So the incompleteness is not a check a validator performs. It is the loop.
`fold_over_roster` below takes the DECLARED roster and a per-member verdict
map, and walks the roster. A member the validator never produced a verdict for
cannot be skipped, because the traversal never looks at the verdict map's keys
— it looks at the roster's. The verdict map is only ever consulted by
`.get(agent_id)`, and a miss is a disclosed `INCONCLUSIVE` link with a reason,
never an absence.

A chain validator therefore cannot forget coverage. It can only produce a
wrong per-member verdict, which is an ordinary bug a test catches, rather
than a silent one that reads as assurance.

## Enumerating the good statuses

PROGRAM_TRACKER standing rule #8. `LINK_CLEAN_STATUSES` names the two link
statuses that are allowed to leave a chain verdict clean. Anything else —
including a status some future version adds — falls through
`chain_verdict`'s rank table to `INCONCLUSIVE` rather than being quietly
absorbed. There is deliberately no "everything not in the bad list is fine"
branch anywhere in this module.

Determinism (D9): every function here is a comparison over recorded trace
content. No LLM decides a link status or a chain verdict.
"""
from __future__ import annotations

from typing import Any, Callable

from ..contract import Severity, Status, ValidatorResult
from ..trace import EventType

#: Re-export, not a second definition — `trace.EventType.AGENT_MESSAGE` is
#: the one place the wire name lives, and its comment documents the event's
#: fields. Bound here only so the five chain validators can write
#: `cc.AGENT_MESSAGE` alongside the other helpers in this module.
AGENT_MESSAGE = EventType.AGENT_MESSAGE

#: The only link statuses that leave a chain verdict clean. Enumerated
#: positively — see the module docstring.
LINK_CLEAN_STATUSES: tuple[str, ...] = (Status.PASS, Status.NOT_APPLICABLE)

#: Worst-wins ranking for rolling per-link statuses up to one chain verdict.
#: Higher rank wins. A status absent from this table is not ranked at all;
#: `chain_verdict` maps it to INCONCLUSIVE rather than guessing where it
#: sits — the unknown-status case rule #8 exists for.
#:
#: FAIL outranks INCONCLUSIVE because a known violation is the headline an
#: examiner is owed first; the uncovered members are still enumerated in the
#: observations of every result regardless of which status won, so an
#: incomplete chain is never silently reported as merely FAIL-with-full-
#: coverage.
_STATUS_RANK: dict[str, int] = {
    Status.NOT_APPLICABLE: 0,
    Status.PASS: 1,
    Status.WARNING: 2,
    Status.INCONCLUSIVE: 3,
    Status.FAIL: 4,
    Status.ERROR: 5,
}

_NO_VERDICT_REASON = (
    "no verdict was produced for this member — its trace is missing from the "
    "execution, so nothing about this link has been evaluated"
)


class Link:
    """One roster member's contribution to a chain verdict.

    `status` is one of `contract.Status`'s six values; `reason` is the
    plain-language sentence an examiner reads. A Link is never constructed
    with a status outside Status.ALL — a typo becomes an error here rather
    than an unrankable status three layers later.
    """

    __slots__ = ("agent_id", "status", "reason", "evidence")

    def __init__(self, agent_id: str, status: str, reason: str, evidence: list | None = None) -> None:
        if status not in Status.ALL:
            raise ValueError(
                f"Link status {status!r} for agent {agent_id!r} is not one of {Status.ALL} — "
                "a chain link must carry one of the six chain-link statuses"
            )
        self.agent_id = agent_id
        self.status = status
        self.reason = reason
        self.evidence = evidence or []

    def to_dict(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "status": self.status, "reason": self.reason}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Link({self.agent_id!r}, {self.status!r}, {self.reason!r})"


def chain_input(context) -> dict[str, Any] | None:
    """`context.inputs["chain"]`, or None when this execution is not a
    multi-agent chain execution at all.

    None and an empty roster are deliberately different answers and the
    callers treat them differently: None means "this bundle was run against a
    single agent, there is no chain here to assure" (NOT_APPLICABLE), while a
    present-but-empty roster means "a chain execution declared no members",
    which is a system nobody has evaluated (INCONCLUSIVE). Collapsing the two
    would make a misconfigured chain execution read as an irrelevant one.
    """
    chain = context.inputs.get("chain")
    if not isinstance(chain, dict):
        return None
    return chain


def roster_of(chain: dict[str, Any]) -> list[dict[str, Any]]:
    """The DECLARED member list, in declaration order. This is the iteration
    domain for every chain validator — never `traces.keys()`."""
    members = chain.get("roster")
    if not isinstance(members, list):
        return []
    return [m for m in members if isinstance(m, dict) and m.get("agent_id")]


def roster_ids(chain: dict[str, Any]) -> list[str]:
    return [str(m["agent_id"]) for m in roster_of(chain)]


def member_by_id(chain: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(m["agent_id"]): m for m in roster_of(chain)}


def traces_of(chain: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """`{agent_id: trace}` for the members whose trace the runner actually
    collected. A roster member absent from this map has not been observed."""
    traces = chain.get("traces")
    if not isinstance(traces, dict):
        return {}
    return {str(k): v for k, v in traces.items() if isinstance(v, dict)}


def trace_events(trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(trace, dict):
        return []
    events = trace.get("events")
    return events if isinstance(events, list) else []


def messages_of(chain: dict[str, Any]) -> list[dict[str, Any]]:
    """Every AGENT_MESSAGE event across every COLLECTED member trace, each
    annotated with `_observed_in` (the agent whose trace carried it).

    Read off the member traces rather than from a separate top-level log so
    there is exactly one wire format for an inter-agent message. A hop
    recorded by both endpoints appears twice here; callers that care about
    hop identity de-duplicate on `message_id` (`unique_messages`).

    Messages carried by a trace whose owner is not on the roster are NOT
    filtered out here — `foreign_message_senders` reports them, because a
    message from outside the declared system is a finding, not noise to drop.
    """
    out: list[dict[str, Any]] = []
    for agent_id, trace in traces_of(chain).items():
        for event in trace_events(trace):
            if event.get("event_type") == AGENT_MESSAGE:
                annotated = dict(event)
                annotated["_observed_in"] = agent_id
                out.append(annotated)
    return out


def unique_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per hop. De-duplicated on `message_id` when present (both
    endpoints record the same hop); a message with no `message_id` is never
    merged with anything, since there is no id to establish that two events
    describe the same hop."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for message in messages:
        mid = message.get("message_id")
        if isinstance(mid, str) and mid:
            if mid in seen:
                continue
            seen.add(mid)
        out.append(message)
    return out


def hop(message: dict[str, Any]) -> tuple[str | None, str | None]:
    """`(from, to)` off one AGENT_MESSAGE, normalised to `str | None`."""
    sender = message.get("from")
    recipient = message.get("to")
    return (
        str(sender) if isinstance(sender, str) and sender else None,
        str(recipient) if isinstance(recipient, str) and recipient else None,
    )


def foreign_participants(chain: dict[str, Any]) -> list[str]:
    """Agent ids that appear as a message endpoint (or own a collected trace)
    but are not on the declared roster, sorted.

    The server rejects such a trace at ingest (`chain_service.
    reject_foreign_member`), so this should be empty in production. It is
    computed here anyway because the SDK runs in the customer's environment
    where nothing has rejected anything yet, and a validator that silently
    evaluated a system larger than the one under assurance would be reporting
    on a subject the campaign never scoped.
    """
    declared = set(roster_ids(chain))
    seen: set[str] = set(traces_of(chain)) - declared
    for message in messages_of(chain):
        sender, recipient = hop(message)
        for party in (sender, recipient):
            if party and party not in declared:
                seen.add(party)
    return sorted(seen)


def fold_over_roster(
    chain: dict[str, Any],
    verdicts: dict[str, Link],
    *,
    validator_id: str,
    version: str,
    clean_observation: str,
    fail_severity: str = Severity.HIGH,
) -> ValidatorResult:
    """Roll per-member verdicts up to one chain-level ValidatorResult.

    **The traversal is over `roster_of(chain)`, not over `verdicts`.** A
    declared member with no verdict yields an INCONCLUSIVE link carrying
    `_NO_VERDICT_REASON`. This is the structural guarantee described in the
    module docstring: incompleteness is produced by the loop, not detected by
    a check a caller could omit.

    Every result — clean or not — carries the coverage line, so an examiner
    reading a FAIL still learns that two of five members were never observed.
    """
    members = roster_of(chain)
    links: list[Link] = []
    for member in members:
        agent_id = str(member["agent_id"])
        link = verdicts.get(agent_id)
        if link is None:
            link = Link(agent_id, Status.INCONCLUSIVE, _NO_VERDICT_REASON)
        links.append(link)

    observed = set(traces_of(chain))
    uncovered = [str(m["agent_id"]) for m in members if str(m["agent_id"]) not in observed]
    foreign = foreign_participants(chain)

    if not members:
        return ValidatorResult(
            validator_id=validator_id, version=version,
            status=Status.INCONCLUSIVE,
            observations=[
                "this chain execution declares no member agents, so there is no system to "
                "assure — an empty roster is an unevaluated system, not a clean one",
            ],
        )

    status = chain_verdict([link.status for link in links])
    if foreign and status in LINK_CLEAN_STATUSES:
        # A message endpoint outside the declared roster means this execution
        # observed a system larger than the one the campaign scoped. Noting it
        # in the observations while still returning PASS would be the same
        # "disclosed but not acted on" gap the coverage rule exists to close.
        status = Status.INCONCLUSIVE
    observations = [
        f"{len(members)} declared member(s); {len(observed & set(roster_ids(chain)))} observed, "
        f"{len(uncovered)} with no trace in this execution",
    ]
    if uncovered:
        observations.append(
            "no trace was collected for: " + ", ".join(sorted(uncovered))
            + " — every link involving these members is INCONCLUSIVE, never inherited from a "
              "member that did pass"
        )
    if foreign:
        observations.append(
            "message endpoints outside the declared system: " + ", ".join(foreign)
            + " — these are not members of the subject under assurance"
        )
    for link in links:
        if link.status not in LINK_CLEAN_STATUSES or link.status == Status.NOT_APPLICABLE:
            observations.append(f"{link.agent_id}: {link.status} — {link.reason}")
    if status in LINK_CLEAN_STATUSES and not uncovered and not foreign:
        observations.append(clean_observation)

    # NOT_APPLICABLE is not counted as clean for scoring: an all-N/A chain
    # must not score 1.0 next to a genuinely all-PASS one.
    clean = sum(1 for link in links if link.status == Status.PASS)
    return ValidatorResult(
        validator_id=validator_id, version=version,
        status=status,
        score=(clean / len(links)) if links else None,
        severity=fail_severity if status in (Status.FAIL, Status.ERROR) else (
            Severity.MEDIUM if status == Status.WARNING else None
        ),
        observations=observations,
    )


def chain_verdict(link_statuses: list[str]) -> str:
    """Worst-wins rollup of link statuses to one chain status.

    An unranked status — one a future version adds without touching
    `_STATUS_RANK` — resolves to INCONCLUSIVE, never to the clean end of the
    scale. This is the positive-enumeration rule (#8) expressed as code
    rather than as a comment: there is no `else: return PASS` here.
    """
    if not link_statuses:
        return Status.INCONCLUSIVE
    if any(status not in _STATUS_RANK for status in link_statuses):
        return Status.INCONCLUSIVE
    worst = max(link_statuses, key=lambda s: _STATUS_RANK[s])
    if worst == Status.NOT_APPLICABLE:
        # Every link had nothing to say. That is NOT_APPLICABLE for the
        # chain too — honest, and distinguishable from PASS by an examiner.
        return Status.NOT_APPLICABLE
    return worst


def not_a_chain(validator_id: str, version: str) -> ValidatorResult:
    """The result for an execution that carries no chain input at all —
    a single-agent run of a bundle that happens to declare a chain
    validator."""
    return ValidatorResult(
        validator_id=validator_id, version=version,
        status=Status.NOT_APPLICABLE,
        observations=[
            "this execution carries no multi-agent chain input (inputs['chain']), so there is "
            "no inter-agent behaviour to evaluate",
        ],
    )


def declared_scope(member: dict[str, Any]) -> list[str] | None:
    """A member's mandated tool scope (`Agent.allowed_tools`), or None when
    the mandate declares none.

    None is NOT "unrestricted, therefore everything passes". Every caller
    turns None into an INCONCLUSIVE link: a mandate with no machine-readable
    scope gives a subset check nothing to compare against, and answering
    "⊆ succeeded" against an undeclared set is the same mistake as reading a
    missing trace as a clean one.

    **An explicit `[]` returns `[]`, not None.** An agent deliberately
    mandated to use zero tools is a real, checkable answer — every hop it
    delegates is outside its authority, which is a FAIL. Collapsing `[]` to
    None turns that FAIL into "no mandate declared, cannot tell", which is
    the single most permissive reading available.

    The first version of this function ended `return named or None` while
    `chain_service.load_roster`'s own comment promised the `[]`/None
    distinction was preserved end to end, and `chain_scope_monotonic.
    _scope_of` already kept `[]` as `[]` on the other side of the same
    subset check. A comment asserting an invariant the code does not enforce
    is this codebase's signature bug, and this was an instance of it in the
    code written to avoid it — caught in review, pinned by
    `test_a_locked_down_agent_delegating_anything_is_a_FAIL_not_unknown`.
    """
    tools = member.get("allowed_tools")
    if not isinstance(tools, list):
        return None
    return [str(t) for t in tools if isinstance(t, str) and t]


def link_for_missing_trace(agent_id: str, what: str) -> Link:
    """The INCONCLUSIVE link for a declared member whose trace never arrived.

    Callers that walk the roster themselves (rather than relying on
    fold_over_roster's default) use this so the reason text stays identical
    across the five validators.
    """
    return Link(
        agent_id, Status.INCONCLUSIVE,
        f"no trace was collected for this member, so {what} cannot be evaluated for it",
    )


def each_member(
    chain: dict[str, Any],
    fn: Callable[[dict[str, Any], dict[str, Any] | None], Link],
    *,
    what: str,
) -> dict[str, Link]:
    """Apply `fn(member, trace)` to every DECLARED member, and hold two
    properties for a member whose trace was never collected — both here, so
    that no chain validator author can get either of them wrong.

    `fn` is called for every roster member, but a member with no collected
    trace is handed an EMPTY trace rather than None. `fn` therefore never
    receives None and never has to remember the missing-trace case — the one
    case a per-member check is most likely to get wrong by returning PASS for
    it. What `fn` returns for such a member is then filtered:

    * **A FAIL is kept**, with the missing trace disclosed in its reason.
      Three of the five chain checks prove a member's violation from the
      OTHER side of a hop: the sender's trace records the payload it pushed
      and the roster supplies the recipient's mandate, so `execute()` can
      attribute a fully-evidenced FAIL to a member whose own trace never
      arrived. The first version of this function replaced that FAIL with the
      generic missing-trace INCONCLUSIVE, which made dropping one member's
      trace upload a way to launder a proven violation down to "we didn't
      look" — the exact inversion of what this function exists to prevent.
    * **Anything else becomes the missing-trace INCONCLUSIVE** — PASS,
      NOT_APPLICABLE, WARNING and INCONCLUSIVE alike. A clean answer computed
      over an absent trace evidences nothing, so it never leaves this
      function as one.
    """
    traces = traces_of(chain)
    out: dict[str, Link] = {}
    for member in roster_of(chain):
        agent_id = str(member["agent_id"])
        trace = traces.get(agent_id)
        if trace is not None:
            out[agent_id] = fn(member, trace)
            continue
        provisional = fn(member, {})
        if provisional.status == Status.FAIL:
            out[agent_id] = Link(
                agent_id, Status.FAIL,
                f"{provisional.reason} (no trace was collected for this member, so this rests "
                "entirely on evidence recorded by the other members of the system)",
                provisional.evidence,
            )
            continue
        out[agent_id] = link_for_missing_trace(agent_id, what)
    return out


def events_by_type(trace: dict[str, Any] | None, event_type: str) -> list[dict[str, Any]]:
    return [e for e in trace_events(trace) if e.get("event_type") == event_type]


__all__ = [
    "AGENT_MESSAGE",
    "EventType",
    "LINK_CLEAN_STATUSES",
    "Link",
    "chain_input",
    "chain_verdict",
    "declared_scope",
    "each_member",
    "events_by_type",
    "fold_over_roster",
    "foreign_participants",
    "hop",
    "link_for_missing_trace",
    "member_by_id",
    "messages_of",
    "not_a_chain",
    "roster_ids",
    "roster_of",
    "trace_events",
    "traces_of",
    "unique_messages",
]
