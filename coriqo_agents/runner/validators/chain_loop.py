"""Multi-agent chain validator 5/5: message cycles in the chain.

Agents that hand work back and forth burn budget, amplify a single bad
instruction, and make a delegation chain unreadable. The spec calls for a
WARNING, not a FAIL, and this validator never returns FAIL: a cycle is a
design smell a human should look at, not a control violation Coriqo is
entitled to call.

What is measured: the directed message graph over declared members. A
strongly-connected component of two or more members is a cycle. The WARNING
fires when a cycle exists AND the messages traversing it exceed
`chain_loop_max_messages` (default 4) — a single A→B→A handback is a normal
delegate-and-return, while eleven of them is a loop.

A member that appears nowhere in the message graph is NOT_APPLICABLE, not
PASS: there were no edges touching it to detect a cycle in, so a clean
verdict for it would be a claim with nothing behind it.

Coverage still applies. A member with no collected trace is INCONCLUSIVE:
half the message graph missing means the cycles found are a lower bound, and
"no cycle detected in the half we can see" is not "no cycle".

Determinism (D9): graph traversal over recorded edges.
"""
from __future__ import annotations

from typing import Any

from ..contract import Status, Validator, ValidatorContext, ValidatorResult
from . import chain_common as cc

DEFAULT_MAX_CYCLE_MESSAGES = 4


class ChainLoopValidator(Validator):
    validator_id = "chain_loop"
    version = "1.0.0"

    def execute(self, context: ValidatorContext) -> Any:
        chain = cc.chain_input(context)
        if chain is None:
            return {"chain": None}
        raw = context.configuration.get("chain_loop_max_messages")
        try:
            limit = int(raw) if raw is not None else DEFAULT_MAX_CYCLE_MESSAGES
        except (TypeError, ValueError):
            limit = DEFAULT_MAX_CYCLE_MESSAGES
        return {"chain": chain, "limit": max(0, limit)}

    def evaluate(self, result: Any) -> ValidatorResult:
        chain = result["chain"]
        if chain is None:
            return cc.not_a_chain(self.validator_id, self.version)

        limit = result["limit"]
        members = cc.member_by_id(chain)
        messages = cc.unique_messages(cc.messages_of(chain))

        edges: dict[str, set[str]] = {}
        edge_traffic: dict[tuple[str, str], int] = {}
        participants: set[str] = set()
        for message in messages:
            sender, recipient = cc.hop(message)
            if sender is None or recipient is None or sender not in members or recipient not in members:
                continue
            edges.setdefault(sender, set()).add(recipient)
            edge_traffic[(sender, recipient)] = edge_traffic.get((sender, recipient), 0) + 1
            participants.add(sender)
            participants.add(recipient)

        components = [c for c in _strongly_connected(cc.roster_ids(chain), edges) if _is_cycle(c, edges)]
        looping = {a for component in components for a in component}
        # Traffic is counted PER COMPONENT: an edge leaving component 1 for a
        # member of component 2 traverses neither cycle, and counting it
        # against component 1 could push a quiet cycle over the threshold on
        # the strength of messages that never went round it.
        traffic_by_component: list[tuple[set[str], int]] = [
            (component, sum(n for (s, r), n in edge_traffic.items()
                            if s in component and r in component))
            for component in components
        ]

        def judge(member: dict, trace: dict) -> cc.Link:
            agent_id = str(member["agent_id"])
            if agent_id not in participants:
                # NOT_APPLICABLE, not PASS. A member that neither sent nor
                # received a single message inside the declared system has no
                # edges for cycle detection to have looked at, and "no cycle
                # was found in a graph this member does not appear in" is not
                # a fact about this member. `chain_scope_monotonic` and
                # `chain_data_boundary` already return NOT_APPLICABLE for
                # their equivalent no-hops case; this validator returned PASS,
                # which meant an execution whose inter-agent message log was
                # empty scored every member clean on a check that ran zero
                # comparisons.
                return cc.Link(agent_id, Status.NOT_APPLICABLE,
                               "this member neither sent nor received an inter-agent message in "
                               "this execution, so it is in no part of the message graph")
            if agent_id not in looping:
                return cc.Link(agent_id, Status.PASS,
                               "this member exchanged inter-agent messages but is not part of any "
                               "message cycle")
            members_of, total = next(
                (c, n) for c, n in traffic_by_component if agent_id in c)
            component = sorted(members_of)
            if total > limit:
                return cc.Link(agent_id, Status.WARNING,
                               f"part of a message cycle {' -> '.join(component)} carrying "
                               f"{total} message(s), above the {limit}-message threshold")
            return cc.Link(agent_id, Status.PASS,
                           f"part of a message cycle {' -> '.join(component)}, but it carries only "
                           f"{total} message(s) — a delegate-and-return, not a loop")

        verdicts = cc.each_member(chain, judge, what="whether it takes part in a message cycle")
        return cc.fold_over_roster(
            chain, verdicts,
            validator_id=self.validator_id, version=self.version,
            clean_observation=(
                "no inter-agent message cycle above the threshold across the declared members"),
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        return []


def _strongly_connected(nodes: list[str], edges: dict[str, set[str]]) -> list[set[str]]:
    """Tarjan's SCC, iterative so a long chain cannot blow the stack.

    Returns every component, including singletons — `_is_cycle` filters.
    """
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: list[set[str]] = []
    counter = 0

    for root in nodes:
        if root in index:
            continue
        work: list[tuple[str, list[str]]] = [(root, sorted(edges.get(root, ())))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, pending = work[-1]
            if pending:
                nxt = pending.pop(0)
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, sorted(edges.get(nxt, ()))))
                elif nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: set[str] = set()
                while True:
                    popped = stack.pop()
                    on_stack.discard(popped)
                    component.add(popped)
                    if popped == node:
                        break
                out.append(component)
    return out


def _is_cycle(component: set[str], edges: dict[str, set[str]]) -> bool:
    """A component is a cycle if it has two or more members, or one member
    that messages itself."""
    if len(component) > 1:
        return True
    only = next(iter(component))
    return only in edges.get(only, ())
