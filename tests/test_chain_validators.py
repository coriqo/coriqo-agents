"""The five multi-agent chain validators plus chain_common's
structural coverage guarantee.

Pure stdlib `unittest`, same convention as test_mcp_validators.py, and every
assertion is a real `assertEqual`/`assertIn` that fails the process — not a
print-and-record helper. Runnable standalone (`python test_chain_validators.py`)
and under pytest.

The class of bug this file exists to pin: a chain whose members were not all
observed must never read as clean. Half of these tests are that one claim,
approached from a different direction each time.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from coriqo_agents.runner.contract import Status, ValidatorContext
from coriqo_agents.runner.trace import EventType
from coriqo_agents.runner.validators import VALIDATOR_REGISTRY
from coriqo_agents.runner.validators import chain_common as cc
from coriqo_agents.runner.validators.chain_data_boundary import ChainDataBoundaryValidator
from coriqo_agents.runner.validators.chain_identity_continuity import (
    ChainIdentityContinuityValidator,
)
from coriqo_agents.runner.validators.chain_loop import ChainLoopValidator
from coriqo_agents.runner.validators.chain_scope_monotonic import ChainScopeMonotonicValidator
from coriqo_agents.runner.validators.chain_shutdown_propagation import (
    ChainShutdownPropagationValidator,
)

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)

CHAIN_VALIDATOR_IDS = (
    "chain_scope_monotonic", "chain_data_boundary", "chain_shutdown_propagation",
    "chain_identity_continuity", "chain_loop",
)


def ts(seconds: float) -> str:
    return (BASE + timedelta(seconds=seconds)).isoformat()


def event(event_id, event_type, at=0.0, **kw):
    return {"event_id": event_id, "event_type": event_type, "timestamp": ts(at), **kw}


def message(mid, sender, recipient, at=0.0, **kw):
    return event(mid, EventType.AGENT_MESSAGE, at, message_id=mid, **{"from": sender, "to": recipient}, **kw)


def trace(agent_id, events):
    return {"trace_id": f"tr_{agent_id}", "agent_id": agent_id, "agent_version": "1.0", "events": events}


def member(agent_id, *, allowed_tools=None, role=None, principal_ref=None):
    entry = {"agent_id": agent_id, "name": agent_id}
    if allowed_tools is not None:
        entry["allowed_tools"] = allowed_tools
    if role is not None:
        entry["role"] = role
    if principal_ref is not None:
        entry["principal_ref"] = principal_ref
    return entry


def run(validator, chain, configuration=None):
    context = ValidatorContext(
        execution_id="exec-1", agent_model_id="sys-1", agent_version_ref="1.0",
        environment="STAGING", configuration=configuration or {},
        inputs={} if chain is None else {"chain": chain},
    )
    return validator.evaluate(validator.execute(context))


# ---------------------------------------------------------------------------
# chain_common — the structural coverage guarantee
# ---------------------------------------------------------------------------

class ChainCommonTest(unittest.TestCase):
    def test_fold_over_roster_iterates_the_roster_not_the_verdicts(self):
        """The load-bearing claim of this packet. A verdict map that says
        PASS for the only member it names must not produce a clean chain
        when the roster declares three."""
        chain = {"roster": [member("a"), member("b"), member("c")],
                 "traces": {"a": trace("a", [])}}
        verdicts = {"a": cc.Link("a", Status.PASS, "fine")}
        result = cc.fold_over_roster(
            chain, verdicts, validator_id="x", version="1.0", clean_observation="all good")
        self.assertEqual(result.status, Status.INCONCLUSIVE)
        self.assertNotIn("all good", " ".join(str(o) for o in result.observations))
        joined = " ".join(str(o) for o in result.observations)
        self.assertIn("b", joined)
        self.assertIn("c", joined)

    def test_a_member_with_a_verdict_but_no_trace_still_reports_uncovered(self):
        """Even a caller that hand-rolls a PASS for an unobserved member
        cannot hide the coverage gap: the coverage line is computed from
        `traces`, independently of the verdict map."""
        chain = {"roster": [member("a"), member("b")], "traces": {"a": trace("a", [])}}
        verdicts = {"a": cc.Link("a", Status.PASS, "fine"), "b": cc.Link("b", Status.PASS, "fine")}
        result = cc.fold_over_roster(
            chain, verdicts, validator_id="x", version="1.0", clean_observation="all good")
        joined = " ".join(str(o) for o in result.observations)
        self.assertIn("no trace was collected for: b", joined)
        self.assertNotIn("all good", joined)

    def test_empty_roster_is_inconclusive_not_not_applicable(self):
        result = cc.fold_over_roster(
            {"roster": [], "traces": {}}, {}, validator_id="x", version="1.0",
            clean_observation="all good")
        self.assertEqual(result.status, Status.INCONCLUSIVE)

    def test_chain_verdict_maps_an_unknown_status_to_inconclusive(self):
        """Rule #8 as code: a status this module has never heard of must not
        be absorbed into the clean end of the scale."""
        self.assertEqual(cc.chain_verdict(["PASS", "SOMETHING_NEW"]), Status.INCONCLUSIVE)

    def test_chain_verdict_all_not_applicable_is_not_pass(self):
        self.assertEqual(cc.chain_verdict(["NOT_APPLICABLE", "NOT_APPLICABLE"]),
                         Status.NOT_APPLICABLE)

    def test_chain_verdict_fail_outranks_inconclusive_but_coverage_is_still_disclosed(self):
        self.assertEqual(cc.chain_verdict(["FAIL", "INCONCLUSIVE"]), Status.FAIL)
        chain = {"roster": [member("a"), member("b")], "traces": {"a": trace("a", [])}}
        result = cc.fold_over_roster(
            chain, {"a": cc.Link("a", Status.FAIL, "bad")}, validator_id="x", version="1.0",
            clean_observation="all good")
        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("no trace was collected for: b",
                      " ".join(str(o) for o in result.observations))

    def test_link_rejects_a_status_outside_the_six(self):
        with self.assertRaises(ValueError):
            cc.Link("a", "MOSTLY_FINE", "reason")

    def test_each_member_never_hands_the_callback_a_missing_trace(self):
        seen = []

        def fn(m, t):
            seen.append((m["agent_id"], t))
            return cc.Link(m["agent_id"], Status.PASS, "ok")

        chain = {"roster": [member("a"), member("b")], "traces": {"a": trace("a", [])}}
        verdicts = cc.each_member(chain, fn, what="anything")
        # `b` has no trace: the callback is still called (so it never has to
        # remember the case), but it is handed an empty trace, never None, and
        # its clean answer does not survive.
        self.assertEqual([s[0] for s in seen], ["a", "b"])
        self.assertEqual(dict(seen)["b"], {})
        self.assertEqual(verdicts["b"].status, Status.INCONCLUSIVE)

    def test_each_member_downgrades_every_clean_status_computed_over_a_missing_trace(self):
        for status in (Status.PASS, Status.NOT_APPLICABLE, Status.WARNING, Status.INCONCLUSIVE):
            with self.subTest(status):
                chain = {"roster": [member("a")], "traces": {}}
                verdicts = cc.each_member(
                    chain, lambda m, t: cc.Link(m["agent_id"], status, "computed over nothing"),
                    what="anything")
                self.assertEqual(verdicts["a"].status, Status.INCONCLUSIVE)
                self.assertIn("no trace was collected", verdicts["a"].reason)

    def test_each_member_keeps_a_fail_proven_without_the_members_own_trace(self):
        """The laundering vector: dropping one member's trace upload must not
        turn a violation the OTHER members' traces already prove into a
        generic 'we didn't look'."""
        chain = {"roster": [member("a")], "traces": {}}
        verdicts = cc.each_member(
            chain,
            lambda m, t: cc.Link(m["agent_id"], Status.FAIL, "held tool data outside its mandate",
                                 [{"record": 1}]),
            what="anything")
        self.assertEqual(verdicts["a"].status, Status.FAIL)
        self.assertIn("held tool data outside its mandate", verdicts["a"].reason)
        # the absence is disclosed alongside the FAIL rather than replacing it
        self.assertIn("no trace was collected for this member", verdicts["a"].reason)
        self.assertEqual(verdicts["a"].evidence, [{"record": 1}])

    def test_absent_chain_input_is_not_applicable_present_but_empty_is_not(self):
        for vid in CHAIN_VALIDATOR_IDS:
            with self.subTest(vid):
                self.assertEqual(run(VALIDATOR_REGISTRY[vid](), None).status,
                                 Status.NOT_APPLICABLE)
                self.assertEqual(run(VALIDATOR_REGISTRY[vid](), {"roster": [], "traces": {}}).status,
                                 Status.INCONCLUSIVE)

    def test_every_chain_validator_is_inconclusive_when_one_member_is_unobserved(self):
        """The cross-cutting version of the packet's central claim: no chain
        validator, whatever its own logic, returns a clean status for a
        roster with an unobserved member."""
        chain = {
            "roster": [member("a", allowed_tools=["t"], role="orchestrator", principal_ref="agent:a"),
                       member("b", allowed_tools=["t"], principal_ref="agent:b")],
            "traces": {"a": trace("a", [
                event("s1", EventType.SHUTDOWN_SIGNAL, 1),
                event("h1", EventType.EXECUTION_HALTED, 1.1),
                event("c1", EventType.TOOL_CALL, 0.5, tool="t", on_behalf_of=["agent:a", "user:jo"]),
                message("m1", "a", "b", 0.6, delegated_scope=["t"]),
            ])},
        }
        for vid in CHAIN_VALIDATOR_IDS:
            with self.subTest(vid):
                result = run(VALIDATOR_REGISTRY[vid](), chain)
                self.assertEqual(
                    result.status, Status.INCONCLUSIVE,
                    f"{vid} returned {result.status} for a chain whose member b was never observed")

    def test_unique_messages_dedupes_on_message_id_only(self):
        both = [message("m1", "a", "b", 1), message("m1", "a", "b", 1), message(None, "a", "b", 2)]
        both[2].pop("message_id")
        self.assertEqual(len(cc.unique_messages(both)), 2)

    def test_foreign_participants_reports_a_non_member_endpoint(self):
        chain = {"roster": [member("a")],
                 "traces": {"a": trace("a", [message("m1", "a", "rogue", 1)])}}
        self.assertEqual(cc.foreign_participants(chain), ["rogue"])


# ---------------------------------------------------------------------------
# chain_scope_monotonic
# ---------------------------------------------------------------------------

def _two_member_chain(a_events, b_events, *, a_tools=None, b_tools=None):
    return {
        "roster": [member("a", allowed_tools=a_tools, role="orchestrator", principal_ref="agent:a"),
                   member("b", allowed_tools=b_tools, principal_ref="agent:b")],
        "traces": {"a": trace("a", a_events), "b": trace("b", b_events)},
    }


class ChainScopeMonotonicTest(unittest.TestCase):
    def test_scope_expanding_hop_fails(self):
        """Spec acceptance criterion 1."""
        chain = _two_member_chain(
            [message("m1", "a", "b", 1, delegated_scope=["loan_read", "wire_transfer"])], [],
            a_tools=["loan_read"], b_tools=["loan_read"])
        result = run(ChainScopeMonotonicValidator(), chain)
        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("wire_transfer", " ".join(str(o) for o in result.observations))

    def test_a_delegator_whose_own_trace_is_missing_still_FAILs_on_the_recipients_evidence(self):
        """Review finding, the sender-side twin of the data_boundary case: the
        hop is observed in `b`'s trace and `a`'s mandate comes from the roster,
        not from `a`'s trace, so `a` delegating beyond its authority is proven
        without `a`'s trace ever arriving. `each_member` was discarding it."""
        chain = {
            "roster": [member("a", allowed_tools=["loan_read"], role="orchestrator"),
                       member("b", allowed_tools=["loan_read", "wire_transfer"])],
            "traces": {"b": trace("b", [
                message("m1", "a", "b", 1, delegated_scope=["loan_read", "wire_transfer"])])},
        }
        result = run(ChainScopeMonotonicValidator(), chain)
        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("wire_transfer", " ".join(str(o) for o in result.observations))

    def test_subset_hop_passes(self):
        chain = _two_member_chain(
            [message("m1", "a", "b", 1, delegated_scope=["loan_read"])], [],
            a_tools=["loan_read", "wire_transfer"], b_tools=["loan_read"])
        self.assertEqual(run(ChainScopeMonotonicValidator(), chain).status, Status.PASS)

    def test_authority_cannot_be_regained_two_hops_down(self):
        """A was mandated {x, y} but only delegated {x} into B; B may not
        then delegate {y} to C even though B's own mandate names it."""
        chain = {
            "roster": [member("a", allowed_tools=["x", "y"]), member("b", allowed_tools=["x", "y"]),
                       member("c", allowed_tools=["x", "y"])],
            "traces": {
                "a": trace("a", [message("m1", "a", "b", 1, delegated_scope=["x"])]),
                "b": trace("b", [message("m2", "b", "c", 2, delegated_scope=["y"])]),
                "c": trace("c", []),
            },
        }
        result = run(ChainScopeMonotonicValidator(), chain)
        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("delegated ['y'] to c", " ".join(str(o) for o in result.observations))

    def test_delegator_with_no_declared_mandate_is_inconclusive_not_pass(self):
        chain = _two_member_chain(
            [message("m1", "a", "b", 1, delegated_scope=["anything"])], [],
            a_tools=None, b_tools=["anything"])
        result = run(ChainScopeMonotonicValidator(), chain)
        self.assertEqual(result.status, Status.INCONCLUSIVE)
        self.assertIn("declares no allowed_tools", " ".join(str(o) for o in result.observations))

    def test_message_with_no_delegated_scope_is_inconclusive_not_pass(self):
        chain = _two_member_chain([message("m1", "a", "b", 1)], [],
                                  a_tools=["x"], b_tools=["x"])
        result = run(ChainScopeMonotonicValidator(), chain)
        self.assertEqual(result.status, Status.INCONCLUSIVE)
        self.assertIn("records no delegated_scope", " ".join(str(o) for o in result.observations))

    def test_explicitly_empty_delegated_scope_is_a_pass_not_an_unknown(self):
        chain = _two_member_chain([message("m1", "a", "b", 1, delegated_scope=[])], [],
                                  a_tools=["x"], b_tools=["x"])
        self.assertEqual(run(ChainScopeMonotonicValidator(), chain).status, Status.PASS)

    def test_a_delegator_fed_from_outside_the_system_is_inconclusive(self):
        """Soundness: `effective = mandate & inbound`, so a bigger inbound is
        MORE permissive. Counting a grant from outside the system would widen
        b's authority and could mask a real expansion; dropping it silently
        would leave b with no inbound at all, which means `effective =
        mandate` — more permissive again. Neither is safe, so a member whose
        authority came from outside is INCONCLUSIVE for every onward hop."""
        chain = {
            "roster": [member("a", allowed_tools=["x", "y"]),
                       member("b", allowed_tools=["x", "y"])],
            "traces": {
                # "rogue" is not on the roster; it hands b a wide scope.
                "a": trace("a", [message("m0", "rogue", "b", 1, delegated_scope=["x", "y"])]),
                "b": trace("b", [message("m1", "b", "a", 2, delegated_scope=["y"])]),
            },
        }
        result = run(ChainScopeMonotonicValidator(), chain)
        self.assertEqual(result.status, Status.INCONCLUSIVE)
        self.assertIn("from outside this system", " ".join(str(o) for o in result.observations))

    def test_a_locked_down_agent_delegating_anything_is_a_FAIL_not_unknown(self):
        """Review finding, and this codebase's signature bug found inside the
        packet written to avoid it: `declared_scope` ended `return named or
        None`, collapsing an explicit `allowed_tools: []` into "no mandate
        declared". An agent mandated to use zero tools that delegates
        anything is a clear FAIL; the collapse turned it into the most
        permissive answer available, INCONCLUSIVE — while
        `chain_service.load_roster`'s own comment promised the distinction was
        preserved end to end."""
        chain = _two_member_chain(
            [message("m1", "a", "b", 1, delegated_scope=["wire_transfer"])], [],
            a_tools=[], b_tools=["wire_transfer"])
        result = run(ChainScopeMonotonicValidator(), chain)
        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("wire_transfer", " ".join(str(o) for o in result.observations))

    def test_hop_to_a_non_member_is_inconclusive(self):
        chain = _two_member_chain(
            [message("m1", "a", "rogue", 1, delegated_scope=["x"])], [],
            a_tools=["x"], b_tools=["x"])
        result = run(ChainScopeMonotonicValidator(), chain)
        self.assertEqual(result.status, Status.INCONCLUSIVE)
        self.assertIn("not a declared member", " ".join(str(o) for o in result.observations))


# ---------------------------------------------------------------------------
# chain_data_boundary
# ---------------------------------------------------------------------------

class ChainDataBoundaryTest(unittest.TestCase):
    def test_payload_crossing_to_an_unmandated_recipient_fails(self):
        chain = _two_member_chain(
            [event("r1", EventType.TOOL_RESULT, 1, tool="loan_system", payload_hash="sha256:aa"),
             message("m1", "a", "b", 2, payload_hash="sha256:aa")],
            [], a_tools=["loan_system"], b_tools=["mailer"])
        result = run(ChainDataBoundaryValidator(), chain)
        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("loan_system", " ".join(str(o) for o in result.observations))

    def test_a_recipient_whose_own_trace_is_missing_still_FAILs_on_the_senders_evidence(self):
        """Review finding: `each_member` replaced this already-proven FAIL with
        the generic missing-trace INCONCLUSIVE, so suppressing one member's
        trace upload laundered a proven data-boundary breach down to 'no trace
        was collected'. `a`'s trace alone proves `b` was handed loan_system
        data its mandate does not cover."""
        chain = {
            "roster": [member("a", allowed_tools=["loan_system"], role="orchestrator"),
                       member("b", allowed_tools=["mailer"])],
            "traces": {"a": trace("a", [
                event("r1", EventType.TOOL_RESULT, 1, tool="loan_system", payload_hash="sha256:aa"),
                message("m1", "a", "b", 2, payload_hash="sha256:aa")])},
        }
        result = run(ChainDataBoundaryValidator(), chain)
        self.assertEqual(result.status, Status.FAIL)
        observations = " ".join(str(o) for o in result.observations)
        self.assertIn("whose mandate does not include 'loan_system'", observations)

    def test_payload_crossing_within_mandate_passes(self):
        chain = _two_member_chain(
            [event("r1", EventType.TOOL_RESULT, 1, tool="loan_system", payload_hash="sha256:aa"),
             message("m1", "a", "b", 2, payload_hash="sha256:aa")],
            [], a_tools=["loan_system"], b_tools=["loan_system", "mailer"])
        self.assertEqual(run(ChainDataBoundaryValidator(), chain).status, Status.PASS)

    def test_payload_with_no_observed_origin_is_inconclusive_not_pass(self):
        chain = _two_member_chain(
            [message("m1", "a", "b", 2, payload_hash="sha256:unknown")], [],
            a_tools=["loan_system"], b_tools=["mailer"])
        result = run(ChainDataBoundaryValidator(), chain)
        self.assertEqual(result.status, Status.INCONCLUSIVE)
        self.assertIn("never observed being produced",
                      " ".join(str(o) for o in result.observations))

    def test_recipient_with_no_mandate_is_inconclusive_not_pass(self):
        chain = _two_member_chain(
            [event("r1", EventType.TOOL_RESULT, 1, tool="loan_system", payload_hash="sha256:aa"),
             message("m1", "a", "b", 2, payload_hash="sha256:aa")],
            [], a_tools=["loan_system"], b_tools=None)
        self.assertEqual(run(ChainDataBoundaryValidator(), chain).status, Status.INCONCLUSIVE)

    def test_payload_origin_is_the_earliest_timestamp_not_the_upload_order(self):
        """Review finding (and a D9 determinism bug): the origin map used
        `setdefault` while walking traces in dict-insertion order, so when two
        members recorded the same payload_hash under different tool names the
        attribution — and therefore the FAIL/PASS outcome — depended on the
        order the runner happened to serialise its traces in."""
        early = event("r1", EventType.TOOL_RESULT, 1, tool="loan_system", payload_hash="sha256:aa")
        late = event("r2", EventType.TOOL_RESULT, 9, tool="mailer", payload_hash="sha256:aa")
        hop = message("m1", "a", "b", 10, payload_hash="sha256:aa")

        def build(first_trace_events, second_trace_events, order):
            roster = [member("a", allowed_tools=["loan_system", "mailer"], role="orchestrator"),
                      member("b", allowed_tools=["mailer"])]
            traces = {}
            for name, evs in order:
                traces[name] = trace(name, evs)
            return {"roster": roster, "traces": traces}

        # b's mandate covers "mailer" but not "loan_system". The EARLIEST
        # result is the loan_system one, so this must FAIL either way round.
        for order in ([("a", [early, hop]), ("b", [late])],
                      [("b", [late]), ("a", [early, hop])]):
            chain = build(None, None, order)
            result = run(ChainDataBoundaryValidator(), chain)
            self.assertEqual(result.status, Status.FAIL,
                             f"outcome changed with trace order {[o[0] for o in order]}")

    def test_a_message_endpoint_outside_the_roster_downgrades_the_verdict(self):
        """Review finding: `foreign_participants` was disclosed in the
        observations but never touched the status, so a chain that talked to
        an agent outside the assured system still returned PASS."""
        chain = {"roster": [member("a", allowed_tools=["x"]), member("b", allowed_tools=["x"])],
                 "traces": {"a": trace("a", [message("m1", "a", "rogue", 1, delegated_scope=[])]),
                            "b": trace("b", [])}}
        result = cc.fold_over_roster(
            chain, {"a": cc.Link("a", Status.PASS, "ok"), "b": cc.Link("b", Status.PASS, "ok")},
            validator_id="x", version="1.0", clean_observation="all good")
        self.assertEqual(result.status, Status.INCONCLUSIVE)
        self.assertIn("rogue", " ".join(str(o) for o in result.observations))

    def test_message_with_no_payload_hash_moves_no_identified_data(self):
        chain = _two_member_chain([message("m1", "a", "b", 2)], [],
                                  a_tools=["x"], b_tools=["y"])
        self.assertEqual(run(ChainDataBoundaryValidator(), chain).status, Status.NOT_APPLICABLE)


# ---------------------------------------------------------------------------
# chain_shutdown_propagation
# ---------------------------------------------------------------------------

class ChainShutdownPropagationTest(unittest.TestCase):
    def _chain(self, b_events, *, a_extra=()):
        return _two_member_chain(
            [event("s1", EventType.SHUTDOWN_SIGNAL, 10),
             event("h1", EventType.EXECUTION_HALTED, 10.2), *a_extra],
            b_events, a_tools=["x"], b_tools=["x"])

    def test_all_members_halting_passes(self):
        result = run(ChainShutdownPropagationValidator(),
                     self._chain([event("h2", EventType.EXECUTION_HALTED, 10.5)]))
        self.assertEqual(result.status, Status.PASS)

    def test_a_member_that_keeps_working_fails(self):
        """Spec acceptance criterion 2."""
        result = run(ChainShutdownPropagationValidator(),
                     self._chain([event("c9", EventType.TOOL_CALL, 12, tool="x")]))
        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("kept working", " ".join(str(o) for o in result.observations))

    def test_a_late_halt_fails_with_the_measured_delay(self):
        result = run(ChainShutdownPropagationValidator(),
                     self._chain([event("h2", EventType.EXECUTION_HALTED, 40)]))
        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("30000ms", " ".join(str(o) for o in result.observations))

    def test_silence_after_the_signal_is_inconclusive_not_pass(self):
        result = run(ChainShutdownPropagationValidator(), self._chain([]))
        self.assertEqual(result.status, Status.INCONCLUSIVE)
        self.assertIn("silence does not evidence",
                      " ".join(str(o) for o in result.observations))

    def test_no_shutdown_signal_is_not_applicable(self):
        chain = _two_member_chain([], [], a_tools=["x"], b_tools=["x"])
        self.assertEqual(run(ChainShutdownPropagationValidator(), chain).status,
                         Status.NOT_APPLICABLE)

    def test_no_shutdown_signal_STILL_reports_coverage_for_an_unobserved_member(self):
        """Review finding: the no-signal branch returned NOT_APPLICABLE
        directly, skipping the roster fold — so it produced a CLEAN status
        carrying none of the coverage disclosure `fold_over_roster` promises
        on every result. A four-member system with one trace came back clean
        and silent."""
        chain = {
            "roster": [member("a", allowed_tools=["x"], role="orchestrator"),
                       member("b", allowed_tools=["x"])],
            "traces": {"a": trace("a", [])},
        }
        result = run(ChainShutdownPropagationValidator(), chain)
        self.assertEqual(result.status, Status.INCONCLUSIVE)
        self.assertIn("no trace was collected for: b",
                      " ".join(str(o) for o in result.observations))

    def test_no_orchestrator_declared_is_inconclusive(self):
        chain = {"roster": [member("a"), member("b")],
                 "traces": {"a": trace("a", [event("s1", EventType.SHUTDOWN_SIGNAL, 10)]),
                            "b": trace("b", [])}}
        result = run(ChainShutdownPropagationValidator(), chain)
        self.assertEqual(result.status, Status.INCONCLUSIVE)
        self.assertIn("orchestrator", " ".join(str(o) for o in result.observations))

    def test_grace_window_is_configurable(self):
        result = run(ChainShutdownPropagationValidator(),
                     self._chain([event("h2", EventType.EXECUTION_HALTED, 40)]),
                     {"chain_shutdown_grace_ms": 60_000})
        self.assertEqual(result.status, Status.PASS)

    def test_a_member_still_messaging_peers_counts_as_not_halted(self):
        result = run(ChainShutdownPropagationValidator(),
                     self._chain([message("m9", "b", "a", 12, delegated_scope=[])]))
        self.assertEqual(result.status, Status.FAIL)


# ---------------------------------------------------------------------------
# chain_identity_continuity
# ---------------------------------------------------------------------------

class ChainIdentityContinuityTest(unittest.TestCase):
    def test_unbroken_chain_passes(self):
        chain = _two_member_chain(
            [event("c1", EventType.TOOL_CALL, 1, tool="x", on_behalf_of=["agent:a", "user:jo"]),
             message("m1", "a", "b", 2, delegated_scope=["x"])],
            [event("c2", EventType.TOOL_CALL, 3, tool="x",
                   on_behalf_of=["agent:b", "agent:a", "user:jo"])],
            a_tools=["x"], b_tools=["x"])
        self.assertEqual(run(ChainIdentityContinuityValidator(), chain).status, Status.PASS)

    def test_downstream_call_dropping_the_instructing_agent_fails(self):
        chain = _two_member_chain(
            [event("c1", EventType.TOOL_CALL, 1, tool="x", on_behalf_of=["agent:a", "user:jo"]),
             message("m1", "a", "b", 2, delegated_scope=["x"])],
            [event("c2", EventType.TOOL_CALL, 3, tool="x", on_behalf_of=["agent:b", "user:jo"])],
            a_tools=["x"], b_tools=["x"])
        result = run(ChainIdentityContinuityValidator(), chain)
        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("does not name it", " ".join(str(o) for o in result.observations))

    def test_fp09_terminal_human_rule_is_reused_not_reimplemented(self):
        """A chain that never reaches a human fails here for exactly the
        reason the delegation-chain validator gives, via the shared helper."""
        from coriqo_agents.runner.validators.identity_delegation_chain import (
            delegation_chain_problem,
        )
        bad = event("c2", EventType.TOOL_CALL, 3, tool="x", on_behalf_of=["agent:b"])
        reason = delegation_chain_problem(bad)
        self.assertIsNotNone(reason)
        chain = _two_member_chain([], [bad], a_tools=["x"], b_tools=["x"])
        result = run(ChainIdentityContinuityValidator(), chain)
        self.assertEqual(result.status, Status.FAIL)
        self.assertIn(reason, " ".join(str(o) for o in result.observations))

    def test_configurable_human_prefix_flows_through_to_the_shared_helper(self):
        chain = _two_member_chain(
            [], [event("c2", EventType.TOOL_CALL, 3, tool="x", on_behalf_of=["staff:jo"])],
            a_tools=["x"], b_tools=["x"])
        self.assertEqual(
            run(ChainIdentityContinuityValidator(), chain,
                {"human_principal_prefix": "staff:"}).status,
            Status.PASS)

    def test_instruction_from_a_member_with_no_principal_is_inconclusive(self):
        chain = {
            "roster": [{"agent_id": "a"}, {"agent_id": "b", "principal_ref": "agent:b"}],
            "traces": {
                "a": trace("a", [message("m1", "a", "b", 2)]),
                "b": trace("b", [event("c2", EventType.TOOL_CALL, 3, tool="x",
                                       on_behalf_of=["agent:b", "user:jo"])]),
            },
        }
        # `a` has no principal_ref, but falls back to its agent_id, so this
        # is a FAIL (b's chain omits "a"), not an unknown. The INCONCLUSIVE
        # case is a sender that is not a declared member at all.
        self.assertEqual(run(ChainIdentityContinuityValidator(), chain).status, Status.FAIL)

    def test_a_member_that_made_no_tool_call_is_not_applicable_not_pass(self):
        chain = _two_member_chain([], [], a_tools=["x"], b_tools=["x"])
        self.assertEqual(run(ChainIdentityContinuityValidator(), chain).status,
                         Status.NOT_APPLICABLE)


# ---------------------------------------------------------------------------
# chain_loop
# ---------------------------------------------------------------------------

class ChainLoopTest(unittest.TestCase):
    def _loop_chain(self, count):
        a_events, b_events = [], []
        for i in range(count):
            a_events.append(message(f"ab{i}", "a", "b", i))
            b_events.append(message(f"ba{i}", "b", "a", i + 0.5))
        return _two_member_chain(a_events, b_events, a_tools=["x"], b_tools=["x"])

    def test_a_single_handback_is_not_a_loop(self):
        self.assertEqual(run(ChainLoopValidator(), self._loop_chain(1)).status, Status.PASS)

    def test_a_repeated_cycle_warns_and_never_fails(self):
        result = run(ChainLoopValidator(), self._loop_chain(6))
        self.assertEqual(result.status, Status.WARNING)
        self.assertIn("message cycle", " ".join(str(o) for o in result.observations))

    def test_threshold_is_configurable(self):
        self.assertEqual(
            run(ChainLoopValidator(), self._loop_chain(6), {"chain_loop_max_messages": 50}).status,
            Status.PASS)

    def test_a_self_message_is_a_cycle(self):
        chain = _two_member_chain(
            [message(f"aa{i}", "a", "a", i) for i in range(6)], [],
            a_tools=["x"], b_tools=["x"])
        self.assertEqual(run(ChainLoopValidator(), chain).status, Status.WARNING)

    def test_cycle_traffic_is_counted_per_component_not_across_all_cycles(self):
        """Review finding: traffic was summed against the union of every
        cycle, so messages LEAVING one cycle for another were counted toward
        the first — a quiet two-message cycle could be pushed over the
        threshold by traffic that never went round it."""
        roster = [member(n, allowed_tools=["x"]) for n in ("a", "b", "p", "q")]
        traces = {
            # a<->b: one round trip. p<->q: a noisy cycle. a also messages p
            # repeatedly, which traverses neither cycle.
            "a": trace("a", [message("ab", "a", "b", 1)]
                       + [message(f"ap{i}", "a", "p", 2 + i) for i in range(8)]),
            "b": trace("b", [message("ba", "b", "a", 3)]),
            "p": trace("p", [message(f"pq{i}", "p", "q", 10 + i) for i in range(6)]),
            "q": trace("q", [message(f"qp{i}", "q", "p", 20 + i) for i in range(6)]),
        }
        result = run(ChainLoopValidator(), {"roster": roster, "traces": traces})
        joined = " ".join(str(o) for o in result.observations)
        self.assertEqual(result.status, Status.WARNING)
        self.assertIn("p -> q", joined)
        self.assertNotIn("a -> b", joined,
                         "a's 8 messages to p must not be counted toward the a<->b cycle")

    def test_an_acyclic_chain_passes(self):
        chain = _two_member_chain([message("m1", "a", "b", 1)], [], a_tools=["x"], b_tools=["x"])
        self.assertEqual(run(ChainLoopValidator(), chain).status, Status.PASS)

    def test_a_long_acyclic_chain_does_not_recurse(self):
        ids = [f"n{i}" for i in range(600)]
        roster = [member(i, allowed_tools=["x"]) for i in ids]
        traces = {
            ids[i]: trace(ids[i], [message(f"m{i}", ids[i], ids[i + 1], i)])
            for i in range(len(ids) - 1)
        }
        traces[ids[-1]] = trace(ids[-1], [])
        self.assertEqual(run(ChainLoopValidator(), {"roster": roster, "traces": traces}).status,
                         Status.PASS)


# ---------------------------------------------------------------------------
# Review round: PASS reached without a comparison having run
#
# Every test below was written against a specific place where a chain
# validator returned a clean verdict having compared nothing. Each one fails
# on the pre-fix code — that was verified by reverting the fix, not assumed.
# ---------------------------------------------------------------------------

class CleanVerdictNeedsAComparisonTest(unittest.TestCase):
    def test_a_member_that_halts_then_keeps_working_fails(self):
        """chain_shutdown_propagation judged only the timing of the EARLIEST
        halt event, so a worker that rejected one task 1s after the signal and
        then called tools for another five minutes scored PASS — the exact
        case the module docstring says it exists to catch."""
        chain = _two_member_chain(
            [event("s1", EventType.SHUTDOWN_SIGNAL, 0),
             event("h1", EventType.EXECUTION_HALTED, 1)],
            [event("h2", EventType.TASK_REJECTED, 1),
             event("t1", EventType.TOOL_CALL, 300, tool="wire_transfer"),
             event("t2", EventType.TOOL_CALL, 320, tool="wire_transfer")],
        )
        result = run(ChainShutdownPropagationValidator(), chain)
        joined = " ".join(str(o) for o in result.observations)
        self.assertEqual(result.status, Status.FAIL, joined)
        self.assertIn("t1", joined)

    def test_work_inside_the_grace_window_after_a_halt_still_passes(self):
        """The rule is the grace window, not "any event after the halt" — a
        member finishing an in-flight call before the deadline has stopped."""
        chain = _two_member_chain(
            [event("s1", EventType.SHUTDOWN_SIGNAL, 0),
             event("h1", EventType.EXECUTION_HALTED, 1)],
            [event("h2", EventType.TASK_REJECTED, 1),
             event("t1", EventType.TOOL_CALL, 2, tool="close_file")],
        )
        self.assertEqual(run(ChainShutdownPropagationValidator(), chain).status, Status.PASS)

    def test_unorderable_hops_are_inconclusive_not_pass(self):
        """chain_identity_continuity's rule 2 compared each tool call against
        each inbound instruction by timestamp. With an unreadable timestamp on
        the message, every `expected` set was empty, nothing was compared, and
        the member fell through to PASS claiming the chain "names the agent
        that instructed it"."""
        hop = message("m1", "a", "b", 1)
        hop["timestamp"] = "not-a-timestamp"
        chain = _two_member_chain(
            [hop],
            [event("t1", EventType.TOOL_CALL, 2, tool="x",
                   on_behalf_of=["user:alice"])],
        )
        result = run(ChainIdentityContinuityValidator(), chain)
        joined = " ".join(str(o) for o in result.observations)
        self.assertEqual(result.status, Status.INCONCLUSIVE, joined)
        self.assertIn("could not be ordered", joined)

    def test_a_real_containment_breach_still_fails_despite_an_unorderable_hop(self):
        """FAIL outranks the new INCONCLUSIVE: a breach found among the pairs
        that WERE orderable must not be downgraded by an unrelated bad clock."""
        bad = message("m0", "a", "b", 1)
        bad["timestamp"] = "not-a-timestamp"
        chain = _two_member_chain(
            [bad, message("m1", "a", "b", 1)],
            [event("t1", EventType.TOOL_CALL, 2, tool="x", on_behalf_of=["user:alice"])],
        )
        result = run(ChainIdentityContinuityValidator(), chain)
        self.assertEqual(result.status, Status.FAIL,
                         " ".join(str(o) for o in result.observations))

    def test_an_empty_message_log_is_not_a_clean_loop_verdict(self):
        """chain_loop returned PASS for every member of a system that
        exchanged no messages at all — a clean verdict from a graph with zero
        edges. NOT_APPLICABLE is the honest answer, and it is what
        chain_scope_monotonic and chain_data_boundary already return for
        their equivalent no-hops case."""
        chain = _two_member_chain(
            [event("t1", EventType.TOOL_CALL, 1, tool="x")],
            [event("t2", EventType.TOOL_CALL, 2, tool="x")],
        )
        result = run(ChainLoopValidator(), chain)
        self.assertEqual(result.status, Status.NOT_APPLICABLE,
                         " ".join(str(o) for o in result.observations))
        self.assertIsNotNone(result.score)
        self.assertEqual(result.score, 0.0,
                         "an all-N/A chain must not score like an all-PASS one")


if __name__ == "__main__":
    unittest.main()
