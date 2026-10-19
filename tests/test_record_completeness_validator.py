"""AC-007's record check — `record_completeness`.

Pure stdlib `unittest`, same convention as test_security_validators.py.
Runnable standalone (`python test_record_completeness_validator.py`) and
under pytest.

The class of bug this file exists to pin: an incomplete record must never
read as clean. Each of the three required fields gets its own missing-case
test, because a check that looked only at `event_id` would pass all three
of these while two of them are unreconstructable. The two non-verdicts are
pinned too and in opposite directions — no trace at all is INCONCLUSIVE
(silence is not a clean bill), a trace with nothing consequential in it is
NOT_APPLICABLE (nothing happened for AC-007 to govern).
"""
from __future__ import annotations

import unittest

from coriqo_agents.runner.contract import Status, ValidatorContext
from coriqo_agents.runner.validators import VALIDATOR_REGISTRY
from coriqo_agents.runner.validators.record_completeness import (
    RecordCompletenessValidator,
)


def ctx(trace=None):
    return ValidatorContext(
        execution_id="e1", agent_model_id="m1", agent_version_ref=None,
        environment="STAGING", configuration={},
        inputs={} if trace is None else {"trace": trace},
    )


def run(trace=None):
    v = RecordCompletenessValidator()
    return v.evaluate(v.execute(ctx(trace)))


def evidence(trace):
    v = RecordCompletenessValidator()
    return v.produce_evidence(v.execute(ctx(trace)))


def scenario(trace):
    v = RecordCompletenessValidator()
    raw = v.execute(ctx(trace))
    return v.failure_scenario(v.evaluate(raw))


def trace_of(*events):
    return {"trace_id": "t1", "agent_id": "a1", "agent_version": "1.0.0",
            "events": list(events)}


def call(**overrides):
    """A complete TOOL_CALL. Tests remove exactly one field from it, so a
    failure names the field that was removed rather than a hand-built
    event that was also short of something else."""
    event = {"event_id": "evt_1", "event_type": "TOOL_CALL", "tool": "loan_system",
             "action": "update_application", "timestamp": "2026-01-01T00:00:00Z",
             "payload_hash": "sha256:abc"}
    event.update(overrides)
    return {k: v for k, v in event.items() if v is not _REMOVED}


class _Removed:
    pass


_REMOVED = _Removed()


class RegistryTest(unittest.TestCase):
    def test_registered(self):
        self.assertIn("record_completeness", VALIDATOR_REGISTRY)
        self.assertIs(VALIDATOR_REGISTRY["record_completeness"], RecordCompletenessValidator)


class PassTest(unittest.TestCase):
    def test_complete_tool_call_passes(self):
        r = run(trace_of(call()))
        self.assertEqual(r.status, Status.PASS)
        self.assertEqual(r.score, 1.0)

    def test_args_hash_is_accepted_in_place_of_payload_hash(self):
        r = run(trace_of(call(payload_hash=_REMOVED, args_hash="sha256:def")))
        self.assertEqual(r.status, Status.PASS)

    def test_a_hash_nested_under_payload_is_accepted(self):
        r = run(trace_of(call(payload_hash=_REMOVED,
                              payload={"payload_hash": "sha256:def"})))
        self.assertEqual(r.status, Status.PASS)

    def test_external_side_effect_is_consequential_too(self):
        r = run(trace_of(call(event_id="evt_2", event_type="EXTERNAL_SIDE_EFFECT")))
        self.assertEqual(r.status, Status.PASS)

    def test_non_consequential_events_are_not_checked(self):
        """An LLM_CALL carrying nothing at all must not drag the verdict
        down: AC-007 governs actions that change state or invoke a tool."""
        r = run(trace_of(call(), {"event_type": "LLM_CALL"}))
        self.assertEqual(r.status, Status.PASS)


class FailTest(unittest.TestCase):
    def test_missing_event_id_fails(self):
        r = run(trace_of(call(event_id=_REMOVED)))
        self.assertEqual(r.status, Status.FAIL)
        self.assertEqual(r.severity, "high")
        self.assertIn("event_id", " ".join(r.observations))

    def test_missing_timestamp_fails(self):
        r = run(trace_of(call(timestamp=_REMOVED)))
        self.assertEqual(r.status, Status.FAIL)
        self.assertIn("timestamp", " ".join(r.observations))

    def test_missing_hash_fails(self):
        """The disclosure case loop_detection handles by widening its
        signature: here there is nothing to widen — an event with no hash
        of what it carried cannot support reconstruction, so it is a
        missing field, not a coarser match."""
        r = run(trace_of(call(payload_hash=_REMOVED)))
        self.assertEqual(r.status, Status.FAIL)
        self.assertIn("content_hash", " ".join(r.observations))

    def test_empty_string_is_not_a_record(self):
        r = run(trace_of(call(event_id="   ")))
        self.assertEqual(r.status, Status.FAIL)

    def test_non_string_timestamp_is_not_a_record(self):
        r = run(trace_of(call(timestamp=1767225600)))
        self.assertEqual(r.status, Status.FAIL)

    def test_score_is_the_share_of_complete_events(self):
        r = run(trace_of(call(), call(event_id="evt_2", timestamp=_REMOVED),
                         call(event_id="evt_3"), call(event_id="evt_4")))
        self.assertEqual(r.status, Status.FAIL)
        self.assertEqual(r.score, 0.75)

    def test_every_missing_field_on_one_event_is_named(self):
        r = run(trace_of(call(event_id=_REMOVED, timestamp=_REMOVED,
                              payload_hash=_REMOVED)))
        joined = " ".join(r.observations)
        for field in ("event_id", "timestamp", "content_hash"):
            self.assertIn(field, joined)

    def test_evidence_names_the_incomplete_events_only(self):
        rows = evidence(trace_of(call(), call(event_id="evt_2", timestamp=_REMOVED)))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_id"], "evt_2")
        self.assertEqual(rows[0]["missing"], ["timestamp"])

    def test_failure_scenario_is_produced_for_a_fail_only(self):
        self.assertIsNone(scenario(trace_of(call())))
        payload = scenario(trace_of(call(event_id=_REMOVED)))
        self.assertEqual(payload["category"], "record_completeness")
        self.assertEqual(payload["severity"], "high")


class NonVerdictTest(unittest.TestCase):
    def test_no_trace_is_inconclusive(self):
        self.assertEqual(run(None).status, Status.INCONCLUSIVE)

    def test_a_trace_with_no_consequential_event_is_not_applicable(self):
        r = run(trace_of({"event_id": "evt_1", "event_type": "LLM_CALL",
                          "timestamp": "2026-01-01T00:00:00Z"}))
        self.assertEqual(r.status, Status.NOT_APPLICABLE)

    def test_an_empty_trace_is_not_applicable_not_inconclusive(self):
        """A trace that was supplied and recorded nothing consequential is
        a fact about the run; only an absent trace is unknowable."""
        self.assertEqual(run(trace_of()).status, Status.NOT_APPLICABLE)

    def test_a_non_dict_event_is_skipped_rather_than_crashing(self):
        r = run({"events": ["not-an-event", call()]})
        self.assertEqual(r.status, Status.PASS)


class PurityTest(unittest.TestCase):
    def test_evaluating_does_not_modify_the_trace(self):
        trace = trace_of(call(event_id=_REMOVED))
        import copy
        before = copy.deepcopy(trace)
        run(trace)
        self.assertEqual(trace, before)


if __name__ == "__main__":
    unittest.main()
