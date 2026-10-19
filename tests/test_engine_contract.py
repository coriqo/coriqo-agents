"""EngineAdapter contract tests. Pure stdlib, like the package."""
import unittest

from coriqo_agents.runner.engines import EngineAdapter, NormalizedResult, RawRun


class _TrivialAdapter(EngineAdapter):
    engine_id = "promptfoo"
    engine_version = "0.1.0"

    def prepare(self, campaign_bundle):
        return {"cases": campaign_bundle.get("cases", [])}

    def run(self, plan):
        return RawRun(
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            output={"results": []},
            artifact=b'{"results": []}',
        )

    def normalize(self, raw):
        return [
            NormalizedResult(
                validator_id="prompt_injection",
                version="1.0.0",
                status="PASS",
                engine=raw.engine_id,
            )
        ]

    def raw_artifact(self, raw):
        return raw.artifact


class EngineAdapterContractTest(unittest.TestCase):
    def test_abc_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            EngineAdapter()

    def test_partial_implementation_cannot_be_instantiated(self):
        """An adapter that skips raw_artifact would upload results with no
        re-checkable source — that must fail at construction."""

        class _Partial(EngineAdapter):
            engine_id = "promptfoo"

            def prepare(self, campaign_bundle):
                return {}

            def run(self, plan):
                return RawRun(engine_id="promptfoo", engine_version="0")

            def normalize(self, raw):
                return []

        with self.assertRaises(TypeError):
            _Partial()

    def test_trivial_subclass_runs_end_to_end(self):
        adapter = _TrivialAdapter()
        plan = adapter.prepare({"cases": [1, 2]})
        raw = adapter.run(plan)
        self.assertIsInstance(raw, RawRun)
        results = adapter.normalize(raw)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].engine, "promptfoo")
        self.assertEqual(adapter.raw_artifact(raw), b'{"results": []}')

    def test_normalize_is_deterministic_for_the_same_raw_run(self):
        adapter = _TrivialAdapter()
        raw = adapter.run(adapter.prepare({}))
        first = [r.to_dict() for r in adapter.normalize(raw)]
        second = [r.to_dict() for r in adapter.normalize(raw)]
        self.assertEqual(first, second)

    def test_normalized_result_carries_the_campaign_fields(self):
        result = NormalizedResult(
            validator_id="tool_authorization", version="1.0.0", status="FAIL",
            severity="critical", engine="promptfoo", control_ids=["AC-002"],
            failure_scenario={"category": "tool_misuse"},
        )
        payload = result.to_dict()
        self.assertEqual(payload["engine"], "promptfoo")
        self.assertEqual(payload["control_ids"], ["AC-002"])
        self.assertEqual(payload["failure_scenario"]["category"], "tool_misuse")
        # Defaults are per-instance, not shared class state.
        self.assertEqual(NormalizedResult("x", "1", "PASS").control_ids, [])


class TraceEventTypeTest(unittest.TestCase):
    def test_new_event_types_are_additive_and_unique(self):
        from coriqo_agents.runner.trace import EventType

        self.assertEqual(len(EventType.ALL), len(set(EventType.ALL)))
        for name in (
            "TASK_REQUEST", "TASK_REJECTED", "SHUTDOWN_SIGNAL",
            "CREDENTIAL_REVOKED", "MEMORY_READ", "MEMORY_WRITE",
            "HUMAN_OVERRIDE", "EXECUTION_HALTED", "TRIGGER_EVENT",
            "EXECUTION_RESUMED",
        ):
            self.assertIn(getattr(EventType, name), EventType.ALL)
        # The seven original types are untouched — existing validators filter
        # on these strings.
        for name in (
            "USER_REQUEST", "LLM_CALL", "TOOL_CALL", "TOOL_RESULT",
            "HUMAN_APPROVAL", "EXTERNAL_SIDE_EFFECT", "FINAL_RESPONSE",
        ):
            self.assertEqual(getattr(EventType, name), name)

    def test_events_of_ignores_unknown_types(self):
        from coriqo_agents.runner.trace import EventType, events_of

        trace = {"events": [
            {"event_type": "TOOL_CALL"},
            {"event_type": EventType.SHUTDOWN_SIGNAL},
        ]}
        self.assertEqual(len(events_of(trace, "TOOL_CALL")), 1)
        self.assertEqual(len(events_of(trace, EventType.SHUTDOWN_SIGNAL)), 1)


if __name__ == "__main__":
    unittest.main()
