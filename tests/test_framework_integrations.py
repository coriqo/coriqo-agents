"""LangChain callback and OpenAI Agents SDK tracing processor: what one run records.

Each test is skipped when its framework is not installed, so the base suite
stays free of both. Install the extras to run them:
pip install -e '.[langchain,openai-agents,test]'."""
from __future__ import annotations

import os
import uuid

import pytest

_REQUIRE = os.environ.get("CORIQO_REQUIRE_FRAMEWORKS") == "1"


def _need(module: str):
    """importorskip, except under `make test-sdk-frameworks`, where a missing
    framework is a failure: a suite that skips everything tests nothing."""
    if _REQUIRE:
        return __import__(module, fromlist=["_"])
    return pytest.importorskip(module)

from coriqo_agents.client import AgentSuspendedError, CoriqoAgentClient


class FakeClient(CoriqoAgentClient):
    def __init__(self, fail: Exception | None = None):  # no HTTP client needed
        self.calls: list[tuple[str, dict]] = []
        self.fail = fail

    def record_trace(self, agent_id, **kwargs):
        if self.fail:
            raise self.fail
        self.calls.append((agent_id, kwargs))
        return {}


# ---------------------------------------------------------------- LangChain


def _lc():
    return _need("coriqo_agents.langchain")


def test_langchain_pairs_parallel_tools_by_run_id_and_folds_nested_runs():
    mod = _lc()
    _need("langchain_core")
    client = FakeClient()
    h = mod.CoriqoCallbackHandler(client, "credit-agent")
    root, inner, t1, t2, llm = (uuid.uuid4() for _ in range(5))
    h.on_chain_start({}, {}, run_id=root)
    h.on_chain_start({}, {}, run_id=inner, parent_run_id=root)
    h.on_chat_model_start({}, [], run_id=llm, parent_run_id=inner,
                          invocation_params={"model": "gpt-4.1"}, metadata={"ls_provider": "openai"})
    h.on_llm_end(type("R", (), {"llm_output": {"token_usage": {"total_tokens": 42}}, "generations": []})(), run_id=llm)
    h.on_tool_start({"name": "lookup_limit"}, "a", run_id=t1, parent_run_id=inner)
    h.on_tool_start({"name": "lookup_score"}, "b", run_id=t2, parent_run_id=inner)
    h.on_tool_end("score=700", run_id=t2)          # finishes first
    h.on_tool_error(ValueError("timeout"), run_id=t1)
    h.on_chain_end({"x": 1}, run_id=inner, parent_run_id=root)  # nested: no trace yet
    assert client.calls == []
    h.on_chain_end({"answer": "approve"}, run_id=root)
    [(agent, kw)] = client.calls
    assert agent == "credit-agent"
    tools = {c["tool"]: c for c in kw["tool_calls"]}
    assert set(tools) == {"lookup_limit", "lookup_score"}
    assert tools["lookup_limit"]["error"] == "timeout"
    assert kw["runtime"] == {"provider": "openai", "model": "gpt-4.1", "framework": "langchain"}
    assert kw["token_count"] == 42 and kw["latency_ms"] >= 0
    assert h._runs == {} and h._root == {}  # nothing left on a reused handler


def test_langchain_failed_run_is_recorded_and_outage_is_buffered(monkeypatch):
    mod = _lc()
    _need("langchain_core")
    buffered = []
    monkeypatch.setattr(mod, "buffer_trace", lambda a, p, e: buffered.append((a, p)))
    h = mod.CoriqoCallbackHandler(FakeClient(fail=RuntimeError("down")), "a1")
    root = uuid.uuid4()
    h.on_chain_start({}, {}, run_id=root)
    h.on_chain_error(RuntimeError("boom"), run_id=root)  # must not raise
    assert buffered and buffered[0][1]["output"] == "Chain run failed: boom"


def test_langchain_suspension_still_raises():
    mod = _lc()
    _need("langchain_core")
    h = mod.CoriqoCallbackHandler(FakeClient(fail=AgentSuspendedError(423, "suspended")), "a1")
    root = uuid.uuid4()
    h.on_chain_start({}, {}, run_id=root)
    with pytest.raises(AgentSuspendedError):
        h.on_chain_end({}, run_id=root)


def test_langchain_suspension_reaches_the_caller_through_a_real_invoke():
    # The handler raising is not enough: LangChain swallows callback errors unless raise_error is set.
    mod = _lc()
    _need("langchain_core")
    from langchain_core.runnables import RunnableLambda

    h = mod.CoriqoCallbackHandler(FakeClient(fail=AgentSuspendedError(423, "suspended")), "a1")
    with pytest.raises(AgentSuspendedError):
        RunnableLambda(lambda x: x).invoke(1, config={"callbacks": [h]})


def test_langchain_outage_does_not_reach_the_caller_through_a_real_invoke(monkeypatch):
    mod = _lc()
    _need("langchain_core")
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setattr(mod, "buffer_trace", lambda *a: None)
    h = mod.CoriqoCallbackHandler(FakeClient(fail=RuntimeError("down")), "a1")
    assert RunnableLambda(lambda x: x + 1).invoke(1, config={"callbacks": [h]}) == 2


# ------------------------------------------------------- OpenAI Agents SDK
def test_openai_agents_run_becomes_one_trace():
    _need("agents")
    from agents.tracing import (agent_span, function_span, generation_span, guardrail_span,
                                handoff_span, set_trace_processors, trace)
    from coriqo_agents.openai_agents import CoriqoTracingProcessor

    client = FakeClient()
    set_trace_processors([CoriqoTracingProcessor(client, "fallback", agent_ids={"Triage": "triage-agent"})])
    try:
        with trace("loan review"):
            with agent_span(name="Triage"):
                with generation_span(model="gpt-4.1", usage={"input_tokens": 10, "output_tokens": 5},
                                     output=[{"role": "assistant", "content": "Approve, limit 5000"}]):
                    pass
                with function_span(name="get_credit_score", input='{"id": "a-1"}') as s:
                    s.span_data.output = 700
                with handoff_span(from_agent="Triage", to_agent="Underwriter"):
                    pass
                with guardrail_span(name="pii_check", triggered=True):
                    pass
    finally:
        set_trace_processors([])
    [(agent, kw)] = client.calls
    assert agent == "triage-agent"
    assert [c["tool"] for c in kw["tool_calls"]] == ["get_credit_score", "handoff:Underwriter", "guardrail:pii_check"]
    assert kw["tool_calls"][0]["result"] == 700
    assert kw["runtime"]["model"] == "gpt-4.1"
    assert kw["token_count"] == 15
    assert kw["output"] == "Approve, limit 5000"
    assert kw["occurred_at"] is not None and kw["latency_ms"] >= 0


def test_openai_agents_outage_never_raises_into_the_run(monkeypatch):
    _need("agents")
    from agents.tracing import set_trace_processors, trace
    import coriqo_agents.openai_agents as mod

    buffered = []
    monkeypatch.setattr(mod, "buffer_trace", lambda a, p, e: buffered.append(a))
    set_trace_processors([mod.CoriqoTracingProcessor(FakeClient(fail=RuntimeError("down")), "a1")])
    try:
        with trace("run"):
            pass
    finally:
        set_trace_processors([])
    assert buffered == ["a1"]


def test_openai_agents_tool_error_stays_on_the_call_and_first_agent_decides():
    _need("agents")
    from agents.tracing import agent_span, function_span, set_trace_processors, trace
    from coriqo_agents.openai_agents import CoriqoTracingProcessor

    client = FakeClient()
    set_trace_processors([CoriqoTracingProcessor(client, "fallback", agent_ids={"Triage": "triage-agent", "Underwriter": "uw-agent"})])
    try:
        with trace("loan review"):
            with agent_span(name="Triage"):
                with function_span(name="fetch_bureau", input="{}") as s:
                    s.set_error({"message": "bureau timeout", "data": None})
            with agent_span(name="Underwriter"):
                pass
    finally:
        set_trace_processors([])
    [(agent, kw)] = client.calls
    assert agent == "triage-agent"
    assert kw["tool_calls"][0]["error"] == "bureau timeout"
    assert not str(kw["output"] or "").startswith("Run failed")
