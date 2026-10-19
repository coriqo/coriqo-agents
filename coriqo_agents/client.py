"""Thin client for Coriqo's agent-monitoring API.

See this SDK's README (or ask your Coriqo contact for the agent monitoring
integration guide) for the full contract this wraps: what gets hashed vs
stored, what the mandate-scope and grounding checks verify, and how to
create the service account this client authenticates as.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from .evals import hash_eval_artifact
from .hashing import hash_value


class CoriqoAPIError(Exception):
    """Raised for any non-2xx response. `status_code` and `detail` let a
    caller branch on specific failures (e.g. 409 = agent has no mandate
    version yet) without parsing the message text."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Coriqo API error {status_code}: {detail}")


class AgentSuspendedError(CoriqoAPIError):
    """Raised on a 423 from record_trace: the agent has been suspended by
    governance (see the integration guide's "no live circuit-breaker"
    caveat — Coriqo can reject further /traces submissions, but can't stop
    your agent from having already acted). A caller whose loop checks for
    this specifically can choose to actually halt; anyone only catching
    CoriqoAPIError still catches this too, since it's a subclass."""


def _hash_tool_calls(tool_calls: list[dict] | None) -> list[dict] | None:
    """Hashes `args`/`result` locally before anything is sent, so raw tool
    payloads never leave the caller's process — matching the integration
    guide's "Coriqo never sees the underlying tool payloads" claim as
    something this client enforces, not just documents. A caller that
    already has pre-computed args_hash/result_hash (e.g. from its own
    audit pipeline) can pass those directly instead; this only computes what
    isn't already there.

    Anything else you put on a call is kept, matching the server: a string or
    number goes through as recorded (truncated at 512 chars), and a structured
    value is replaced by `<key>_hash`, so the same never-send-raw-payloads rule
    that covers args/result covers everything else you attach."""
    if not tool_calls:
        return tool_calls
    hashed = []
    for tc in tool_calls:
        entry = {"step": tc.get("step"), "tool": tc.get("tool")}
        entry["args_hash"] = tc.get("args_hash") or hash_value(tc.get("args"))
        entry["result_hash"] = tc.get("result_hash") or hash_value(tc.get("result"))
        for key, value in tc.items():
            if key in _CORE_TOOL_CALL_KEYS or key in entry:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                entry[key] = value[:_SCALAR_MAX] if isinstance(value, str) else value
            else:
                entry[f"{key}_hash"] = hash_value(value)
        hashed.append(entry)
    return hashed


_CORE_TOOL_CALL_KEYS = frozenset({"step", "tool", "args", "result", "args_hash", "result_hash"})
_SCALAR_MAX = 512

OUTCOMES = ("confirmed", "overturned", "corrected", "inconclusive")


def _iso_occurred_at(occurred_at: datetime | None) -> str | None:
    """ISO-8601 for the wire, or None to let the server default to now.

    Rejects a naive datetime. The server treats one as already UTC (see its
    own "a client may POST a naive timestamp" comment), so silently accepting
    datetime.now() would seal a local time as UTC — hours off in most
    timezones, with nothing in the chain flagging it. Shared by record_trace,
    record_traces, record_eval_run and record_adversarial_run so all four
    reject the same input the same way."""
    if occurred_at is None:
        return None
    if occurred_at.tzinfo is None:
        raise ValueError(
            "occurred_at must be timezone-aware (e.g. datetime.now(timezone.utc)) — "
            "a naive datetime would be sealed as UTC even if it isn't"
        )
    return occurred_at.isoformat()


def _artifact_hash_field(
    artifact_inputs: Any | None, artifact_hash: str | None, hash_inputs_locally: bool,
) -> tuple[Any | None, str | None]:
    """Settle artifact_inputs vs artifact_hash before the request goes out.

    The server rejects both-at-once with a 422, on purpose: a caller-supplied
    hash sent alongside the real inputs could contradict them, and the sealed
    record promises artifact_hash is re-derivable from what was hashed. This
    raises locally instead of spending a round-trip to be told so — the same
    reason record_traces checks its own 200-item cap.

    With only `artifact_inputs` and hashing left on (the default), the digest
    is computed here and the raw inputs are dropped from the body. That is the
    point of the helper: Coriqo stores the hash and nothing else, so sending a
    whole dataset over the wire buys nothing and exposes it to everything
    between the harness and the API."""
    if artifact_inputs is not None and artifact_hash is not None:
        raise ValueError(
            "Send artifact_inputs or artifact_hash, not both — the server rejects the pair "
            "with a 422 because a supplied hash could contradict the inputs sent with it. "
            "To hash it yourself, pass artifact_hash=hash_eval_artifact(inputs) alone."
        )
    if artifact_inputs is not None and hash_inputs_locally:
        return None, hash_eval_artifact(artifact_inputs)
    return artifact_inputs, artifact_hash



def hash_subject_ref(subject_ref: str | None) -> str | None:
    """Hashes the subject reference locally, the same way tool-call args are
    hashed, so the raw identifier (an account number, a customer id, a claim
    number) never leaves the caller's process.

    Coriqo only ever holds the digest, which is enough to group every decision
    that touched the same subject and to answer traces_by_subject() — a lookup
    only someone who already knows the raw identifier can perform. The cost is
    that the digest is opaque: Coriqo cannot show an examiner *which* customer
    a group of decisions concerns, and cannot search by partial or fuzzy match.

    Already-hashed values (anything starting "sha256:") pass through unchanged,
    so a caller with its own hashing pipeline can supply the digest directly.
    The same function backs record_trace/record_traces and traces_by_subject,
    which is what makes a query match what was stored."""
    if subject_ref is None:
        return None
    if isinstance(subject_ref, str) and subject_ref.startswith("sha256:"):
        return subject_ref
    return hash_value(subject_ref)


def external_ref(ref_id: str, *, system: str | None = None, label: str | None = None) -> dict:
    """A grounding reference to evidence that lives OUTSIDE Coriqo — a ledger
    anchor, a record id in an upstream system.

    Pass the result in `grounding_refs` alongside ordinary Coriqo record ids.
    Coriqo records and displays it but leaves it out of the decision's grounding
    score, because it has no way to resolve it. Sending such an anchor as a bare
    string instead makes Coriqo try to resolve it as one of its own records,
    fail, and score the decision as citing a record that doesn't exist — which
    reads as a fabricated citation. It is not counted as verified either: an
    external anchor is reported separately, not scored in either direction."""
    ref = {"type": "external", "id": ref_id}
    if system:
        ref["system"] = system
    if label:
        ref["label"] = label
    return ref


class CoriqoAgentClient:
    """Authenticates as a Coriqo service account (X-API-Key + X-Tenant-Slug)
    rather than a human JWT — see the integration guide for how to create
    one. Not thread-safe across concurrent calls sharing one instance beyond
    what the underlying httpx.Client already guarantees; construct one
    client per agent process, not a shared global.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        tenant_slug: str | None = None,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
        http_client: httpx.Client | None = None,
    ):
        """Either pass base_url/api_key/tenant_slug (the normal path), or an
        already-configured http_client — e.g. FastAPI's TestClient in tests,
        or your own httpx.Client with custom retry/proxy settings — that
        already carries the X-API-Key/X-Tenant-Slug headers."""
        if http_client is not None:
            self._client = http_client
            self._owns_client = False  # caller's — never close it out from under them
            return
        if base_url is None or api_key is None or tenant_slug is None:
            raise ValueError(
                "base_url, api_key, and tenant_slug are required unless http_client is given"
            )
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key, "X-Tenant-Slug": tenant_slug},
            timeout=timeout,
            transport=transport,
        )
        self._owns_client = True

    def close(self) -> None:
        """No-op if this client was constructed from a caller-supplied
        http_client — closing someone else's shared httpx.Client/TestClient
        out from under them would break its next use elsewhere in their
        process. Only closes a client this instance built itself."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "CoriqoAgentClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._parse_response(self._client.request(method, path, **kwargs))

    @staticmethod
    def _parse_response(resp: Any) -> Any:
        if resp.status_code >= 400:
            detail = resp.text
            if resp.headers.get("content-type", "").startswith("application/json"):
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:
                    pass
            if resp.status_code == 423:
                raise AgentSuspendedError(resp.status_code, detail)
            raise CoriqoAPIError(resp.status_code, detail)
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError as e:  # json.JSONDecodeError is a ValueError subclass
            # A 2xx with a non-JSON body (a proxy/load-balancer intercepting
            # the request, a maintenance page) isn't a Coriqo API response —
            # surface it as the same error type every other failure raises,
            # so a caller's `except CoriqoAPIError` still catches it instead
            # of crashing on an unexpected exception type.
            raise CoriqoAPIError(resp.status_code, f"Non-JSON response body: {e}") from e

    # --- agents ------------------------------------------------------------

    def register_agent(
        self,
        *,
        name: str,
        mandate: str | None = None,
        system: str | None = None,
        risk_tier: str | None = None,
        owner_id: str | None = None,
        allowed_tools: list[str] | None = None,
        external_id: str | None = None,
        mandate_enforcement: str = "enforce",
        expected_report_interval_minutes: int | None = None,
    ) -> dict:
        """Registers the agent. Lands at 'in review' — this never
        pre-approves it, same as any model.

        Pass `external_id` (your own stable id — a deployment name, a config
        key) to make this idempotent: calling it again with the same one
        returns the agent that already exists instead of registering a second
        copy. Without it, every call — including the one your process makes on
        each restart — registers a new agent. Re-registering returns the agent
        as it stands and does NOT push `mandate`/`allowed_tools` onto it; those
        are a governed, separately-approved version, so changing them goes
        through amend_mandate().

        `mandate_enforcement="observe"` records an out-of-mandate tool call on
        the decision without flagging it or raising a finding — for getting
        allowed_tools right from real traffic (see observed_tools()) before you
        turn enforcement on. It applies at registration; changing it later is an
        amend_mandate() call, because relaxing enforcement is a mandate change.

        `expected_report_interval_minutes` says how often this agent is
        supposed to report a decision. Coriqo uses it to tell "quiet because
        there was nothing to do" apart from "stopped reporting" — an agent
        that misses the interval is surfaced as silent rather than healthy.
        Set it from the agent's real cadence, including its idle periods: too
        tight a value produces a silence alert every weekend. Leave it unset
        and no silence check runs.
        """
        return self._request("POST", "/api/v1/agents", json={
            "name": name, "mandate": mandate, "system": system,
            "risk_tier": risk_tier, "owner_id": owner_id, "allowed_tools": allowed_tools,
            "external_id": external_id, "mandate_enforcement": mandate_enforcement,
            "expected_report_interval_minutes": expected_report_interval_minutes,
        })

    def get_agent(self, agent_id: str) -> dict:
        return self._request("GET", f"/api/v1/agents/{agent_id}")

    # --- kill switch ---------------------------------------------------------
    # Governance-level, not a live circuit-breaker — see the integration
    # guide. Suspending stops further record_trace() calls from succeeding
    # (raises AgentSuspendedError); it can't stop your agent mid-action.

    def suspend_agent(self, agent_id: str, *, reason: str) -> dict:
        return self._request("POST", f"/api/v1/agents/{agent_id}/suspend", json={"reason": reason})

    def resume_agent(self, agent_id: str) -> dict:
        return self._request("POST", f"/api/v1/agents/{agent_id}/resume")

    # --- mandate amendment -----------------------------------------------------

    def amend_mandate(
        self, agent_id: str, *, reason: str, mandate: str | None = None,
        allowed_tools: list[str] | None = None, mandate_enforcement: str | None = None,
    ) -> dict:
        """Points the agent at a new mandate version. Decisions already
        recorded stay anchored to the version they actually ran under;
        only decisions recorded after this call see the new mandate/
        allowed_tools. At least one of mandate/allowed_tools/
        mandate_enforcement is required, and a reason always is."""
        return self._request("POST", f"/api/v1/agents/{agent_id}/mandate", json={
            "mandate": mandate, "allowed_tools": allowed_tools, "reason": reason,
            "mandate_enforcement": mandate_enforcement,
        })

    def mandate_history(self, agent_id: str) -> list[dict]:
        return self._request("GET", f"/api/v1/agents/{agent_id}/mandate/history")

    def deploy_check(
        self, agent_id: str, *, mandate_sha256: str | None = None, require_approved: bool = True,
    ) -> dict:
        """May this agent go live right now? Read-only (GET .../deploy-check).

        Returns {"allowed": bool, "reasons": [...], "checks": [...], ...}. A
        blocked agent is a normal answer, not an exception. An unknown agent
        raises CoriqoAPIError(404): Coriqo has no mandate for it."""
        params: dict[str, Any] = {"require_approved": "true" if require_approved else "false"}
        if mandate_sha256:
            params["mandate_sha256"] = mandate_sha256
        return self._request("GET", f"/api/v1/agents/{agent_id}/deploy-check", params=params)

    def observed_tools(self, agent_id: str, *, limit: int = 2000) -> dict:
        """Which tools this agent's decisions actually used, and which of them
        its mandate doesn't cover. allowed_tools is checked on every decision,
        so a list written from memory flags every normal run — read this first
        and set the list from what the agent really does."""
        return self._request(
            "GET", f"/api/v1/agents/{agent_id}/mandate/observed-tools", params={"limit": limit},
        )

    # --- trajectories (multi-step runs, including sub-agent runs) -------------

    def start_trajectory(
        self, agent_id: str, *, goal: str | None = None, use_case: str | None = None,
        parent_trajectory_id: str | None = None,
        delegated_from_trajectory_id: str | None = None,
    ) -> dict:
        """Opens a multi-step run, so its decisions land as one auditable unit
        instead of unlinked traces.

        Two different pointers, for two different relations — pick by whether
        the same agent is doing the work:

        `parent_trajectory_id` NESTS a sub-run under another run of the SAME
        agent. A flagged step anywhere inside it rolls up through every ancestor
        run. The parent must belong to this agent and still be open.

        `delegated_from_trajectory_id` records a hand-off to a DIFFERENT agent.
        The run stays on this agent's own chain and merely references the
        delegating run, so one agent's flags never roll up onto another's. The
        delegating run must still be open, and the delegating agent's mandate
        must permit delegation (`delegation_policy="attenuated"`) within its
        `max_delegation_depth`."""
        return self._request("POST", f"/api/v1/agents/{agent_id}/trajectories", json={
            "goal": goal, "use_case": use_case, "parent_trajectory_id": parent_trajectory_id,
            "delegated_from_trajectory_id": delegated_from_trajectory_id,
        })

    def complete_trajectory(self, agent_id: str, trajectory_id: str) -> dict:
        """Seals the run as done — completed, or flagged if any of its steps (or
        any step of a run nested under it) went out of mandate. Sub-runs must be
        completed first: the seal names a final status, so nothing underneath it
        may still be able to change that status afterwards."""
        return self._request(
            "POST", f"/api/v1/agents/{agent_id}/trajectories/{trajectory_id}/complete",
        )

    def fail_trajectory(self, agent_id: str, trajectory_id: str, *, reason: str) -> dict:
        """Closes the run as failed — the crashed/timed-out/abandoned ending.
        Use it in your `except` and shutdown paths: a run left open reads as
        still running forever, which quietly inflates how many runs look
        healthy. Like complete_trajectory(), this seals a final status, so
        record it once and after any sub-runs are closed."""
        return self._request(
            "POST", f"/api/v1/agents/{agent_id}/trajectories/{trajectory_id}/fail",
            json={"reason": reason},
        )

    def list_trajectories(
        self, agent_id: str, *, status: str | None = None, limit: int = 200,
    ) -> list[dict]:
        """This agent's runs, newest first, each with its `depth` in the
        delegation tree and its step counts."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._request("GET", f"/api/v1/agents/{agent_id}/trajectories", params=params)

    # --- decision traces -----------------------------------------------------

    def record_trace(
        self,
        agent_id: str,
        *,
        inputs: Any | None = None,
        output: str | None = None,
        tool_calls: list[dict] | None = None,
        grounding_refs: list[str] | None = None,
        occurred_at: datetime | None = None,
        latency_ms: int | None = None,
        cost_usd: float | None = None,
        token_count: int | None = None,
        trajectory_id: str | None = None,
        step_index: int | None = None,
        subject_ref: str | None = None,
        subject_type: str | None = None,
        runtime: dict | None = None,
    ) -> dict:
        """Records one decision. `inputs` is sent as-is and hashed on the
        server on arrival — it is never stored raw, but it IS transmitted
        over the wire, so send only what the server needs to compute the
        hash, not raw customer PII you can't have Coriqo see at all. (For
        data that must never leave your process, `tool_calls`' `args`/
        `result` ARE hashed locally before the request goes out — see below —
        or you can pass already-computed `args_hash`/`result_hash`
        directly.)

        `latency_ms`/`cost_usd`/`token_count` are optional telemetry — a
        caller using @track_decision gets latency_ms measured automatically;
        cost_usd/token_count are always caller-supplied since Coriqo has no
        way to measure what a decision actually cost.

        `grounding_refs` entries are Coriqo record ids (governance event ids or
        hashes, document ids) and are checked against the chain — one that
        doesn't resolve counts against the decision's grounding score. To cite
        evidence held outside Coriqo, use external_ref() so it's recorded
        without being scored as a fabricated citation.

        Pass `trajectory_id` (and optionally `step_index`) to record the
        decision as a step of a run started with start_trajectory().

        `subject_ref` is who or what the decision was about — an account
        number, a customer id, a claim number. It is HASHED LOCALLY before the
        request goes out (same function as tool-call args, see
        hash_subject_ref), so the raw identifier never leaves your process;
        Coriqo stores only the digest. That is enough to group every decision
        touching the same subject and to answer traces_by_subject(), which is
        the question that comes up when a customer complains and someone has to
        find every automated decision made about them. What it costs: the
        digest is opaque, so Coriqo can't name the subject on screen and can't
        match partially. Pass an already-computed "sha256:..." value to use
        your own hashing instead. `subject_type` ("customer", "account",
        "claim") is a plain string — keep identifiers out of it, it is not
        hashed.

        `runtime` is a DICT describing what actually produced this decision —
        {"provider": "anthropic", "model": "claude-sonnet-4", "prompt_hash":
        "...", "temperature": 0} — not a bare runtime name. It is stored
        verbatim as configuration, so keep customer data out of it. Coriqo
        fingerprints it and seals a runtime_changed event when it differs from
        this agent's previous decision, which is how a swapped model or an
        edited prompt stops being invisible under an unchanged mandate.

        Keys you put on a tool call beyond step/tool/args/result are kept as
        recorded when they're strings or numbers; anything structured is stored
        as `<key>_hash`, under the same rule that keeps args/result out of the
        record."""
        body: dict[str, Any] = {
            "inputs": inputs,
            "subject_ref_hash": hash_subject_ref(subject_ref),
            "subject_type": subject_type,
            "runtime": runtime,
            "output": output,
            "tool_calls": _hash_tool_calls(tool_calls),
            "grounding_refs": grounding_refs,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "token_count": token_count,
            "trajectory_id": trajectory_id,
            "step_index": step_index,
        }
        if occurred_at is not None:
            body["occurred_at"] = _iso_occurred_at(occurred_at)
        return self._request("POST", f"/api/v1/agents/{agent_id}/traces", json=body)

    def push_prompt(self, body: dict) -> dict:
        """Tell the prompt register about one prompt file (POST /api/v1/prompts/push).
        `body` is {path, sha256, commit?, pr_url?, author?, approvers?, ...}: a hash, never the text.
        Sending the same path, hash and commit again records nothing new."""
        return self._request("POST", "/api/v1/prompts/push", json=body)

    def get_receipt(self, event_hash: str, *, wait: float | None = None) -> dict | None:
        """Fetch the receipt bundle for a sealed event (GET /api/v1/receipts/{event_hash}).

        A 202 means no checkpoint covers the event yet: returns None. With
        wait=N it polls (2s, doubling to 30s) for up to N seconds first."""
        import time
        deadline = time.monotonic() + (wait or 0)
        delay = 2.0
        while True:
            resp = self._client.get(f"/api/v1/receipts/{event_hash}")
            if resp.status_code != 202:
                return self._parse_response(resp)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 30.0)

    @staticmethod
    def verify_receipt(bundle: dict, public_key_pem: str | None = None,
                       *, allow_unpinned: bool = False) -> dict:
        """Offline check of a receipt bundle. See coriqo_agents.receipts."""
        from .receipts import verify_receipt
        return verify_receipt(bundle, public_key_pem, allow_unpinned=allow_unpinned)

    def record_traces(self, agent_id: str, *, traces: list[dict]) -> dict:
        """Records up to 200 decisions in one call — for flushing a finished run
        rather than posting each step as it happens.

        Each item is a record_trace() body: {inputs, output, tool_calls,
        grounding_refs, occurred_at, latency_ms, cost_usd, token_count,
        trajectory_id, step_index, subject_ref, subject_type, runtime}.
        tool_calls and subject_ref are hashed locally here, exactly as
        record_trace() does, so neither raw args/result nor the raw subject
        identifier leaves the process.

        All-or-nothing: if the server rejects any decision, none are recorded
        and the error names the failing index. Each decision is still sealed as
        its own chain event — this batches the request, not the ledger."""
        if len(traces) > 200:
            # Fail here rather than sending it: a 200-item cap means an
            # oversized batch is rejected whole, so the caller would burn a
            # round-trip to learn what it can check locally.
            raise ValueError(
                f"record_traces takes at most 200 decisions per call, got {len(traces)}. "
                "Send them in chunks — each chunk is its own all-or-nothing batch."
            )
        body = []
        for t in traces:
            item = dict(t)
            if item.get("tool_calls"):
                item["tool_calls"] = _hash_tool_calls(item["tool_calls"])
            if item.get("subject_ref") is not None:
                item["subject_ref_hash"] = hash_subject_ref(item.pop("subject_ref"))
            occurred_at = item.get("occurred_at")
            if isinstance(occurred_at, datetime):
                item["occurred_at"] = _iso_occurred_at(occurred_at)
            body.append(item)
        return self._request(
            "POST", f"/api/v1/agents/{agent_id}/traces/batch", json={"traces": body},
        )

    def list_traces(self, agent_id: str, *, status: str | None = None, limit: int = 200) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._request("GET", f"/api/v1/agents/{agent_id}/traces", params=params)

    def get_trace(self, agent_id: str, trace_id: str) -> dict:
        return self._request("GET", f"/api/v1/agents/{agent_id}/traces/{trace_id}")

    def traces_by_subject(self, subject_ref: str, *, limit: int = 200) -> list[dict]:
        """Every decision recorded against this subject, across all agents in
        the tenant — the "what did our agents decide about this customer"
        lookup, for a complaint or a subject-access request.

        The ref is hashed locally with the same function record_trace() used,
        so the query matches what was stored and the raw identifier never goes
        over the wire. A ref that was never recorded is indistinguishable from
        one hashed differently: both return an empty list."""
        return self._request("GET", "/api/v1/agents/traces/by-subject", params={
            "subject_ref_hash": hash_subject_ref(subject_ref), "limit": limit,
        })

    def record_outcome(
        self, agent_id: str, trace_id: str, *, outcome: str, detail: str | None = None,
    ) -> dict:
        """Records what a human found when they reviewed this decision:
        "confirmed" (the agent was right), "overturned" (reversed), "corrected"
        (kept but changed), or "inconclusive". Returns the updated decision.

        This is the feedback half of the record — without it Coriqo can report
        how much an agent decided but not how often it was right. The outcome
        is appended to the decision's sealed chain, so it doesn't rewrite the
        original decision and can't be un-recorded; record it once the review
        is actually final."""
        if outcome not in OUTCOMES:
            raise ValueError(
                f"outcome must be one of {', '.join(OUTCOMES)} — got {outcome!r}"
            )
        return self._request(
            "POST", f"/api/v1/agents/{agent_id}/traces/{trace_id}/outcome",
            json={"outcome": outcome, "detail": detail},
        )

    # --- eval + adversarial runs ---------------------------------------------
    # Two endpoints per run type, same sealing path, different attestation:
    # the service-account route this client authenticates for, and a
    # device-signed route under /api/v1/agent-runtime. `device_signed=True`
    # targets the second, and only works if the http_client you passed to
    # __init__ signs requests (ed25519, X-Coriqo-Signature and friends) — that
    # route refuses an X-API-Key outright. This SDK does not sign; byoai-runtime
    # does. See "Device-signed posts" in the README.
    #
    # Neither method takes entry_method. Its two caller-settable values,
    # manual_entry and csv_import, both mean a person transcribed the number
    # off a report, and a harness posting its own results is by definition not
    # that. The server resolves an SDK post to api, or device_signed on the
    # signed route.

    def _runs_path(self, agent_id: str, kind: str, *, device_signed: bool) -> str:
        base = "/api/v1/agent-runtime/agents" if device_signed else "/api/v1/agents"
        return f"{base}/{agent_id}/{kind}"

    def record_eval_run(
        self,
        agent_id: str,
        *,
        suite_name: str,
        suite_version: str | None = None,
        scoring_method: str | None = None,
        metrics: list[dict] | None = None,
        artifact_inputs: Any | None = None,
        artifact_hash: str | None = None,
        hash_inputs_locally: bool = True,
        notes: str | None = None,
        occurred_at: datetime | None = None,
        device_signed: bool = False,
    ) -> dict:
        """Records one eval run and seals it onto the agent's mandate version,
        the same way a decision trace is sealed. Returns the stored run.

        `metrics` is a list of {"metric": ..., "value": ...} entries. Build
        them with eval_metric() to declare a unit — direction, scale,
        threshold — because a score reported on a 0-100 scale with no scale
        declared is read as a 0..1 score and goes unjudged by the
        eval-readiness check.

        `artifact_inputs` is the dataset or prompt set the run used. Coriqo
        stores only a sha256 of it, so by default this hashes it here and sends
        the 64-character digest instead of the inputs — nothing raw leaves the
        process. Pass `artifact_hash=` yourself if you already have the digest
        (hash_eval_artifact() computes the one the server would). Sending both
        raises: the server 422s the pair, and this catches it before the round
        trip. `hash_inputs_locally=False` sends the raw inputs for the server
        to hash, which is only worth doing to check the two canonicalisations
        still agree.

        `occurred_at` is when the suite actually ran, not when you posted it,
        and must be timezone-aware. Leave it unset and the server stamps now.

        Raises CoriqoAPIError(409) if the agent has no mandate version yet —
        there is nothing to anchor the run to until it has one.
        """
        artifact_inputs, artifact_hash = _artifact_hash_field(
            artifact_inputs, artifact_hash, hash_inputs_locally,
        )
        body: dict[str, Any] = {
            "suite_name": suite_name,
            "suite_version": suite_version,
            "scoring_method": scoring_method,
            "metrics": metrics,
            "artifact_inputs": artifact_inputs,
            "artifact_hash": artifact_hash,
            "notes": notes,
            "occurred_at": _iso_occurred_at(occurred_at),
        }
        return self._request(
            "POST", self._runs_path(agent_id, "eval-runs", device_signed=device_signed), json=body,
        )

    def record_adversarial_run(
        self,
        agent_id: str,
        *,
        suite_name: str,
        attempt_class: str,
        suite_version: str | None = None,
        attempt_count: int | None = None,
        success_rate: float | None = None,
        exemplars: list[Any] | None = None,
        artifact_inputs: Any | None = None,
        artifact_hash: str | None = None,
        hash_inputs_locally: bool = True,
        notes: str | None = None,
        occurred_at: datetime | None = None,
        device_signed: bool = False,
    ) -> dict:
        """Records one adversarial-testing run and seals it onto the agent's
        mandate version. Same artifact and timestamp rules as
        record_eval_run().

        Where an eval run reports arbitrary metrics, an adversarial run reports
        the same two facts every time, so they are their own fields:
        `attempt_class` ("prompt_injection", "jailbreak", "tool_misuse", or
        whatever class your suite reports) and `success_rate`, the fraction of
        attempts that got through, between 0 and 1.

        `exemplars` are attack transcripts. Whether they are stored is decided
        server-side by the tenant's payload_mode — on a hash-only tenant they
        are dropped. Do not send anything you need kept, and check
        `payload_mode` on the returned run to see what happened.
        """
        artifact_inputs, artifact_hash = _artifact_hash_field(
            artifact_inputs, artifact_hash, hash_inputs_locally,
        )
        body: dict[str, Any] = {
            "suite_name": suite_name,
            "suite_version": suite_version,
            "attempt_class": attempt_class,
            "attempt_count": attempt_count,
            "success_rate": success_rate,
            "exemplars": exemplars,
            "artifact_inputs": artifact_inputs,
            "artifact_hash": artifact_hash,
            "notes": notes,
            "occurred_at": _iso_occurred_at(occurred_at),
        }
        return self._request(
            "POST",
            self._runs_path(agent_id, "adversarial-runs", device_signed=device_signed),
            json=body,
        )

    def list_eval_runs(self, agent_id: str) -> list[dict]:
        """This agent's eval runs, newest first — what the Evaluations tab
        shows. Needs model:read on the service account."""
        return self._request("GET", f"/api/v1/agents/{agent_id}/eval-runs")

    def list_adversarial_runs(self, agent_id: str) -> list[dict]:
        """This agent's adversarial-testing runs, newest first."""
        return self._request("GET", f"/api/v1/agents/{agent_id}/adversarial-runs")

    def eval_readiness(self, agent_id: str) -> dict:
        """Whether the recorded evaluation record is current and inside the
        tenant's thresholds, with the applied bar on the response as
        `thresholds`. Advisory — it reads what has been recorded and never
        sets validation status or blocks a promotion."""
        return self._request("GET", f"/api/v1/agents/{agent_id}/eval-readiness")

    # --- grounding + assurance ----------------------------------------------

    def verify_trace_grounding(self, agent_id: str, trace_id: str) -> dict:
        return self._request("GET", f"/api/v1/agents/{agent_id}/traces/{trace_id}/grounding")

    def get_assurance(self, agent_id: str) -> dict:
        return self._request("GET", f"/api/v1/agents/{agent_id}/assurance")

    def run_assurance(self, agent_id: str) -> dict:
        return self._request("POST", f"/api/v1/agents/{agent_id}/assurance/run")
