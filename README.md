# coriqo-agents

A thin Python client for reporting AI agent decisions to Coriqo's agent
monitoring API. Ask your Coriqo contact for the agent monitoring integration
guide for the full contract — what gets hashed vs stored, what the
mandate-scope and grounding checks verify, and how to create the service
account this client authenticates as. (It's maintained as
`partners_doc/agent_monitoring_integration_guide.md` internally, but that
directory is gitignored and handed to partners directly rather than
published in this repo's history.)

## Install

```
pip install -e sdk/coriqo-agents
# with the LangChain callback, or the OpenAI Agents SDK processor:
pip install -e "sdk/coriqo-agents[langchain]"
pip install -e "sdk/coriqo-agents[openai-agents]"
```

Not yet published to PyPI. Install from the repo path for now.

## Which way in

Pick the first one that fits. You only need one.

1. **You already send OpenTelemetry.** Point your OTLP/HTTP exporter at
   Coriqo and install nothing. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to your
   Coriqo URL (the exporter adds `/v1/traces`; `/api/v1/agents/otlp/traces`
   also works behind a path prefix) and send the service-account headers:
   `OTEL_EXPORTER_OTLP_HEADERS="X-API-Key=cq_sa_...,X-Tenant-Slug=acme_bank"`.
   JSON and protobuf are both accepted, gzip too. Name the agent with the
   span attribute `coriqo.agent_id`; without it, `service.name` is used.
2. **You build on the OpenAI Agents SDK.** Add `CoriqoTracingProcessor`
   (below). One `Runner.run` is one decision.
3. **You build on LangChain.** Pass `CoriqoCallbackHandler` in the run's
   callbacks (below). One top-level `invoke` is one decision.
4. **Anything else.** Wrap the decision function with `@track_decision`, or
   call `client.record_trace` yourself.

## Wiring your agent

```python
from coriqo_agents import CoriqoAgentClient

client = CoriqoAgentClient(
    base_url="https://your-coriqo-instance.example.com",
    api_key="cq_sa_...",       # from POST /api/v1/service-accounts
    tenant_slug="acme_bank",   # returned alongside the key at creation
)

agent = client.register_agent(
    name="loan-underwriting-agent",
    mandate="Approve or deny consumer loan applications under $50,000",
    system="loan-origination",
    risk_tier="high",
    allowed_tools=["credit_bureau_lookup", "approve_loan", "deny_loan"],
)
agent_id = agent["agent_id"]

# after your agent finishes a decision:
client.record_trace(
    agent_id,
    inputs={"applicant_id": "a-123", "requested_amount": 12000},
    output="Approved: income verified, credit score 720, DTI 0.31 within policy",
    tool_calls=[
        {"step": 1, "tool": "credit_bureau_lookup", "args": {"applicant_id": "a-123"}, "result": {"score": 720}},
        {"step": 2, "tool": "approve_loan", "args": {"amount": 12000}, "result": {"status": "approved"}},
    ],
    grounding_refs=["evt_a1b2c3..."],
    subject_ref="acct-88431",       # hashed locally, see below
    subject_type="account",
    runtime={"provider": "anthropic", "model": "claude-sonnet-4",
             "prompt_hash": "sha256:...", "temperature": 0},
)
```

`args`/`result` in `tool_calls` are hashed locally before the request goes
out, so the raw values never leave your process. `inputs` is different: it IS
sent to Coriqo as-is, then hashed on arrival and never stored. Don't put
anything in `inputs` you can't have Coriqo's backend see at all, even briefly
in transit; use `tool_calls`' local hashing (or your own pre-computed hash)
for that.

## Subjects

`subject_ref` says who or what a decision was about — an account number, a
customer id, a claim number. It is hashed in your process (`hash_subject_ref`,
the same sha256 used for tool-call args) before the request is sent, so Coriqo
holds only the digest.

What you get: every decision about the same subject groups together, and
`traces_by_subject` answers "what did our agents decide about this person"
when a complaint or a subject-access request lands. What it costs: the digest
is opaque, so Coriqo can't display the subject's name anywhere, can't match on
a partial reference, and can't reverse the hash for you. Anyone querying has to
already know the raw identifier.

```python
for trace in client.traces_by_subject("acct-88431", limit=50):
    print(trace["occurred_at"], trace["agent_id"], trace["output"])
```

Pass a value already starting `sha256:` if you hash identifiers yourself.
`subject_type` is stored as plain text — keep identifiers out of it.

`runtime` is a **dict** describing what actually produced the decision
(provider, model, prompt hash, settings), stored verbatim as configuration —
so keep customer data out of it. Coriqo fingerprints it and seals a
`runtime_changed` event when it differs from this agent's previous decision:
that is how a swapped model or an edited prompt becomes visible under an
otherwise unchanged mandate. Passing a bare string here is rejected 422.

## Review outcomes

When a human reviews a decision, record what they found. Without this Coriqo
can report how much an agent decided but not how often it was right.

```python
client.record_outcome(
    agent_id,
    trace_id,
    outcome="overturned",   # confirmed | overturned | corrected | inconclusive
    detail="Underwriter reversed on appeal — bureau file had a duplicate tradeline",
)
```

Anything outside those four values raises `ValueError` before a request is
made. The outcome is appended to the decision's sealed chain rather than
rewriting it, so it can't be taken back — record it once the review is final.

## Failed runs

A run that crashed or timed out should be closed as failed, not left open. An
open trajectory reads as still running, which makes a stalled fleet look
healthy.

```python
run = client.start_trajectory(agent_id, goal="Underwrite application a-123")
try:
    ...  # your steps, each recorded with trajectory_id=run["trajectory_id"]
except Exception as e:
    client.fail_trajectory(agent_id, run["trajectory_id"], reason=f"{type(e).__name__}: {e}")
    raise
else:
    client.complete_trajectory(agent_id, run["trajectory_id"])
```

Close any sub-runs first — both `fail_trajectory` and `complete_trajectory`
seal a final status.

## Eval and adversarial runs

When an eval suite or a red-team suite finishes, post the result. It is sealed
onto the agent's current mandate version, the same chain a decision trace
lands on, so the evidence says which mandate the numbers were measured
against.

```python
from datetime import datetime, timezone
from coriqo_agents import eval_metric, hash_eval_artifact

dataset = {"dataset": "consumer-loan-holdout-2026Q2", "rows": 4821, "seed": 17}

run = client.record_eval_run(
    agent_id,
    suite_name="hallucination-bench",
    suite_version="2026.2",
    scoring_method="llm_judge:claude-opus",
    metrics=[
        eval_metric("accuracy", 87.0, direction="higher_is_better",
                    scale=(0, 100), threshold=80.0),
        eval_metric("hallucination_rate", 0.03, direction="lower_is_better"),
    ],
    artifact_inputs=dataset,
    occurred_at=datetime.now(timezone.utc),
    notes="Quarterly holdout run",
)
print(run["artifact_hash"] == hash_eval_artifact(dataset))   # True

client.record_adversarial_run(
    agent_id,
    suite_name="promptfoo-redteam",
    suite_version="0.9.1",
    attempt_class="prompt_injection",
    attempt_count=250,
    success_rate=0.012,
    artifact_inputs={"corpus": "owasp-llm-top10", "variants": 250},
)
```

Read them back with `list_eval_runs(agent_id)` and
`list_adversarial_runs(agent_id)`. `eval_readiness(agent_id)` reports whether
the recorded evaluation record is current and inside the tenant's thresholds;
it is advisory and never blocks a promotion.

Both calls need a mandate version on the agent — a run has nothing to anchor
to without one, and you get a `CoriqoAPIError` with status 409.

### Declare what your numbers mean

`eval_metric("accuracy", 0.94)` is enough to record a score. It is not enough
to have it judged if the number is not on a 0..1 scale: 87 with no declared
scale reads as a score of 87 on a 0..1 metric, and the readiness check skips
it. Four optional arguments fix that:

- `direction` — `"higher_is_better"` or `"lower_is_better"`, overriding the
  guess made from the metric name. A direction is not a unit, so it does not
  on its own make a value outside 0..1 judgeable.
- `scale` — the bounds you report in, `(0, 100)` for a percentage. Also
  accepts `{"min": 0, "max": 100}`.
- `threshold` — an explicit bar in the metric's own units, compared directly,
  with no normalization and no tenant floor. Wins over `scale`.
- `threshold_direction` — only when the bar points the other way from
  `direction`.

Leave one unset and it is left out of the request, so a bare
`{"metric", "value"}` entry stores exactly what it always did.

### The eval inputs are hashed locally

Coriqo stores a sha256 of an eval's inputs and never the inputs themselves.
Pass `artifact_inputs=` and the SDK computes that digest in your process and
sends the 64 hex characters — the dataset does not cross the network at all.
`hash_eval_artifact()` computes the same digest on its own if you want to
record it elsewhere.

The digest matches what the server derives from the same object, byte for
byte; `tests/test_eval_runs.py` posts one dataset both ways and asserts the
two stored hashes are equal, so a drift in either canonicalisation fails a
test.

Send `artifact_inputs` or `artifact_hash`, never both. A supplied hash beside
the real inputs could contradict them, and the sealed record promises the hash
is re-derivable from what was hashed, so the server rejects the pair with a
422 and the SDK raises a `ValueError` before spending the round trip.

`hash_inputs_locally=False` sends the raw inputs for the server to hash. The
only reason to reach for it is checking that the two canonicalisations still
agree.

### entry\_method

Neither call takes an `entry_method` argument. Its two caller-settable values,
`manual_entry` and `csv_import`, both mean a person transcribed the number off
a report, and a harness posting its own results is not that. The server
records an SDK post as `api`. Passing `entry_method=` raises a `TypeError`.

### Device-signed posts

The same two operations also exist under `/api/v1/agent-runtime`, signed by an
enrolled device keypair rather than a service-account key. Pass
`device_signed=True` to target those paths. This SDK does not sign requests
(byoai-runtime does), so it only works if the `http_client` you handed to
`CoriqoAgentClient` adds the ed25519 signature headers itself. The runtime
route refuses an `X-API-Key` outright. The server records those posts as
`device_signed` and a body field cannot override it.

## Silence detection

`register_agent(..., expected_report_interval_minutes=60)` tells Coriqo how
often this agent should be reporting. An agent that goes quiet past that
interval is surfaced as silent instead of healthy. Set it from the agent's
real cadence including idle periods — too tight a value alerts every weekend.
Leave it unset and no silence check runs.

## Decorator

For an agent that isn't built on LangChain — wrap the function that makes
the decision and every call is recorded automatically:

```python
from coriqo_agents import track_decision

@track_decision(client, agent_id)
def approve_loan(applicant_id: str, requested_amount: int) -> str:
    # ... your decision logic ...
    return "Approved: income verified, credit score 720"
```

The function's arguments become `inputs` (bound to their parameter names,
e.g. `{"applicant_id": "a-123", "requested_amount": 12000}`), its return
value becomes `output`. A raised exception is still recorded (as the
output) and then re-raised — the decorator never swallows your function's
errors. Works on `async def` functions too.

To tag the subject, pass a function of the bound inputs (or a plain string if
every call is about the same subject); it goes through the same local hashing:

```python
@track_decision(client, agent_id, subject_ref=lambda i: i["applicant_id"],
                subject_type="applicant",
                runtime={"provider": "openai", "model": "gpt-4o"})
def approve_loan(applicant_id: str, requested_amount: int) -> str:
    ...
```

Only captures inputs/output;
for `tool_calls`/`grounding_refs`, call `record_trace` directly instead (or
from inside the decorated function, alongside your own logic).

## LangChain

```python
from coriqo_agents.langchain import CoriqoCallbackHandler

handler = CoriqoCallbackHandler(client, agent_id)
agent.invoke({"input": question}, config={"callbacks": [handler]})
```

One top-level run is one decision. Nested sub-chains fold into it. Each tool
call is paired with its result by LangChain's `run_id`, so parallel tool calls
cannot swap results, and a tool that raised is recorded with its error. The
model that answered is the trace's `runtime`, tokens are summed across the
run, and latency is measured. A run that raised is still recorded, as
`Chain run failed: ...`. If Coriqo is unreachable the trace goes to the local
outage buffer instead of failing your run. A suspended agent makes `invoke`
raise `AgentSuspendedError`; the trace is sent when the run ends, so that run
has already acted. To stop a suspended agent before it acts, check
`client.get_agent(agent_id)` first. The handler can't set `grounding_refs` (LangChain has
no notion of which sealed Coriqo record a decision relied on): call
`record_trace` yourself when you need them.

## OpenAI Agents SDK

```python
from agents import add_trace_processor
from coriqo_agents.openai_agents import CoriqoTracingProcessor

add_trace_processor(CoriqoTracingProcessor(client, agent_id="credit-limit-agent",
                                           agent_ids={"Underwriter": "underwriting-agent"}))
```

Register it once at start-up. Each `Runner.run` becomes one decision: function
tools as tool calls, a handoff as `handoff:<agent>`, a tripped guardrail as
`guardrail:<name>`, the model as `runtime`, tokens summed and latency measured.
`agent_ids` maps SDK agent names to Coriqo agents (the first agent that ran
decides); unmapped runs use `agent_id`. Recording never raises into the run:
a failed send goes to the local outage buffer, and a suspended agent is logged
at error level because the run has already finished when its trace ends. To
stop a suspended agent before it acts, check `client.get_agent(agent_id)` first.

## Webhook

For an agent platform that can only "call a webhook" (Zapier, Make, n8n, an
internal RPA/orchestration tool) rather than run this SDK directly, there
are two levels:

- **No Python at all**: point the platform's webhook action straight at
  `POST /api/v1/agents/{agent_id}/traces` with the `X-API-Key`/
  `X-Tenant-Slug` headers and a JSON body — the API accepts raw
  `args`/`result` in `tool_calls` directly (hashed on arrival, server-side,
  never stored) precisely so a no-code tool that can't compute a sha256
  hash inline still gets the same guarantee. See the integration guide's
  "No-code / webhook-only integration" section for the full contract. This
  is the simplest path if you have nowhere to run Python at all.
- **You do run some Python, just not this specific agent's code**: use
  `handle_webhook` below to receive the platform's webhook in your own
  route and forward it, still going through this SDK.

```python
from fastapi import FastAPI, Request
from coriqo_agents import CoriqoAgentClient, handle_webhook

app = FastAPI()
client = CoriqoAgentClient(base_url="...", api_key="...", tenant_slug="...")

@app.post("/coriqo-webhook/{agent_id}")
async def coriqo_webhook(agent_id: str, request: Request):
    return handle_webhook(client, agent_id, await request.json())
```

The payload should already be shaped like the `/traces` contract above —
`inputs`/`output`/`tool_calls`/`grounding_refs`/`occurred_at`/`subject_ref`/
`subject_type`/`runtime`, any subset;
other keys are ignored. This is a thin pass-through, not a universal
webhook-format translator. If your platform's payload looks different,
reshape it to this contract in your route before calling `handle_webhook`
(or call `record_trace` directly with whatever fields you already have).
This module ships no web server of its own, so it adds no framework
dependency to the SDK. The snippet above works the same way in Flask,
Django, or anything else that can hand you a parsed JSON dict.

## Limits

Same ones the integration guide documents: assurance checks scan the most
recent 2,000 decisions per agent; trace listing returns up to 1,000 per
call. This client doesn't paginate for you.

## The assurance runner (`coriqo_agents.runner`)

The customer-side assurance runner. Validators run inside your own
environment, not on Coriqo's servers, so the runner ships in this SDK, next
to telemetry.

It provides the `Validator` base class (`execute`/`evaluate`/`produce_evidence`),
`RunnerEngine` (runs a compiled protocol bundle's validators, collects each
one's evidence, and builds the signed execution manifest), `DeviceSigner`
(Ed25519 signing over the same request tuple Coriqo's API already verifies
for enrolled devices, so there is no second scheme), and `validators.resolve_validators()`
/ `VALIDATOR_REGISTRY`, which map a compiled bundle's declared validator ids
to the real classes.

`VALIDATOR_REGISTRY` is the authoritative list (49 validators at the time of
writing). `protocol_loader.py` holds the matching per-template roster: 15
protocol templates declaring 49 validator ids between them. Two of the
templates:

- **High-Risk Agent Assurance v1.0** (five): `tool_authorization`,
  `prompt_injection`, `data_exfiltration`, `action_integrity`,
  `policy_compliance`.
- **Emergency Agent Control Validation v1.0** (six kill-switch checks):
  `ks_new_task_rejection`, `ks_active_execution_halt`,
  `ks_tool_access_disabled`, `ks_credential_revocation`,
  `ks_shutdown_auditability`, `ks_recovery_process`.

`stub_validators.StubToolAuthorizationValidator` still exists (always PASS,
clearly labeled) for exercising the pipeline in isolation, but a real run
should resolve validators from the bundle, not import the stub directly.

`ValidatorContext.configuration` keys a caller commonly sets:

| Key | Used by | Meaning |
|---|---|---|
| `authorized_tools` | `tool_authorization` | The tool-name allowlist a call outside it fails against. |
| `policy_rules` | `policy_compliance` | Rules to check tool calls against; omitted entirely, not an empty list, when none apply — the validator reports `NOT_APPLICABLE` rather than a false PASS. |
| `shutdown_grace_ms` | `ks_tool_access_disabled` | How long after a `SHUTDOWN_SIGNAL` event a still-in-flight tool call is tolerated before it counts as a FAIL (default 2000). |

Needs the `runner` extra for Ed25519 signing:

```
pip install -e "sdk/coriqo-agents[runner]"
```

```python
from coriqo_agents.runner import DeviceSigner, ExecutionUploader, RunnerEngine, ValidatorContext
from coriqo_agents.runner.validators import resolve_validators

# `bundle` is what POST /api/v1/assurance/protocols/compile (or a later
# GET) returned — {id, name, version, template, bundle_hash, ...}.
engine = RunnerEngine(bundle)
validators = resolve_validators(bundle)  # the bundle's declared validator ids -> real classes
context = ValidatorContext(
    execution_id="ae_123", agent_model_id="model_1", agent_version_ref="ver_1",
    environment="PRODUCTION",
)
results, manifest, evidence, failure_scenarios = engine.run(validators, context)

signer = DeviceSigner(private_key=your_enrolled_device_key)  # from enrollment
uploader = ExecutionUploader(base_url="https://your-coriqo-instance.example.com",
                             signer=signer, tenant_slug="acme_bank")
uploader.upload_manifest("ae_123", manifest)
uploader.upload_results("ae_123", [r.to_dict() for r in results], evidence, failure_scenarios)
```

The runner never calls an LLM and never needs an API key beyond its
enrolled device key. It has no LLM dependencies.

## Mandate check: no mandate, no deploy

Put this in a release pipeline ahead of the deploy step. It asks Coriqo whether an agent may go live and fails the job when the answer is no.

```bash
export CORIQO_BASE_URL=https://your-coriqo.example CORIQO_TENANT_SLUG=your_bank CORIQO_API_KEY=...
coriqo-agents mandate check --agent-id <agent id> --mandate-file mandate.txt
```

It checks that the agent is active, has a written mandate, has a mandate a named person approved, and, with `--mandate-file`, that the file in your repository is the mandate Coriqo holds. `--allow-unapproved` drops the approval requirement. A pending mandate change does not block.

Exit codes: `0` allowed, `1` blocked (the output says why and links to the agent), `2` could not ask (missing settings, bad key, no network). A job that cannot ask fails too, so an outage never lets an unchecked agent ship.

From code: `client.deploy_check(agent_id, mandate_sha256=..., require_approved=True)` returns `{"allowed": bool, "reasons": [...], "checks": [...]}`. `coriqo_agents.hashing.mandate_sha256(text)` hashes a mandate the same way Coriqo does: line endings and trailing whitespace are ignored, inner whitespace is not.

Ready-made jobs: `ci/github-mandate-check.yml` and `ci/gitlab-mandate-check.yml`.

## Receipts: prove the record wasn't modified

After sending a decision trace or eval run, you can fetch a receipt and verify it offline — proving your record reached Coriqo unaltered. A receipt chains your decision to a sealed checkpoint signed by Coriqo and anchored to Sigstore Rekor (and optionally an RFC 3161 TSA), so an examiner can verify without trusting Coriqo.

```python
# Fetch the receipt
receipt = client.get_receipt(event_hash, wait=60)
if receipt is None:
    print("Receipt still pending. Check again later.")
else:
    # Verify offline against public keys
    result = client.verify_receipt(receipt)
    if result["ok"]:
        print("Record verified — sealed in Coriqo's checkpoint")
        print(f"Merkle root matches external anchor: {result['checks']['root_in_sth']['status']}")
    else:
        for check, detail in result["checks"].items():
            if detail["status"] == "fail":
                print(f"FAIL: {check} — {detail['reason']}")
            elif detail["status"] == "not_checked":
                print(f"SKIPPED: {check} — {detail['reason']}")
```

`record_trace()` returns the `event_hash` on the result object, so you can fetch later. `get_receipt()` polls at 2-second intervals (doubling to 30s) for up to `wait` seconds if the receipt isn't ready yet; omit `wait` to return immediately (returns `None` if still pending).

`verify_receipt()` checks five properties offline:
- **leaf_inclusion** — your record hashes correctly into the Merkle tree.
- **root_in_sth** — the tree's root appears in the sealed checkpoint.
- **signature** — Coriqo's Ed25519 signature over the checkpoint is valid.
- **tsa_imprint** — RFC 3161 token's digest matches the checkpoint. (Appears as `not_checked` offline unless you configure a TSA.)
- **rekor** — entry appears in Sigstore Rekor. (Appears as `not_checked` offline unless `REKOR_URL` is set on the server.)

`key_pinned` is `pass` only when you supply `public_key_pem=`. `ok` is `true` only when `leaf_inclusion`, `root_in_sth`, `signature` and `key_pinned` all pass. Without a pinned key the bundle's own key is used, `key_pinned` is `not_checked` and `ok` is `false`; pass `allow_unpinned=True` to accept that. A `tsa_imprint` fails if the token's own signature is invalid.

Pending means the event was recorded but hasn't been sealed into a checkpoint yet; try again after the next checkpoint interval. Not_checked means the check cannot be performed offline or would require external services.

Auth: you need role `receipt:read` and must be the decision's original actor.

```
export CORIQO_API_KEY=cq_sa_...
coriqo-agents receipt verify <event_hash> \
    --base-url https://your-coriqo-instance.example.com \
    --tenant acme_bank \
    --pubkey coriqo-signing-key.pem
```

Set the key in `CORIQO_API_KEY`. `--api-key` also works but is discouraged, since the value shows up in `ps`.

Use `--file bundle.json` to verify a saved receipt, `--pubkey file.pem` to pin a key, and `--wait 120` to keep polling. Exit 0 if verified against a pinned key, 1 if a check fails, 2 if still pending or usage error, 3 if the receipt is valid but the key was not pinned (add `--pubkey`, or `--allow-unpinned` to accept the bundle's key).

### Engine adapters — running Promptfoo from the command line

The runner can also drive a third-party red-team tool instead of its own
validators. Coriqo's servers never invoke that tool: it runs here, in your
environment, against a target Coriqo never sees.

```
python -m coriqo_agents.runner \
    --execution ae_123 --engine promptfoo \
    --bundle bundle.json --target target.json --scope scope.json \
    --workdir ./run \
    --base-url https://your-coriqo-instance.example.com \
    --tenant acme_bank --device-key ./device.key
```

`--dry-run` writes the engine's config and prints the plan without executing
anything or uploading anything, so you can read exactly what would be sent at
your agent before it is.

`target.json` points at the agent and never contains a secret:

```json
{
  "endpoint_ref": "https://agent.internal.example/chat",
  "auth_env": "AGENT_TOKEN",
  "purpose": "Handles retail payment queries.",
  "transform_response": "json.output"
}
```

`auth_env` is the NAME of an environment variable you set in the shell that
runs the command. The token's value never enters the config file, the plan
hash, the manifest, or anything uploaded to Coriqo.

Promptfoo itself is a Node package you install (`npm install -g promptfoo`);
this SDK only shells out to it and adds no Python dependency. Pinned and
tested against promptfoo 0.122.2. `--engine manual` records testing a person
did by hand, reading a `manual_results.json` from the workdir; Coriqo labels
those results as unattested.

What Coriqo receives from either engine: the normalized results, the engine's
id and version in the signed manifest, and the tool's own output stored as
hashed evidence.

### Example scripts

Three runnable end-to-end scripts in `examples/`, all HTTP-only against a
Coriqo instance at `http://localhost:8000` by default:

- **`otel_demo.py`** — one standalone execution against a synthetic OTLP
  export of a prompt-injected wire transfer, ingested through
  `otel_ingest.otlp_json_to_trace` and run through the five agent-assurance
  validators. Needs an `AssuranceExecution` row created on the server first;
  the script's docstring explains why.
- **`kill_switch_demo.py`** — the same standalone-execution shape, against a
  hand-built kill-switch drill trace and the six `ks_*` validators.
- **`campaign_demo.py`** — the full campaign story, HTTP-only end to end (no
  setup step needed, because `POST /assurance/campaigns/{id}/executions`
  returns a real execution id directly): compile both protocols, create and
  approve an `AGENT_ASSURANCE` campaign, run it against the injected-wire-
  transfer trace, retest clean, determine, export the v1.1 PDF, then repeat
  run/determine/export for an `EMERGENCY_CONTROL_TEST` campaign. Prints
  every URL it creates.
