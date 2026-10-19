# OTel ingestion sanity check

`otel_demo.py` is the first example app for the `coriqo-agents` SDK. It
exists to answer one question by actually running it, not by reading the
code: does a real OpenTelemetry trace export turn into something the
assurance runner's five validators can evaluate, end to end, against the
real API?

It builds one OTLP/HTTP+JSON export representing an agent that got prompt-
injected into an unauthorized wire transfer (an authorized `TOOL_CALL`,
`customer_notes_lookup`, whose result carries an injected instruction; the
agent then calls `wire_transfer_system`, which is neither authorized nor
the tool the injected text named literally, but matches the demo's intent-
following check), converts it with `otel_ingest.otlp_json_to_trace`, runs
the five real validators against the result, and uploads the signed
manifest + results through the same device-auth path a real runner
deployment uses. Expect `tool_authorization` and `prompt_injection` to FAIL
— that's the point, not a bug in the demo.

## Option 1 — run it directly on the host (simplest)

Needs: the stack up and seeded (`make dev`), and `docker` on your PATH (used
for exactly one step — see "Why one step needs `docker compose exec`" below).

```bash
PYTHONPATH="$PWD:$PWD/sdk/coriqo-agents" \
    python sdk/coriqo-agents/examples/otel_demo.py
```

That's it — it logs in as `admin`/`admin123` on `acme_bank`, registers a
demo agent, creates an engagement, compiles the protocol, bootstraps an
execution, enrolls a device, builds and converts the OTLP trace, runs the
validators, and uploads the results. Read the `[otel_demo]` lines on
stderr for what it's doing at each step, and the final JSON dump for the
execution's rolled-up status.

## Option 2 — run it inside Docker

The `demo-agent` compose service (profile `demo`) runs the exact same
script inside the compose network, talking to `api:8000` instead of
`localhost:8000`. It has no docker socket of its own, so it can't do the
one `docker compose exec` step itself — do that from the host first, then
hand the container the `execution_id` it produced:

```bash
# 1. Bootstrap the execution row (see "Why one step needs docker compose
#    exec" below for what this is and why it's not just another HTTP call).
#    Run option 1 once first if you don't already have these ids — it prints
#    each one as it creates it (model_id, mandate_version_id, engagement_id,
#    protocol id/version).
EXECUTION_ID=$(docker compose exec -T \
    -e TENANT_SCHEMA=t_acme_bank \
    -e AGENT_MODEL_ID=<model_id> \
    -e AGENT_VERSION_ID=<mandate_version_id> \
    -e ENGAGEMENT_ID=<engagement_id> \
    -e PROTOCOL_ID=<protocol_id> \
    -e PROTOCOL_VERSION=<protocol_version> \
    api python -m api.seed.assurance_demo_bootstrap_execution \
    | tail -1 | python3 -c 'import json,sys; print(json.load(sys.stdin)["execution_id"])')

# 2. Run the demo agent container against it.
docker compose --profile demo run --rm -e EXECUTION_ID="$EXECUTION_ID" demo-agent
```

Each `EXECUTION_ID` is single-use — step 1's manifest/results upload is a one-shot per execution, like a real runner's would be. Bootstrap a fresh one for each `demo-agent` run rather than reusing an id from a previous run: a second results upload against the same execution is rejected with a clean 400 (`manifest_service.ResultsAlreadyIngestedError`) — this used to be a raw 500 from a duplicate evidence-record key, since results are content-addressed, until that got fixed alongside this demo.

## Why one step needs `docker compose exec`

Every other step here — login, agent registration, engagement creation,
protocol compile, device enrollment — has a real HTTP endpoint a genuine
customer integration would call, and the script only ever uses those.
Creating the `AssuranceExecution` row itself does not: there is no
`POST /api/v1/assurance/executions` in the API today (it is
not built yet; the demo bootstrap seed script creates the row instead). Every
place a row gets created today does it the same way that script does —
direct ORM construction, run where `DATABASE_URL` is the real tenant
Postgres (inside the `api` container), the same pattern every
server seed script already follows. If a real create-execution endpoint
ships, delete that script and this README's step 1 in favor of it.

## Configuration

All read from the environment, with defaults matching a freshly-seeded
`make dev` stack — see `otel_demo.py`'s own docstring for the full list
(`CORIQO_BASE_URL`, `CORIQO_TENANT_SLUG`, `CORIQO_TENANT_SCHEMA`,
`CORIQO_ADMIN_USERNAME`, `CORIQO_ADMIN_PASSWORD`, `EXECUTION_ID`,
`COMPOSE_API_SERVICE`).

## `decision_proof_demo.py`: check one decision yourself

Posts one loan decision with two tool calls, issues a checkpoint, downloads
the decision's proof, and runs the public verifier
([coriqo-seal-verifier](https://github.com/coriqo/coriqo-seal-verifier)) on
it with the pinned key. It then swaps the two tool calls and runs the check
again, which fails. Run instructions are in the script's docstring. Exit code 0
means the real proof verified and the changed one failed.
