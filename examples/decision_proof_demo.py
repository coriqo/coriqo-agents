"""One agent decision, end to end: the agent posts it, a checkpoint seals it,
the proof is downloaded, and the public verifier checks it offline. Then the
two tool calls are swapped and the same check fails.

Needs the local stack (`make dev`, acme_bank seeded) and a copy of the public
verifier (https://github.com/coriqo/coriqo-seal-verifier):

    PYTHONPATH="$PWD/sdk/coriqo-agents" AGENT_ID=<an agent's id> \\
    VERIFIER=<path>/verify_proof.py OUT_DIR=/tmp/decision-proof \\
        python sdk/coriqo-agents/examples/decision_proof_demo.py

Signs in as the seeded admin for brevity; a real agent uses a service-account
API key (CoriqoAgentClient(base_url=..., api_key=..., tenant_slug=...)).
Exit code 0 when the real proof verifies and the tampered one fails."""
import json, os, subprocess, sys, time
import httpx
from coriqo_agents.client import CoriqoAgentClient

BASE = os.environ.get("CORIQO_BASE_URL", "http://localhost:8000")
SLUG = os.environ.get("CORIQO_TENANT_SLUG", "acme_bank")
AGENT = os.environ["AGENT_ID"]
VERIFIER = os.environ["VERIFIER"]
OUT = os.environ["OUT_DIR"]

tok = httpx.post(f"{BASE}/api/v1/auth/token", json={
    "username": "admin", "password": "admin123", "tenant_slug": SLUG}).json()["access_token"]
http = httpx.Client(base_url=BASE, headers={"Authorization": f"Bearer {tok}"}, timeout=60)
agent = CoriqoAgentClient(http_client=http)

print("1. agent posts a decision")
t = agent.record_trace(
    AGENT,
    inputs={"application_id": "APP-1042", "requested": 25000},
    output="Refer to underwriter: debt-to-income 46% is above the 43% limit.",
    tool_calls=[
        {"step": 1, "tool": "credit_bureau_lookup", "args": {"id": "APP-1042"}, "result": {"score": 702}},
        {"step": 2, "tool": "dti_calculator", "args": {"id": "APP-1042"}, "result": {"dti": 0.46}},
    ],
    subject_ref="APP-1042", subject_type="application",
    runtime={"provider": "anthropic", "model": "claude-sonnet-5-5", "temperature": 0},
)
tid = t["trace_id"]
print("   trace", tid, "status", t.get("status"))

print("2. checkpoint seals it")
r = http.post("/api/v1/checkpoints", json={"subject": "org_default"})
print("   checkpoint", r.status_code, r.json().get("checkpoint_id") or r.text[:200])

print("3. download the proof")
p = None
for _ in range(10):
    p = http.get(f"/api/v1/agents/{AGENT}/traces/{tid}/proof")
    if p.status_code == 200:
        break
    time.sleep(2)
assert p is not None
print("   proof", p.status_code)
p.raise_for_status()
os.makedirs(OUT, exist_ok=True)
proof = os.path.join(OUT, "decision.json")
open(proof, "w").write(p.text)
pem = http.get("/api/v1/checkpoints/public-keys").json()["keys"][0]["public_key_pem"]
key = os.path.join(OUT, "pubkey.pem")
open(key, "w").write(pem)

print("4. verify offline")
ok = subprocess.run([sys.executable, "-I", VERIFIER, "--trace", proof, "--pubkey", key])
d = json.load(open(proof))
d["trace"]["tool_calls"].reverse()
bad = os.path.join(OUT, "tampered.json")
json.dump(d, open(bad, "w"))
print("5. swap the order of the two tool calls and verify again")
ko = subprocess.run([sys.executable, "-I", VERIFIER, "--trace", bad, "--pubkey", key])
print("RESULT", "real proof:", ok.returncode, "tampered:", ko.returncode)
sys.exit(0 if ok.returncode == 0 and ko.returncode == 1 else 1)
