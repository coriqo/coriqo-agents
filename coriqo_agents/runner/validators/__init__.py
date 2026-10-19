"""The real validators, replacing stub_validators.py's single stub
under the same validator_id ("tool_authorization") plus the four new ones.

`VALIDATOR_REGISTRY` is what `RunnerEngine` (or any caller building a
validator list from a compiled protocol bundle) resolves validator ids to —
the bundle only ever names validators by id+version (protocol_loader.py's
HARDCODED_TEMPLATE), so something has to map "tool_authorization" back to a
class to run. The first version had no such mapping because it only ever ran
one stub by direct reference; `resolve_validators` replaced that wiring.
"""
from __future__ import annotations

from ..contract import Validator
from .action_integrity import ActionIntegrityValidator
from .chain_data_boundary import ChainDataBoundaryValidator
from .chain_identity_continuity import ChainIdentityContinuityValidator
from .chain_loop import ChainLoopValidator
from .chain_scope_monotonic import ChainScopeMonotonicValidator
from .chain_shutdown_propagation import ChainShutdownPropagationValidator
from .containment_budget import ContainmentBudgetValidator
from .containment_reach import ContainmentReachValidator
from .credential_isolation import CredentialIsolationValidator
from .data_exfiltration import DataExfiltrationValidator
from .extraction_query_pattern import ExtractionQueryPatternValidator
from .genai_instruction_hierarchy import GenaiInstructionHierarchyValidator
from .goal_drift import GoalDriftValidator
from .genai_jailbreak_resistance import GenaiJailbreakResistanceValidator
from .genai_sensitive_disclosure import GenaiSensitiveDisclosureValidator
from .identity_credential_ttl import IdentityCredentialTtlValidator
from .identity_delegation_chain import IdentityDelegationChainValidator
from .identity_zombie import IdentityZombieValidator
from .ks_active_execution_halt import KillSwitchActiveExecutionHaltValidator
from .ks_credential_revocation import KillSwitchCredentialRevocationValidator
from .ks_new_task_rejection import KillSwitchNewTaskRejectionValidator
from .ks_recovery_process import KillSwitchRecoveryProcessValidator
from .ks_shutdown_auditability import KillSwitchAuditabilityValidator
from .ks_tool_access_disabled import KillSwitchToolAccessDisabledValidator
from .human_override import HumanOverrideValidator
from .indirect_prompt_injection import IndirectPromptInjectionValidator
from .loop_detection import LoopDetectionValidator
from .mcp_cross_server_flow import McpCrossServerFlowValidator
from .mcp_server_allowlist import McpServerAllowlistValidator
from .mcp_tool_description_injection import McpToolDescriptionInjectionValidator
from .mcp_tool_shadowing import McpToolShadowingValidator
from .memory_integrity import MemoryIntegrityValidator
from .model_drift_psi import ModelDriftPsiValidator
from .model_fairness_parity import ModelFairnessParityValidator
from .model_performance_threshold import ModelPerformanceThresholdValidator
from .model_reproducibility import ModelReproducibilityValidator
from .policy_compliance import PolicyComplianceValidator
from .privilege_escalation import PrivilegeEscalationValidator
from .prompt_injection import PromptInjectionValidator
from .rag_citation_correctness import RagCitationCorrectnessValidator
from .rag_context_poisoning import RagContextPoisoningValidator
from .rag_document_provenance import RagDocumentProvenanceValidator
from .rag_grounding import RagGroundingValidator
from .rag_retrieval_completeness import RagRetrievalCompletenessValidator
from .record_completeness import RecordCompletenessValidator
from .side_effect_reversibility import SideEffectReversibilityValidator
from .supply_chain_bom_integrity import SupplyChainBomIntegrityValidator
from .tool_authorization import ToolAuthorizationValidator
from .training_data_provenance import TrainingDataProvenanceValidator

VALIDATOR_REGISTRY: dict[str, type[Validator]] = {
    ToolAuthorizationValidator.validator_id: ToolAuthorizationValidator,
    PromptInjectionValidator.validator_id: PromptInjectionValidator,
    DataExfiltrationValidator.validator_id: DataExfiltrationValidator,
    ActionIntegrityValidator.validator_id: ActionIntegrityValidator,
    PolicyComplianceValidator.validator_id: PolicyComplianceValidator,
    # The six emergency-control validators (detect -> initiate -> stop
    # -> revoke -> recover).
    KillSwitchNewTaskRejectionValidator.validator_id: KillSwitchNewTaskRejectionValidator,
    KillSwitchActiveExecutionHaltValidator.validator_id: KillSwitchActiveExecutionHaltValidator,
    KillSwitchToolAccessDisabledValidator.validator_id: KillSwitchToolAccessDisabledValidator,
    KillSwitchCredentialRevocationValidator.validator_id: KillSwitchCredentialRevocationValidator,
    KillSwitchAuditabilityValidator.validator_id: KillSwitchAuditabilityValidator,
    KillSwitchRecoveryProcessValidator.validator_id: KillSwitchRecoveryProcessValidator,
    # The five additional agent-behavioural validators.
    IndirectPromptInjectionValidator.validator_id: IndirectPromptInjectionValidator,
    PrivilegeEscalationValidator.validator_id: PrivilegeEscalationValidator,
    MemoryIntegrityValidator.validator_id: MemoryIntegrityValidator,
    CredentialIsolationValidator.validator_id: CredentialIsolationValidator,
    HumanOverrideValidator.validator_id: HumanOverrideValidator,
    # The four MCP security validators (AC-011).
    McpServerAllowlistValidator.validator_id: McpServerAllowlistValidator,
    McpToolDescriptionInjectionValidator.validator_id: McpToolDescriptionInjectionValidator,
    McpToolShadowingValidator.validator_id: McpToolShadowingValidator,
    McpCrossServerFlowValidator.validator_id: McpCrossServerFlowValidator,
    # The three agent-identity validators (AC-012).
    IdentityCredentialTtlValidator.validator_id: IdentityCredentialTtlValidator,
    IdentityDelegationChainValidator.validator_id: IdentityDelegationChainValidator,
    IdentityZombieValidator.validator_id: IdentityZombieValidator,
    # The five multi-agent chain validators (AC-014). Unlike every
    # validator above, these read `inputs["chain"]` (a roster plus one trace
    # per member) rather than `inputs["trace"]`, and produce one verdict for
    # the SYSTEM — see validators/chain_common.py for why the declared
    # roster, not the collected traces, is the iteration domain.
    ChainScopeMonotonicValidator.validator_id: ChainScopeMonotonicValidator,
    ChainDataBoundaryValidator.validator_id: ChainDataBoundaryValidator,
    ChainShutdownPropagationValidator.validator_id: ChainShutdownPropagationValidator,
    ChainIdentityContinuityValidator.validator_id: ChainIdentityContinuityValidator,
    ChainLoopValidator.validator_id: ChainLoopValidator,
    # The five RAG validators (AC-015). Unlike every validator above,
    # these read `inputs["rag"]` (one answered question against a
    # customer-supplied Q/A/citation set) rather than `inputs["trace"]`,
    # and produce one verdict per question — see validators/rag_common.py
    # for why a missing key reads as INCONCLUSIVE, never as clean.
    RagRetrievalCompletenessValidator.validator_id: RagRetrievalCompletenessValidator,
    RagCitationCorrectnessValidator.validator_id: RagCitationCorrectnessValidator,
    RagGroundingValidator.validator_id: RagGroundingValidator,
    RagContextPoisoningValidator.validator_id: RagContextPoisoningValidator,
    RagDocumentProvenanceValidator.validator_id: RagDocumentProvenanceValidator,
    # The three GenAI validators. These read `inputs["genai"]` (model
    # outputs, one per prompt) rather than `inputs["trace"]` — the unit is
    # a hosted application's response, not an agent's tool calls.
    GenaiSensitiveDisclosureValidator.validator_id: GenaiSensitiveDisclosureValidator,
    GenaiJailbreakResistanceValidator.validator_id: GenaiJailbreakResistanceValidator,
    GenaiInstructionHierarchyValidator.validator_id: GenaiInstructionHierarchyValidator,
    # The four MODEL validators (AC-016). These read `inputs["model"]`
    # (customer-computed metrics against a held-out set and validation
    # report) rather than `inputs["trace"]` — Coriqo re-derives a verdict
    # from numbers the customer already produced, never re-scores the
    # model itself.
    ModelPerformanceThresholdValidator.validator_id: ModelPerformanceThresholdValidator,
    ModelFairnessParityValidator.validator_id: ModelFairnessParityValidator,
    ModelDriftPsiValidator.validator_id: ModelDriftPsiValidator,
    ModelReproducibilityValidator.validator_id: ModelReproducibilityValidator,
    # The three security-assurance validators (AC-017). These read
    # `inputs["security"]` — a supply-chain bill of materials, per-
    # principal query stats, and dataset provenance flags — facts the
    # harness supplies, never a verdict it computed. None of the three
    # are the same-shaped sibling of an existing check, so AC-017 is new.
    SupplyChainBomIntegrityValidator.validator_id: SupplyChainBomIntegrityValidator,
    ExtractionQueryPatternValidator.validator_id: ExtractionQueryPatternValidator,
    TrainingDataProvenanceValidator.validator_id: TrainingDataProvenanceValidator,
    # The five autonomy-containment validators (AC-013). These read
    # `inputs["trace"]` like the shipped five, but against a deliberately
    # OVER-PROVISIONED sandbox: the environment held tools, data and
    # credentials the mandate does not allow, and the question is whether
    # the agent stayed inside the mandate anyway. What was available comes
    # from `configuration["environment_manifest"]`, hashed and signed into
    # the CEI envelope — see validators/containment_common.py for why an
    # environment that offered nothing outside the mandate makes a clean
    # run INCONCLUSIVE rather than a PASS.
    ContainmentReachValidator.validator_id: ContainmentReachValidator,
    ContainmentBudgetValidator.validator_id: ContainmentBudgetValidator,
    GoalDriftValidator.validator_id: GoalDriftValidator,
    LoopDetectionValidator.validator_id: LoopDetectionValidator,
    SideEffectReversibilityValidator.validator_id: SideEffectReversibilityValidator,
    # AC-007's record check. Reads `inputs["trace"]` like the shipped five,
    # and unlike every other validator here it asks nothing about what the
    # agent DID — only whether the events recording its consequential
    # actions carry enough (an id, a timestamp, a content hash) to
    # reconstruct them. Added to the default template, so it runs on an
    # ordinary agent assurance run rather than only under a campaign type.
    RecordCompletenessValidator.validator_id: RecordCompletenessValidator,
}


def resolve_validators(bundle: dict, *, only: list[str] | None = None) -> list[Validator]:
    """Instantiate one validator per id the compiled `bundle` declares
    (protocol_loader.HARDCODED_TEMPLATE's `protocol.validators` list),
    looked up in VALIDATOR_REGISTRY. `only` restricts to a subset of ids
    (e.g. re-running a single failed validator) without changing the
    bundle. Raises KeyError naming the id if the bundle declares a
    validator this package doesn't ship an implementation for — the same
    fail-fast posture RunnerEngine.run() already takes for the reverse
    mismatch (a validator not declared by the bundle)."""
    declared = [v["id"] for v in bundle["template"]["protocol"]["validators"]]
    if only is not None:
        unknown_only = [vid for vid in only if vid not in declared]
        if unknown_only:
            raise KeyError(
                f"only={unknown_only} named id(s) the bundle {bundle.get('id')!r} does not declare "
                f"(declares: {sorted(declared)}) — a caller asking to re-run a specific validator "
                "must not silently get zero validators back"
            )
    ids = [vid for vid in declared if only is None or vid in only]
    missing = [vid for vid in ids if vid not in VALIDATOR_REGISTRY]
    if missing:
        raise KeyError(
            f"bundle declares validator id(s) {missing} with no registered implementation "
            f"(registry has: {sorted(VALIDATOR_REGISTRY)})"
        )
    # The bundle is handed to every validator (Validator.__init__ takes it,
    # defaulted) so one that carries declarative test material — the
    # server-generated attack-scenario corpus — can read it without a second
    # wiring path. Validators that don't need it simply ignore it.
    return [VALIDATOR_REGISTRY[vid](bundle) for vid in ids]


__all__ = [
    "ActionIntegrityValidator",
    "ChainDataBoundaryValidator",
    "ChainIdentityContinuityValidator",
    "ChainLoopValidator",
    "ChainScopeMonotonicValidator",
    "ChainShutdownPropagationValidator",
    "ContainmentBudgetValidator",
    "ContainmentReachValidator",
    "CredentialIsolationValidator",
    "DataExfiltrationValidator",
    "KillSwitchActiveExecutionHaltValidator",
    "KillSwitchAuditabilityValidator",
    "KillSwitchCredentialRevocationValidator",
    "KillSwitchNewTaskRejectionValidator",
    "KillSwitchRecoveryProcessValidator",
    "KillSwitchToolAccessDisabledValidator",
    "GoalDriftValidator",
    "HumanOverrideValidator",
    "IdentityCredentialTtlValidator",
    "IdentityDelegationChainValidator",
    "IdentityZombieValidator",
    "IndirectPromptInjectionValidator",
    "LoopDetectionValidator",
    "McpCrossServerFlowValidator",
    "McpServerAllowlistValidator",
    "McpToolDescriptionInjectionValidator",
    "McpToolShadowingValidator",
    "MemoryIntegrityValidator",
    "PolicyComplianceValidator",
    "PrivilegeEscalationValidator",
    "PromptInjectionValidator",
    "RecordCompletenessValidator",
    "SideEffectReversibilityValidator",
    "ToolAuthorizationValidator",
    "VALIDATOR_REGISTRY",
    "resolve_validators",
]
