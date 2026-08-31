-- SQL Template: Persist Synthesized Adjudication Decision Record
INSERT INTO adjudication_audit_logs (
    claim_id,
    policy_number,
    created_at,
    claimed_amount,
    approved_amount,
    adjudication_verdict,
    cross_modal_consistency,
    rationale,
    deductions_json,
    citations_json,
    investigation_triggers_json
) VALUES (
    {{ claim_id }},
    {{ policy_number }},
    datetime('now'),
    {{ claimed_amount }},
    {{ approved_amount }},
    {{ adjudication_verdict }},
    {{ cross_modal_consistency }},
    {{ rationale }},
    {{ deductions_json }},
    {{ citations_json }},
    {{ investigation_triggers_json }}
);
