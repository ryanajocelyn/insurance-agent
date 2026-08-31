-- SQL Template: Fetch Customer Claim History within Lookback Horizon
SELECT 
    claim_id,
    customer_id,
    policy_number,
    claim_date,
    claimed_amount,
    approved_amount,
    claim_type,
    adjudication_verdict
FROM claims_history
WHERE customer_id = {{ customer_id }}
  AND claim_date >= datetime('now', '-' || {{ lookback_months }} || ' month')
ORDER BY claim_date DESC;
