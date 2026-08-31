"""
Custom Claim Adjudication Evaluation Metrics Module.
"""

from typing import Dict, Any, List


def calculate_claim_evaluation_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {
            "verdict_accuracy": 0.0,
            "payout_mae": 0.0,
            "total_eval_samples": 0,
        }

    correct_verdicts = 0
    total_mae_error = 0.0
    total_count = len(results)

    for item in results:
        pred_verdict = item.get("predicted_verdict", "").upper()
        gt_verdict = item.get("ground_truth_verdict", "").upper()

        if pred_verdict == gt_verdict:
            correct_verdicts += 1

        pred_amt = float(item.get("predicted_amount", 0.0))
        gt_amt = float(item.get("ground_truth_amount", 0.0))
        total_mae_error += abs(pred_amt - gt_amt)

    accuracy = correct_verdicts / total_count
    mae = total_mae_error / total_count

    return {
        "verdict_accuracy": round(accuracy, 4),
        "payout_mae": round(mae, 2),
        "total_eval_samples": total_count,
    }
