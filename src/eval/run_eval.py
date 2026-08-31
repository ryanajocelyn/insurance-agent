"""
Automated Evaluation Benchmark Runner CLI Script.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
import mlflow

from src.config import config
from src.core.graph import run_claim_adjudication
from src.utils.file_utils import read_json_file
from src.eval.claim_metrics import calculate_claim_evaluation_metrics
from src.eval.ragas_evaluator import RagasEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation benchmarks on test dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(config.SRC_DIR / "eval" / "eval_dataset.json"),
        help="Path to evaluation dataset JSON file.",
    )
    return parser.parse_args()


def run_evaluation_benchmark(dataset_path: str) -> Dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Evaluation dataset file not found at: {dataset_file}")

    test_samples = read_json_file(dataset_file)
    print(f"\n--> Starting Evaluation Benchmark on {len(test_samples)} claim test packets...")

    eval_results: List[Dict[str, Any]] = []
    questions: List[str] = []
    contexts: List[List[str]] = []
    answers: List[str] = []
    ground_truths: List[str] = []

    for item in test_samples:
        claim_id = item.get("claim_id", "EVAL-000")
        print(f"    Processing {claim_id}...")

        final_state = run_claim_adjudication(initial_state=item)

        pred_verdict = final_state.get("adjudication_verdict", "ESCALATE")
        pred_amount = float(final_state.get("approved_amount", 0.0))
        gt_verdict = item.get("ground_truth_verdict", "ESCALATE")
        gt_amount = float(item.get("ground_truth_approved_amount", 0.0))

        eval_results.append(
            {
                "claim_id": claim_id,
                "predicted_verdict": pred_verdict,
                "ground_truth_verdict": gt_verdict,
                "predicted_amount": pred_amount,
                "ground_truth_amount": gt_amount,
            }
        )

        questions.append(item.get("incident_narrative", ""))
        retrieved_clauses = final_state.get("retrieved_policy_clauses", [])
        contexts.append([c.get("clause_text", "") for c in retrieved_clauses] if retrieved_clauses else ["None"])
        answers.append(final_state.get("adjudication_rationale", ""))
        ground_truths.append(gt_verdict)

    claim_metrics = calculate_claim_evaluation_metrics(eval_results)

    ragas = RagasEvaluator()
    ragas_metrics = ragas.evaluate_retrieval_and_generation(
        questions=questions,
        contexts=contexts,
        answers=answers,
        ground_truths=ground_truths,
    )

    combined_metrics = {**claim_metrics, **ragas_metrics}

    print("\n========================================================")
    print("   EVALUATION BENCHMARK RESULTS")
    print(f"   Verdict Classification Accuracy : {combined_metrics['verdict_accuracy'] * 100:.1f}%")
    print(f"   Payout Settlement MAE            : ₹{combined_metrics['payout_mae']:,.2f}")
    print(f"   RAGAS Context Precision          : {combined_metrics['context_precision']:.2f}")
    print(f"   RAGAS Context Recall             : {combined_metrics['context_recall']:.2f}")
    print(f"   RAGAS Faithfulness (Zero-Halluc) : {combined_metrics['faithfulness']:.2f}")
    print("========================================================\n")

    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="Evaluation_Benchmark_Suite"):
            for k, v in combined_metrics.items():
                mlflow.log_metric(k, float(v))
    except Exception as exc:
        print(f"[MLFLOW WARNING] Failed to log evaluation to MLflow: {exc}")

    return combined_metrics


if __name__ == "__main__":
    args = parse_args()
    run_evaluation_benchmark(args.dataset)
