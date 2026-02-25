"""
Evaluate a saved surrogate model against the held-out test set.

This is the ONLY script in the pipeline that loads test_configs.json.
Neither train_surrogate.py nor optimize.py ever touch the test data.

Usage:
    python surrogate/test_surrogate.py --model_path surrogate/data/model_initial.pkl
    python surrogate/test_surrogate.py --model_path surrogate/data/model_optimized.pkl
    python surrogate/test_surrogate.py --model_path surrogate/data/model_iter_3.pkl --synthetic
"""

import argparse
import json
import os
import pickle
import sys

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from surrogate.config import MODEL_POOL, AGENT_ROLES
from surrogate.train_surrogate import load_real_data, generate_synthetic_data


# ─── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(model, encoder, X_test_raw, y_test):
    """
    Evaluate the model on the test set.

    Returns
    -------
    metrics : dict  – per-objective MSE, MAE, R²
    y_pred  : np.ndarray – predictions shape (n, 2)
    """
    X_test = encoder.transform(X_test_raw)
    y_pred = model.predict(X_test)

    obj_names = ["accuracy", "cost"]
    metrics = {}
    for i, name in enumerate(obj_names):
        metrics[name] = {
            "mse": mean_squared_error(y_test[:, i], y_pred[:, i]),
            "mae": mean_absolute_error(y_test[:, i], y_pred[:, i]),
            "r2": r2_score(y_test[:, i], y_pred[:, i]),
        }

    # Per-tree uncertainty (std across estimators) on test set
    for i, (name, estimator) in enumerate(
        zip(obj_names, model.estimators_)
    ):
        tree_preds = np.array(
            [tree.predict(X_test) for tree in estimator.estimators_]
        )
        mean_std = np.mean(np.std(tree_preds, axis=0))
        metrics[name]["mean_tree_std"] = mean_std

    return metrics, y_pred


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved surrogate model on the held-out test set"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the saved model pickle "
             "(e.g. surrogate/data/model_initial.pkl or model_optimized.pkl)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="surrogate/data",
        help="Directory containing test_configs.json and results/"
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Override results directory (default: <data_dir>/results)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic test data instead of real benchmark results"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    results_dir = args.results_dir or os.path.join(data_dir, "results")

    # ── Load the model ───────────────────────────────────────────────────
    print(f"Loading model from: {args.model_path}")
    with open(args.model_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    encoder = data["encoder"]
    print(f"  Model type : {type(model).__name__}")
    print(f"  Encoder    : {type(encoder).__name__}")

    # ── Load the TEST set (strictly isolated) ────────────────────────────
    test_cfg_path = os.path.join(data_dir, "test_configs.json")

    with open(test_cfg_path, "r") as f:
        test_cfgs = json.load(f)

    print(f"  Test configs : {len(test_cfgs)}")
    print(f"  Model pool   : {MODEL_POOL}\n")

    if args.synthetic:
        print("▶ Using SYNTHETIC test data (dry-run mode)\n")
        X_test_raw, y_test = generate_synthetic_data(test_cfgs)
    else:
        print("▶ Loading REAL benchmark results for test set\n")
        X_test_raw, y_test = load_real_data(test_cfg_path, results_dir)

        if len(y_test) == 0:
            print("ERROR: No test result data found. Run benchmarks first, "
                  "or use --synthetic for a dry-run.")
            sys.exit(1)

    print(f"Test samples: {len(y_test)}\n")

    # ── Evaluate ─────────────────────────────────────────────────────────
    metrics, y_pred = evaluate_model(model, encoder, X_test_raw, y_test)

    # ── Report ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("  SURROGATE MODEL EVALUATION (Held-Out Test Set)")
    print("=" * 60)
    print(f"  Model : {args.model_path}")
    print(f"  Samples : {len(y_test)}")
    for obj_name, m in metrics.items():
        print(f"\n  {obj_name.upper()}")
        print(f"    MSE             : {m['mse']:.4f}")
        print(f"    MAE             : {m['mae']:.4f}")
        print(f"    R²              : {m['r2']:.4f}")
        print(f"    Mean tree σ     : {m['mean_tree_std']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
