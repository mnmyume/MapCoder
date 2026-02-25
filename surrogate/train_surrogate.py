"""
Train a Random Forest surrogate model on MapCoderMAS benchmark data.

Features : 4 categorical model assignments (ordinal-encoded)
Targets  : accuracy (%), cost ($)

This script is STRICTLY for initial model training.  It never loads or
evaluates on the held-out test set (test_configs.json).  Use
test_surrogate.py for evaluation against the test set.

Usage (with real benchmark results):
    python surrogate/train_surrogate.py

Usage (synthetic dry-run, no real benchmarks needed):
    python surrogate/train_surrogate.py --synthetic

Output:
    - Saves the trained model to surrogate/data/model_initial.pkl
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OrdinalEncoder

from surrogate.config import MODEL_POOL, AGENT_ROLES


# ─── Data loading ────────────────────────────────────────────────────────────

def load_real_data(config_path: str, results_dir: str):
    """
    Load configs + their benchmark results.

    Returns
    -------
    X_raw : list[list[str]]   – model names per role
    y     : np.ndarray shape (n, 2) – [accuracy, cost]
    """
    with open(config_path, "r") as f:
        configs = json.load(f)

    X_raw, y = [], []
    missing = []

    for cfg in configs:
        idx = cfg["config_index"]
        summary_path = os.path.join(results_dir, f"summary_{idx}.json")

        if not os.path.exists(summary_path):
            missing.append(idx)
            continue

        with open(summary_path, "r") as f:
            result = json.load(f)

        X_raw.append([cfg[role] for role in AGENT_ROLES])
        y.append([result["accuracy"], result["cost"]])

    if missing:
        print(f"  ⚠  Missing results for config indices: {missing}")

    return X_raw, np.array(y)


def generate_synthetic_data(configs_dicts: list):
    """
    Generate fake accuracy/cost values for dry-run testing.
    A deterministic toy function so results are reproducible.
    """
    # Assign each model a fictional score
    model_acc = {m: 0.5 + 0.1 * i for i, m in enumerate(MODEL_POOL)}
    model_cost = {m: 1.0 + 2.0 * i for i, m in enumerate(MODEL_POOL)}
    role_weights = [0.4, 0.3, 0.2, 0.1]

    rng = np.random.RandomState(123)

    X_raw, y = [], []
    for cfg in configs_dicts:
        models = [cfg[role] for role in AGENT_ROLES]
        acc = sum(model_acc[m] * w for m, w in zip(models, role_weights))
        cost = sum(model_cost[m] for m in models)
        # Add small noise
        acc = min(100.0, max(0.0, acc * 100 + rng.normal(0, 2)))
        cost = max(0.0, cost + rng.normal(0, 0.5))
        X_raw.append(models)
        y.append([acc, cost])

    return X_raw, np.array(y)


# ─── Training ───────────────────────────────────────────────────────────────

def fit_model(X_raw, y, model_pool=None, n_estimators=100, random_state=42):
    """
    Fit a MultiOutput Random Forest on training data only.

    Returns
    -------
    model   : MultiOutputRegressor  – fitted model
    encoder : OrdinalEncoder        – fitted encoder
    """
    if model_pool is None:
        model_pool = MODEL_POOL

    # Encode categorical features
    encoder = OrdinalEncoder(categories=[model_pool] * len(AGENT_ROLES))
    X = encoder.fit_transform(X_raw)

    # Train
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model = MultiOutputRegressor(rf)
    model.fit(X, y)

    return model, encoder


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train the initial RF surrogate model (training data only)"
    )
    parser.add_argument("--data_dir", type=str, default="surrogate/data",
                        help="Directory containing config JSONs and results/")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Override results directory (default: <data_dir>/results)")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for dry-run testing")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save model pickle (default: <data_dir>/model_initial.pkl)")
    args = parser.parse_args()

    data_dir = args.data_dir
    results_dir = args.results_dir or os.path.join(data_dir, "results")
    output_path = args.output or os.path.join(data_dir, "model_initial.pkl")

    # ── Only load training configs (NEVER test_configs.json) ──────────────
    train_cfg_path = os.path.join(data_dir, "train_configs.json")

    with open(train_cfg_path, "r") as f:
        train_cfgs = json.load(f)

    print(f"Train configs: {len(train_cfgs)}")
    print(f"Model pool   : {MODEL_POOL}\n")

    if args.synthetic:
        print("▶ Using SYNTHETIC data (dry-run mode)\n")
        X_train_raw, y_train = generate_synthetic_data(train_cfgs)
    else:
        print("▶ Loading REAL benchmark results\n")
        X_train_raw, y_train = load_real_data(train_cfg_path, results_dir)

        if len(y_train) == 0:
            print("ERROR: No training data found. Run benchmarks first, or "
                  "use --synthetic for a dry-run.")
            sys.exit(1)

    print(f"Training samples: {len(y_train)}")

    # Train
    model, encoder = fit_model(
        X_train_raw, y_train,
        n_estimators=args.n_estimators,
    )

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"model": model, "encoder": encoder}, f)

    print(f"\n{'='*60}")
    print(f"  INITIAL SURROGATE MODEL TRAINED")
    print(f"{'='*60}")
    print(f"  Training samples : {len(y_train)}")
    print(f"  Features         : {len(AGENT_ROLES)} roles × {len(MODEL_POOL)} models")
    print(f"  Model saved      : {output_path}")
    print(f"{'='*60}")
    print(f"\n  → Run test_surrogate.py to evaluate on the held-out test set.")


if __name__ == "__main__":
    main()
