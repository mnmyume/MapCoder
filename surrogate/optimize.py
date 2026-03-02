"""
Active-learning optimisation loop for MapCoderMAS multi-objective search.

Each iteration:
  1. Use the RF surrogate to select top-N untested configs (Pareto-aware).
  2. Write them to a per-iteration config JSON.
  3. Submit a Slurm job array to evaluate them in parallel.
  4. Poll squeue until all jobs complete (sync barrier).
  5. Parse results, append to training data, retrain the surrogate.
  6. Compute hypervolume, check early stopping.
  7. Save model checkpoint (per-iteration + final optimized model).

DATA ISOLATION: This script NEVER loads, references, or trains on
test_configs.json.  The test set is strictly reserved for
test_surrogate.py.

Usage (real, on cluster):
    sbatch slurm/run_optimize.sh \
    --max_iterations 10 \
    --n_per_iter 5 \
    --tolerance 0.01 \
    --patience 3

Usage (synthetic dry-run, no Slurm):
    python surrogate/optimize.py --dry_run \
        --max_iterations 3 --n_per_iter 3
"""

import argparse
import datetime
import json
import os
import pickle
import re
import subprocess
import sys
import time

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OrdinalEncoder

from surrogate.config import MODEL_POOL, AGENT_ROLES
from surrogate.hypervolume import hypervolume_2d, EarlyStoppingTracker
from surrogate.selector import select_top_n
from surrogate.train_surrogate import (
    load_real_data,
    generate_synthetic_data,
)


# ─── Globals / Paths ────────────────────────────────────────────────────────

DATA_DIR = os.path.join("surrogate", "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
LOG_PATH = os.path.join(DATA_DIR, "optimization_log.json")
RUN_LOG_PATH = os.path.join(DATA_DIR, "optimize_run.log")
SLURM_TEMPLATE = os.path.join("slurm", "run_iteration.sh")


class _TeeLogger:
    """Duplicate stdout to both the terminal and a log file."""

    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._terminal = sys.stdout
        self._log = open(log_path, "a")
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log.write(f"\n{'='*60}\n  Run started: {ts}\n{'='*60}\n")
        self._log.flush()

    def write(self, msg):
        self._terminal.write(msg)
        self._log.write(msg)
        self._log.flush()

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()


# ─── Slurm helpers ──────────────────────────────────────────────────────────

def submit_slurm_array(
    config_indices: list,
    config_file: str,
    concurrency_limit: int = 6,
) -> str:
    """
    Generate a concrete Slurm script from the template and submit it.

    Returns the Slurm job ID string.
    """
    # Build array spec, e.g. "0,3,7,12,55%6"
    idx_str = ",".join(str(i) for i in config_indices)
    array_spec = f"{idx_str}%{concurrency_limit}"

    # Read template
    with open(SLURM_TEMPLATE, "r") as f:
        script = f.read()

    # Substitute placeholders
    script = script.replace("__ARRAY_SPEC__", array_spec)
    script = script.replace("__CONFIG_FILE__", config_file)

    # Write concrete script
    concrete_path = os.path.join("slurm", "run_iteration_current.sh")
    with open(concrete_path, "w") as f:
        f.write(script)

    # Submit
    result = subprocess.run(
        ["sbatch", concrete_path],
        capture_output=True, text=True, cwd=_PROJECT_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr}")

    # Parse job ID from "Submitted batch job 12345"
    match = re.search(r"(\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID from: {result.stdout}")

    job_id = match.group(1)
    print(f"[optimize] Submitted Slurm job array {job_id} "
          f"(tasks: {idx_str})", flush=True)
    return job_id


def wait_for_slurm_jobs(
    job_id: str,
    poll_interval: int = 60,
    timeout_hours: float = 24.0,
) -> bool:
    """
    Poll squeue until all tasks in the job array have completed.

    Returns True if all tasks finished, False on timeout.
    """
    max_polls = int(timeout_hours * 3600 / poll_interval)
    print(f"[optimize] Waiting for job {job_id} to complete "
          f"(poll every {poll_interval}s, timeout {timeout_hours}h)...",
          flush=True)

    for poll in range(max_polls):
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h"],
            capture_output=True, text=True,
        )
        remaining = result.stdout.strip()
        if not remaining:
            print(f"[optimize] All tasks in job {job_id} completed.", flush=True)
            return True

        n_remaining = len(remaining.strip().split("\n"))
        if poll % 5 == 0:  # Print status every 5 polls
            print(f"[optimize]   ... {n_remaining} tasks still running "
                  f"(poll {poll + 1})", flush=True)

        time.sleep(poll_interval)

    print(f"[optimize] TIMEOUT waiting for job {job_id}.", flush=True)
    return False


# ─── Result parsing ─────────────────────────────────────────────────────────

def parse_iteration_results(
    config_indices: list,
    results_dir: str = RESULTS_DIR,
) -> list:
    """
    Load summary JSON files for the given config indices.
    Returns a list of result dicts (may be shorter if some failed).
    """
    results = []
    for idx in config_indices:
        summary_path = os.path.join(results_dir, f"summary_{idx}.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                results.append(json.load(f))
        else:
            print(f"[optimize] WARNING: Missing result for config {idx}",
                  flush=True)
    return results


# ─── Surrogate model management ─────────────────────────────────────────────

def load_surrogate(model_path: str):
    """Load a pickled surrogate model + encoder."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["encoder"]


def retrain_surrogate(
    all_X_raw: list,
    all_y: np.ndarray,
    n_estimators: int = 100,
):
    """
    Retrain the surrogate on all accumulated training data.

    Uses an 80/20 internal split for metrics reporting, then trains the
    final model on ALL accumulated data.

    NOTE: This never touches the held-out test set (test_configs.json).
    """
    n = len(all_X_raw)
    encoder = OrdinalEncoder(categories=[MODEL_POOL] * len(AGENT_ROLES))

    if n < 5:
        print(f"[optimize] Only {n} data points — training on all, no internal split.")
        X = encoder.fit_transform(all_X_raw)
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model = MultiOutputRegressor(rf)
        model.fit(X, all_y)
        return model, encoder, {}

    # 80/20 internal split for metrics reporting (NOT the held-out test set)
    split = int(n * 0.8)
    indices = np.random.RandomState(42).permutation(n)
    train_idx, val_idx = indices[:split], indices[split:]

    X_train_raw = [all_X_raw[i] for i in train_idx]
    X_val_raw = [all_X_raw[i] for i in val_idx]
    y_train = all_y[train_idx]
    y_val = all_y[val_idx]

    # Fit on the train split to get metrics
    X_train = encoder.fit_transform(X_train_raw)
    X_val = encoder.transform(X_val_raw)

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    split_model = MultiOutputRegressor(rf)
    split_model.fit(X_train, y_train)

    y_pred = split_model.predict(X_val)

    from sklearn.metrics import r2_score, mean_absolute_error
    obj_names = ["accuracy", "cost"]
    metrics = {}
    for i, name in enumerate(obj_names):
        metrics[name] = {
            "r2": r2_score(y_val[:, i], y_pred[:, i]),
            "mae": mean_absolute_error(y_val[:, i], y_pred[:, i]),
        }

    # Now retrain on ALL data for the actual model to save
    X_all = encoder.fit_transform(all_X_raw)
    final_rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    final_model = MultiOutputRegressor(final_rf)
    final_model.fit(X_all, all_y)

    return final_model, encoder, metrics


def save_surrogate(model, encoder, metrics, model_path: str):
    """Save the surrogate model + encoder to disk."""
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "encoder": encoder, "metrics": metrics}, f)
    print(f"[optimize] Surrogate saved → {model_path}", flush=True)


# ─── Synthetic evaluator (for dry-run) ──────────────────────────────────────

def synthetic_evaluate(configs: list) -> list:
    """
    Simulate benchmark results without Slurm.
    Uses the same model as train_surrogate's synthetic mode.
    """
    model_acc = {m: 0.5 + 0.1 * i for i, m in enumerate(MODEL_POOL)}
    model_cost = {m: 1.0 + 2.0 * i for i, m in enumerate(MODEL_POOL)}
    role_weights = [0.4, 0.3, 0.2, 0.1]
    rng = np.random.RandomState(None)  # non-deterministic noise

    results = []
    for cfg in configs:
        models = [cfg[role] for role in AGENT_ROLES]
        acc = sum(model_acc[m] * w for m, w in zip(models, role_weights))
        cost = sum(model_cost[m] for m in models)
        acc = min(100.0, max(0.0, acc * 100 + rng.normal(0, 2)))
        cost = max(0.0, cost + rng.normal(0, 0.5))
        results.append({
            "config_index": cfg.get("config_index", -1),
            **cfg,
            "accuracy": acc,
            "cost": cost,
        })
    return results


# ─── Main loop ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Active-learning optimisation loop for MapCoderMAS"
    )
    parser.add_argument("--max_iterations", type=int, default=10,
                        help="Maximum number of optimisation iterations")
    parser.add_argument("--n_per_iter", type=int, default=5,
                        help="Number of configs to evaluate per iteration")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="HV improvement tolerance for early stopping")
    parser.add_argument("--patience", type=int, default=3,
                        help="Patience for early stopping (consecutive stalls)")
    parser.add_argument("--kappa", type=float, default=1.96,
                        help="Exploration parameter for UCB acquisition")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of RF estimators")
    parser.add_argument("--poll_interval", type=int, default=60,
                        help="Slurm poll interval in seconds")
    parser.add_argument("--slurm_timeout", type=float, default=24.0,
                        help="Slurm timeout in hours")
    parser.add_argument("--concurrency", type=int, default=6,
                        help="Max concurrent Slurm tasks")
    parser.add_argument("--model_path", type=str,
                        default=os.path.join(DATA_DIR, "model_initial.pkl"),
                        help="Path to the initial surrogate model pickle")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for model checkpoints (default: <data_dir>)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Use synthetic evaluation (no Slurm)")
    args = parser.parse_args()

    output_dir = args.output_dir or DATA_DIR

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Tee all print output to a persistent log file
    sys.stdout = _TeeLogger(RUN_LOG_PATH)

    # ── Load initial surrogate ───────────────────────────────────────────
    if os.path.exists(args.model_path):
        print(f"[optimize] Loading existing surrogate from {args.model_path}")
        model, encoder = load_surrogate(args.model_path)
    else:
        print("[optimize] No existing surrogate found — training from scratch "
              "on available data...")
        # Try loading whatever training results exist
        train_cfg_path = os.path.join(DATA_DIR, "train_configs.json")
        if os.path.exists(train_cfg_path):
            X_raw, y = load_real_data(train_cfg_path, RESULTS_DIR)
            if len(X_raw) < 5:
                if args.dry_run:
                    print("[optimize] Bootstrapping with synthetic data for dry-run.")
                    with open(train_cfg_path) as f:
                        cfgs = json.load(f)
                    X_raw, y = generate_synthetic_data(cfgs[:20])
                else:
                    print("[optimize] ERROR: Need at least 5 evaluated configs to start. "
                          "Run the initial benchmark sweep first.")
                    sys.exit(1)
            model, encoder, _ = retrain_surrogate(X_raw, y, args.n_estimators)
        else:
            print("[optimize] ERROR: No config files found. Run config_generator.py first.")
            sys.exit(1)

    # ── Collect all evaluated configs so far ──────────────────────────────
    all_X_raw = []  # list of [model_name, ...] per role
    all_y = []      # list of [accuracy, cost]
    evaluated_set = set()  # set of tuples for dedup

    # Load from existing results
    for fname in os.listdir(RESULTS_DIR):
        if fname.startswith("summary_") and fname.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                r = json.load(f)
            cfg_tuple = tuple(r[f"{role}_model"] for role in AGENT_ROLES
                              if f"{role}_model" in r)
            if len(cfg_tuple) == len(AGENT_ROLES) and cfg_tuple not in evaluated_set:
                evaluated_set.add(cfg_tuple)
                all_X_raw.append(list(cfg_tuple))
                all_y.append([r["accuracy"], r["cost"]])

    all_y = np.array(all_y) if all_y else np.empty((0, 2))
    print(f"[optimize] Starting with {len(all_X_raw)} evaluated configs.\n")

    # ── Optimisation log ─────────────────────────────────────────────────
    opt_log = {
        "iterations": [],
        "settings": vars(args),
    }

    # Load existing log if resuming
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            opt_log = json.load(f)
        print(f"[optimize] Resuming from existing log "
              f"({len(opt_log['iterations'])} previous iterations).\n")

    early_stopper = EarlyStoppingTracker(
        tolerance=args.tolerance, patience=args.patience,
    )
    # Replay HV history if resuming
    for prev_iter in opt_log.get("iterations", []):
        early_stopper.update(prev_iter.get("hypervolume", 0.0))

    # ── Determine next config index ──────────────────────────────────────
    existing_indices = set()
    for fname in os.listdir(RESULTS_DIR):
        m = re.match(r"summary_(\d+)\.json", fname)
        if m:
            existing_indices.add(int(m.group(1)))
    next_config_index = max(existing_indices) + 1 if existing_indices else 0

    # ── Main loop ────────────────────────────────────────────────────────
    start_iter = len(opt_log["iterations"])

    for iteration in range(start_iter, start_iter + args.max_iterations):
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}")
        print(f"{'='*60}\n")

        # 1. Select top-N untested configs
        print("[optimize] Step 1: Selecting candidates...", flush=True)
        candidates = select_top_n(
            model, encoder, evaluated_set,
            n=args.n_per_iter,
            kappa=args.kappa,
            n_scalarizations=50,
            rng_seed=iteration * 1000,
        )

        if not candidates:
            print("[optimize] No more untested configs — stopping.")
            break

        # Assign config indices
        for i, cfg in enumerate(candidates):
            cfg["config_index"] = next_config_index + i
        config_indices = [c["config_index"] for c in candidates]
        next_config_index += len(candidates)

        # 2. Write per-iteration config file
        iter_config_path = os.path.join(
            DATA_DIR, f"iter_{iteration}_configs.json"
        )
        with open(iter_config_path, "w") as f:
            json.dump(candidates, f, indent=2)
        print(f"[optimize] Wrote {len(candidates)} configs → {iter_config_path}",
              flush=True)

        # 3. Evaluate
        if args.dry_run:
            print("[optimize] Step 3: Synthetic evaluation (dry-run)...", flush=True)
            iter_results = synthetic_evaluate(candidates)
        else:
            print("[optimize] Step 3: Submitting Slurm job array...", flush=True)
            job_id = submit_slurm_array(
                config_indices, iter_config_path,
                concurrency_limit=args.concurrency,
            )

            # 4. Wait for completion
            print("[optimize] Step 4: Waiting for jobs...", flush=True)
            completed = wait_for_slurm_jobs(
                job_id,
                poll_interval=args.poll_interval,
                timeout_hours=args.slurm_timeout,
            )
            if not completed:
                print("[optimize] WARNING: Jobs timed out. Proceeding with "
                      "available results.", flush=True)

            iter_results = parse_iteration_results(config_indices)

        # 5. Append results to accumulated data
        n_new = 0
        for r in iter_results:
            cfg_tuple = tuple(r.get(f"{role}_model", "") for role in AGENT_ROLES)
            if cfg_tuple not in evaluated_set:
                evaluated_set.add(cfg_tuple)
                all_X_raw.append(list(cfg_tuple))
                all_y = np.vstack([all_y, [[r["accuracy"], r["cost"]]]])
                n_new += 1

        print(f"[optimize] Step 5: Added {n_new} new results "
              f"(total: {len(all_X_raw)})", flush=True)

        # 6. Retrain surrogate
        print("[optimize] Step 6: Retraining surrogate...", flush=True)
        model, encoder, metrics = retrain_surrogate(
            all_X_raw, all_y, n_estimators=args.n_estimators,
        )

        # Save per-iteration checkpoint
        iter_model_path = os.path.join(output_dir, f"model_iter_{iteration}.pkl")
        save_surrogate(model, encoder, metrics, iter_model_path)

        if metrics:
            for obj_name, m in metrics.items():
                print(f"  {obj_name}: R²={m['r2']:.4f}  MAE={m['mae']:.4f}")

        # 7. Compute hypervolume
        hv = hypervolume_2d(all_y[:, 0], all_y[:, 1])
        print(f"[optimize] Step 7: Hypervolume = {hv:.4f}", flush=True)

        # 8. Log iteration
        iter_log = {
            "iteration": iteration,
            "n_candidates": len(candidates),
            "n_new_results": n_new,
            "total_evaluated": len(all_X_raw),
            "hypervolume": hv,
            "surrogate_metrics": metrics,
            "candidates": candidates,
            "model_checkpoint": iter_model_path,
        }
        opt_log["iterations"].append(iter_log)

        with open(LOG_PATH, "w") as f:
            json.dump(opt_log, f, indent=2)

        # 9. Check early stopping
        if early_stopper.update(hv):
            print(f"\n[optimize] EARLY STOPPING at iteration {iteration}.")
            break

        print(f"\n[optimize] Iteration {iteration} complete. "
              f"HV={hv:.4f}, total configs={len(all_X_raw)}")

    # ── Save final optimized model ───────────────────────────────────────
    final_model_path = os.path.join(output_dir, "model_optimized.pkl")
    save_surrogate(model, encoder, metrics, final_model_path)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  OPTIMISATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total iterations      : {len(opt_log['iterations'])}")
    print(f"  Total evaluated       : {len(all_X_raw)}")
    print(f"  Best hypervolume      : {early_stopper.best_hv:.4f}")
    print(f"  Log saved             : {LOG_PATH}")
    print(f"  Final model saved     : {final_model_path}")
    print(f"  Per-iter checkpoints  : {output_dir}/model_iter_*.pkl")
    print(f"{'='*60}")
    print(f"\n  → Run test_surrogate.py --model_path {final_model_path} "
          f"to evaluate on the held-out test set.\n")


if __name__ == "__main__":
    main()
