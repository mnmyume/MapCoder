"""
Slurm dispatch helper — runs a single benchmark configuration.

This script is invoked by the Slurm job array. It reads the merged config
file, picks the entry matching SLURM_ARRAY_TASK_ID (or --task_id), and
launches ``src/main_mas.py`` as a subprocess.

Usage (inside Slurm):
    python surrogate/run_benchmark.py \
        --config_file surrogate/data/all_configs.json \
        --task_id $SLURM_ARRAY_TASK_ID

Manual testing:
    python surrogate/run_benchmark.py \
        --config_file surrogate/data/all_configs.json \
        --task_id 0 \
        --dry_run
"""

import argparse
import json
import os
import subprocess
import sys

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from surrogate.config import (
    DEFAULT_DATASET,
    DEFAULT_TEMPERATURE,
    DEFAULT_PASS_AT_K,
    DEFAULT_LANGUAGE,
)


def load_config(config_file: str, task_id: int) -> dict:
    with open(config_file, "r") as f:
        configs = json.load(f)

    for cfg in configs:
        if cfg["config_index"] == task_id:
            return cfg

    raise ValueError(
        f"No config with config_index={task_id} found in {config_file}"
    )


def build_command(cfg: dict, output_dir: str, dataset: str,
                  temperature: float, pass_at_k: int, language: str) -> list:
    """Build the subprocess command list for main_mas.py."""
    return [
        sys.executable, "src/main_mas.py",
        "--retrieval_model", cfg["retrieval"],
        "--planning_model", cfg["planning"],
        "--coding_model", cfg["coding"],
        "--debugging_model", cfg["debugging"],
        "--dataset", dataset,
        "--temperature", str(temperature),
        "--pass_at_k", str(pass_at_k),
        "--language", language,
        "--config_index", str(cfg["config_index"]),
        "--output_dir", output_dir,
    ]


def main():
    parser = argparse.ArgumentParser(description="Run a single benchmark config")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to all_configs.json")
    parser.add_argument("--task_id", type=int, required=True,
                        help="Config index (typically SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--output_dir", type=str, default="surrogate/data/results",
                        help="Directory for result JSONs")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--pass_at_k", type=int, default=DEFAULT_PASS_AT_K)
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the command without executing")
    args = parser.parse_args()

    cfg = load_config(args.config_file, args.task_id)
    cmd = build_command(cfg, args.output_dir, args.dataset,
                        args.temperature, args.pass_at_k, args.language)

    print(f"[run_benchmark] Config index: {args.task_id}")
    print(f"[run_benchmark] Config: {cfg}")
    print(f"[run_benchmark] Command: {' '.join(cmd)}")

    if args.dry_run:
        print("[run_benchmark] DRY RUN — skipping execution")
        return

    result = subprocess.run(cmd, cwd=os.getcwd())
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
