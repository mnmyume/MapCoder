"""
Generate random, unique model configurations for the surrogate data pipeline.

Usage:
    python surrogate/config_generator.py                 # uses defaults
    python surrogate/config_generator.py --n_train 60 --n_test 60 --seed 42

Outputs:
    surrogate/data/train_configs.json   (60 configs, indices 0-59)
    surrogate/data/test_configs.json    (60 configs, indices 60-119)
    surrogate/data/all_configs.json     (merged 120 configs, for Slurm array)
"""

import argparse
import itertools
import json
import os
import random
import sys
from typing import List, Tuple

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from surrogate.config import MODEL_POOL, AGENT_ROLES, NONE_INELIGIBLE_ROLES


# ─── Core sampling logic ────────────────────────────────────────────────────

def _is_valid_config(cfg: Tuple[str, ...]) -> bool:
    """Check that no NONE_INELIGIBLE_ROLES role is assigned 'None'."""
    for role, model in zip(AGENT_ROLES, cfg):
        if model == "None" and role in NONE_INELIGIBLE_ROLES:
            return False
    return True


def generate_all_possible_configs(
    model_pool: List[str],
    num_roles: int,
) -> List[Tuple[str, ...]]:
    """Return every valid configuration (Cartesian product minus constrained ones)."""
    all_cfgs = list(itertools.product(model_pool, repeat=num_roles))
    return [c for c in all_cfgs if _is_valid_config(c)]


def sample_unique_configs(
    n: int,
    model_pool: List[str],
    num_roles: int,
    exclude: set | None = None,
    rng: random.Random | None = None,
) -> List[Tuple[str, ...]]:
    """
    Sample *n* unique configurations from the model pool.

    Parameters
    ----------
    n : int
        Number of configs to sample.
    model_pool : list[str]
        Available model names.
    num_roles : int
        Number of agent slots (4 for MapCoderMAS).
    exclude : set, optional
        Configurations to exclude (e.g., already sampled for train).
    rng : random.Random, optional
        Seeded RNG for reproducibility.

    Returns
    -------
    list[tuple[str, ...]]
    """
    if rng is None:
        rng = random.Random()

    all_configs = generate_all_possible_configs(model_pool, num_roles)
    if exclude:
        all_configs = [c for c in all_configs if c not in exclude]

    if n > len(all_configs):
        raise ValueError(
            f"Requested {n} unique configs but only {len(all_configs)} are "
            f"available (pool size={len(model_pool)}, roles={num_roles})."
        )

    return rng.sample(all_configs, n)


def generate_train_test_configs(
    n_train: int = 60,
    n_test: int = 60,
    model_pool: List[str] | None = None,
    seed: int = 42,
) -> Tuple[List[Tuple[str, ...]], List[Tuple[str, ...]]]:
    """
    Generate non-overlapping train and test configuration sets.

    Returns
    -------
    (train_configs, test_configs)
    """
    if model_pool is None:
        model_pool = MODEL_POOL

    num_roles = len(AGENT_ROLES)
    rng = random.Random(seed)

    train = sample_unique_configs(n_train, model_pool, num_roles, rng=rng)
    train_set = set(train)
    test = sample_unique_configs(n_test, model_pool, num_roles,
                                 exclude=train_set, rng=rng)

    return train, test


# ─── Serialisation helpers ───────────────────────────────────────────────────

def configs_to_dicts(
    configs: List[Tuple[str, ...]],
    start_index: int = 0,
) -> List[dict]:
    """Convert tuples into labelled dicts with sequential indices."""
    records = []
    for i, cfg in enumerate(configs):
        record = {"config_index": start_index + i}
        for role, model in zip(AGENT_ROLES, cfg):
            record[role] = model
        records.append(record)
    return records


def save_configs(records: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  ✓ Saved {len(records)} configs → {path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate surrogate configs")
    parser.add_argument("--n_train", type=int, default=60)
    parser.add_argument("--n_test", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="surrogate/data")
    args = parser.parse_args()

    total = len(MODEL_POOL) ** len(AGENT_ROLES)
    print(f"Model pool : {MODEL_POOL}")
    print(f"Agent roles: {AGENT_ROLES}")
    print(f"Total possible configs: {total}")
    print(f"Sampling {args.n_train} train + {args.n_test} test "
          f"(seed={args.seed})\n")

    train_cfgs, test_cfgs = generate_train_test_configs(
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
    )

    train_records = configs_to_dicts(train_cfgs, start_index=0)
    test_records = configs_to_dicts(test_cfgs, start_index=args.n_train)
    all_records = train_records + test_records

    save_configs(train_records, os.path.join(args.output_dir, "train_configs.json"))
    save_configs(test_records, os.path.join(args.output_dir, "test_configs.json"))
    save_configs(all_records, os.path.join(args.output_dir, "all_configs.json"))

    # Quick sanity check
    train_tuples = {tuple(r[role] for role in AGENT_ROLES) for r in train_records}
    test_tuples = {tuple(r[role] for role in AGENT_ROLES) for r in test_records}
    overlap = train_tuples & test_tuples
    print(f"\nSanity check — overlap between train & test: {len(overlap)} "
          f"(expected 0)")


if __name__ == "__main__":
    main()
