"""
Hypervolume computation and early-stopping tracker for 2-objective optimisation.

Objectives
----------
  accuracy : MAXIMISE
  cost     : MINIMISE

The hypervolume is computed in 2-D using a sweep-line algorithm after
extracting the non-dominated (Pareto) front from the evaluated points.
"""

import numpy as np
from typing import List


# ─── Pareto front ────────────────────────────────────────────────────────────

def compute_pareto_front(
    accuracies: np.ndarray,
    costs: np.ndarray,
) -> np.ndarray:
    """
    Return the *indices* of non-dominated solutions.

    A solution *i* dominates *j* iff
        acc[i] >= acc[j]  AND  cost[i] <= cost[j]
    with at least one strict inequality.
    """
    n = len(accuracies)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # j dominates i?
            if (accuracies[j] >= accuracies[i] and costs[j] <= costs[i] and
                    (accuracies[j] > accuracies[i] or costs[j] < costs[i])):
                is_dominated[i] = True
                break

    return np.where(~is_dominated)[0]


# ─── 2-D Hypervolume ────────────────────────────────────────────────────────

def hypervolume_2d(
    accuracies: np.ndarray,
    costs: np.ndarray,
    ref_acc: float = 0.0,
    ref_cost: float = None,
) -> float:
    """
    Compute 2-D hypervolume indicator for (maximise accuracy, minimise cost).

    Sweep algorithm on the Pareto front, sorted by accuracy ascending.
    Each Pareto point contributes a rectangle of width (acc_i - acc_{i-1})
    and height (ref_cost - cost_i).

    Parameters
    ----------
    accuracies : 1-D array of accuracy values (higher is better).
    costs      : 1-D array of cost values (lower is better).
    ref_acc    : reference accuracy (worst).  Default 0.
    ref_cost   : reference cost (worst / highest).  Default max(costs)*1.1 + 1.

    Returns
    -------
    float – hypervolume indicator.
    """
    if len(accuracies) == 0:
        return 0.0

    pareto_idx = compute_pareto_front(
        np.asarray(accuracies, dtype=float),
        np.asarray(costs, dtype=float),
    )
    if len(pareto_idx) == 0:
        return 0.0

    p_acc = np.asarray(accuracies, dtype=float)[pareto_idx]
    p_cost = np.asarray(costs, dtype=float)[pareto_idx]

    if ref_cost is None:
        ref_cost = float(np.max(costs)) * 1.1 + 1.0

    # Sort by accuracy ascending
    order = np.argsort(p_acc)
    p_acc = p_acc[order]
    p_cost = p_cost[order]

    hv = 0.0
    prev_acc = ref_acc

    for acc, cost in zip(p_acc, p_cost):
        if acc <= ref_acc or cost >= ref_cost:
            continue
        hv += (acc - prev_acc) * (ref_cost - cost)
        prev_acc = acc

    return hv


# ─── Early-stopping tracker ─────────────────────────────────────────────────

class EarlyStoppingTracker:
    """
    Track hypervolume across iterations and trigger early stopping
    when improvement falls below *tolerance* for *patience* consecutive
    iterations.
    """

    def __init__(self, tolerance: float = 0.01, patience: int = 3):
        self.tolerance = tolerance
        self.patience = patience
        self.history: List[float] = []
        self._stall_count = 0

    def update(self, hv: float) -> bool:
        """
        Record a new hypervolume value.

        Returns True if early stopping is triggered.
        """
        if len(self.history) > 0:
            prev = self.history[-1]
            improvement = (hv - prev) / max(abs(prev), 1e-12)
            if improvement < self.tolerance:
                self._stall_count += 1
            else:
                self._stall_count = 0
        else:
            self._stall_count = 0

        self.history.append(hv)

        triggered = self._stall_count >= self.patience
        if triggered:
            print(f"[EarlyStopping] Triggered after {len(self.history)} iterations "
                  f"(stall count = {self._stall_count}).", flush=True)
        return triggered

    @property
    def best_hv(self) -> float:
        return max(self.history) if self.history else 0.0

    @property
    def last_hv(self) -> float:
        return self.history[-1] if self.history else 0.0
