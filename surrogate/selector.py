"""
Pareto-aware Top-N selector using ParEGO-style random scalarisation + LCB.

Given a trained RF surrogate, this module:
1. Enumerates all untested configurations from MODEL_POOL^4.
2. Predicts (accuracy, cost) and per-tree uncertainty for each.
3. Uses multiple random weight vectors to scalarise the two objectives.
4. Applies Lower Confidence Bound acquisition to balance exploration/exploitation.
5. Returns N diverse, non-redundant candidates for the next iteration.
"""

import itertools
import numpy as np
from typing import List, Tuple, Set, Dict

from surrogate.config import MODEL_POOL, AGENT_ROLES, NONE_INELIGIBLE_ROLES


def _is_valid_config(cfg: Tuple[str, ...]) -> bool:
    """Check that no NONE_INELIGIBLE_ROLES role is assigned 'None'."""
    for role, model in zip(AGENT_ROLES, cfg):
        if model == "None" and role in NONE_INELIGIBLE_ROLES:
            return False
    return True


def _all_configs() -> List[Tuple[str, ...]]:
    """Enumerate all valid configs from MODEL_POOL (excluding constrained ones)."""
    return [c for c in itertools.product(MODEL_POOL, repeat=len(AGENT_ROLES))
            if _is_valid_config(c)]


def _configs_to_feature_rows(
    configs: List[Tuple[str, ...]],
    encoder,
) -> np.ndarray:
    """Encode a list of config tuples into numerical features."""
    return encoder.transform([list(c) for c in configs])


def _predict_with_uncertainty(
    model,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict means and per-tree standard deviations for each objective.

    Returns
    -------
    means : (n, 2) array – predicted [accuracy, cost]
    stds  : (n, 2) array – per-tree std for [accuracy, cost]
    """
    n = X.shape[0]
    n_obj = len(model.estimators_)

    means = model.predict(X)  # (n, 2)

    stds = np.zeros((n, n_obj))
    for obj_idx, estimator in enumerate(model.estimators_):
        tree_preds = np.array(
            [tree.predict(X) for tree in estimator.estimators_]
        )  # (n_trees, n)
        stds[:, obj_idx] = np.std(tree_preds, axis=0)

    return means, stds


def select_top_n(
    model,
    encoder,
    evaluated_configs: Set[Tuple[str, ...]],
    n: int = 5,
    kappa: float = 1.96,
    n_scalarizations: int = 50,
    rng_seed: int = None,
) -> List[Dict[str, str]]:
    """
    Select the top-N most promising untested configurations.

    Strategy: ParEGO-style random scalarisation with LCB acquisition.
      - For each of `n_scalarizations` random weight vectors (w_acc, w_cost):
        1. Normalise mean predictions to [0, 1].
        2. Scalarise:  s = w_acc × acc_norm − w_cost × cost_norm
           (maximise acc, minimise cost).
        3. Compute LCB = s_mean − κ × s_std   (explore via uncertainty).
        4. Pick the untested config with the highest LCB.
      - De-duplicate across scalarizations, return top-N unique.

    Parameters
    ----------
    model            : trained MultiOutputRegressor
    encoder          : fitted OrdinalEncoder
    evaluated_configs: set of already-evaluated (model, model, model, model) tuples
    n                : number of candidates to return
    kappa            : exploration parameter (higher = more exploration)
    n_scalarizations : number of random weight vectors to sample
    rng_seed         : optional random seed for reproducibility

    Returns
    -------
    List of dicts, each with keys from AGENT_ROLES.
    """
    rng = np.random.RandomState(rng_seed)

    # 1. Build the untested candidate set
    all_cfgs = _all_configs()
    untested = [c for c in all_cfgs if c not in evaluated_configs]

    if len(untested) == 0:
        print("[selector] WARNING: No untested configurations remain!")
        return []

    if len(untested) <= n:
        print(f"[selector] Only {len(untested)} untested configs remain; returning all.")
        return [dict(zip(AGENT_ROLES, c)) for c in untested]

    # 2. Predict means & uncertainty
    X = _configs_to_feature_rows(untested, encoder)
    means, stds = _predict_with_uncertainty(model, X)

    # 3. Normalise predictions to [0, 1]
    acc_min, acc_max = means[:, 0].min(), means[:, 0].max()
    cost_min, cost_max = means[:, 1].min(), means[:, 1].max()

    acc_range = max(acc_max - acc_min, 1e-8)
    cost_range = max(cost_max - cost_min, 1e-8)

    acc_norm = (means[:, 0] - acc_min) / acc_range
    cost_norm = (means[:, 1] - cost_min) / cost_range

    acc_std_norm = stds[:, 0] / acc_range
    cost_std_norm = stds[:, 1] / cost_range

    # 4. ParEGO: random scalarizations → LCB → pick best per scalarization
    selected_indices = set()
    candidate_scores = []  # (index, best_lcb) for tie-breaking

    for _ in range(n_scalarizations):
        # Random weight on the simplex
        w_acc = rng.uniform(0.05, 0.95)
        w_cost = 1.0 - w_acc

        # Scalarised mean (higher is better: +acc, −cost)
        s_mean = w_acc * acc_norm - w_cost * cost_norm

        # Scalarised std (combined uncertainty)
        s_std = np.sqrt((w_acc * acc_std_norm) ** 2 + (w_cost * cost_std_norm) ** 2)

        # LCB: we *maximise* s_mean, so LCB = s_mean - kappa * s_std
        # actually we want to *maximise* the acquisition, so use
        # UCB-like: score = s_mean + kappa * s_std  (explore high-uncertainty)
        # But ParEGO convention: minimise the scalarised objective
        # Let's think clearly:
        #   s_mean = w_acc * acc_norm - w_cost * cost_norm
        #   We want to MAXIMISE s_mean (high acc, low cost).
        #   To explore, we add uncertainty bonus:
        #   acquisition = s_mean + kappa * s_std
        acquisition = s_mean + kappa * s_std

        best_idx = np.argmax(acquisition)
        selected_indices.add(best_idx)
        candidate_scores.append((best_idx, acquisition[best_idx]))

    # 5. Rank all unique selected indices by their best score
    best_score_per_idx = {}
    for idx, score in candidate_scores:
        if idx not in best_score_per_idx or score > best_score_per_idx[idx]:
            best_score_per_idx[idx] = score

    ranked = sorted(best_score_per_idx.keys(),
                    key=lambda i: best_score_per_idx[i], reverse=True)

    # If we didn't get enough unique candidates from scalarizations,
    # fill from the overall best-predicted configs
    if len(ranked) < n:
        # Use predicted accuracy as tiebreaker
        for fallback_idx in np.argsort(-means[:, 0]):
            if fallback_idx not in set(ranked):
                ranked.append(fallback_idx)
            if len(ranked) >= n:
                break

    top_n_indices = ranked[:n]

    # 6. Convert to dicts
    results = []
    for idx in top_n_indices:
        cfg_tuple = untested[idx]
        cfg_dict = dict(zip(AGENT_ROLES, cfg_tuple))
        pred_acc = means[idx, 0]
        pred_cost = means[idx, 1]
        unc_acc = stds[idx, 0]
        unc_cost = stds[idx, 1]
        print(f"  [selector] Selected: {cfg_dict}  "
              f"pred_acc={pred_acc:.2f}  pred_cost={pred_cost:.2f}  "
              f"unc_acc={unc_acc:.2f}  unc_cost={unc_cost:.2f}")
        results.append(cfg_dict)

    return results
