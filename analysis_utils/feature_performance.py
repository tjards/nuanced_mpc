"""
Feature-performance diagnostics for Nuanced MPC / CALA.

Purpose
-------
This analysis separates three distinct questions:

1. DISTURBANCE ESTIMATOR PERFORMANCE
   How much of the true disturbance d_true is captured by d_hat?

2. CALA POLICY USE
   What is the magnitude of the learned CALA adjustment, and which configured
   feature groups (bias / radial / time / ...) contribute to it?

3. SPATIAL ALIGNMENT
   Does the structured residual left by the disturbance estimator occur in the
   same regions where the learned CALA R policy departs from nominal?

4. CONTEXT-LEARNING DIAGNOSTICS
   Did different contextual features actually learn different parameters, and
   does matching the learned policy to the current context materially change
   the policy output relative to a context-collapsed or shuffled policy?

5. FEATURE REPRESENTATION VALUE (OPTIONAL)
   Does augmenting the current context basis with radial x time interactions
   improve out-of-fold prediction of CALA's actual learning/performance signal
   (reward and advantage), conditional on the sampled CALA adjustment?

Important
---------
CALA is NOT assumed to model d_true directly.

If the configured mode is a disturbance adjustment, the module may compare

    d_true
    d_hat
    d_hat + CALA residual correction

because that is dimensionally meaningful.

If the configured mode is R adjustment, CALA is plotted separately as

    R / R0 = exp(rl_adjustment)

and is never added to d_hat.

Existing feature activations are read from the CALA training HDF5 when
available. For evaluation-policy plots they are reconstructed with
cala.RTFeatureMap, which preserves the normalization and group weights from
config_cala.json.

The proposed space x time diagnostic basis is constructed as all products
between the existing radial and temporal feature activations. It is used only
for offline diagnostic regression; it does not alter CALA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

from cala import RTFeatureMap


_EPS = 1e-12


# =============================================================================
# Helpers
# =============================================================================

def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _resolve_path(path_value: str | Path, configs_base: Path) -> Path:
    p = Path(path_value)

    if p.is_absolute():
        return p

    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate

    config_candidate = configs_base / p
    if config_candidate.exists():
        return config_candidate

    return cwd_candidate


def _read_h5_group(filepath: Path, group: str) -> Dict[str, np.ndarray]:
    if not filepath.exists():
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        if group not in f:
            raise KeyError(
                f"Group '{group}' not found in {filepath}. "
                f"Available groups: {list(f.keys())}"
            )

        g = f[group]
        return {key: g[key][...] for key in g.keys()}


def _first_existing(data: dict, names: Sequence[str]) -> Optional[np.ndarray]:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    return None


def _find_key_recursive(obj, key: str):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_key_recursive(value, key)
            if found is not None:
                return found
    return None


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _score(y: np.ndarray, y_hat: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y_hat.ndim == 1:
        y_hat = y_hat.reshape(-1, 1)

    err = y_hat - y

    rmse_per_channel = np.sqrt(np.mean(err**2, axis=0))
    rmse = float(np.sqrt(np.mean(err**2)))

    rms_true_per_channel = np.sqrt(np.mean(y**2, axis=0))
    nrmse_per_channel = rmse_per_channel / (rms_true_per_channel + _EPS)

    ss_res = np.sum(err**2, axis=0)
    y_mean = np.mean(y, axis=0, keepdims=True)
    ss_tot = np.sum((y - y_mean) ** 2, axis=0)
    r2_per_channel = 1.0 - ss_res / (ss_tot + _EPS)

    ss_res_all = float(np.sum(err**2))
    y_mean_all = float(np.mean(y))
    ss_tot_all = float(np.sum((y - y_mean_all) ** 2))
    r2 = 1.0 - ss_res_all / (ss_tot_all + _EPS)

    rms_true = float(np.sqrt(np.mean(y**2)))
    nrmse = rmse / (rms_true + _EPS)

    return {
        "rmse": rmse,
        "rmse_per_channel": rmse_per_channel,
        "rms_true": rms_true,
        "rms_true_per_channel": rms_true_per_channel,
        "nrmse": float(nrmse),
        "nrmse_per_channel": nrmse_per_channel,
        "r2": float(r2),
        "r2_per_channel": r2_per_channel,
        "rms_fraction_captured": float(1.0 - nrmse),
        "rms_fraction_captured_per_channel": 1.0 - nrmse_per_channel,
    }



def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average ranks for ties; sufficient for dependency-free Spearman rho."""
    x = np.asarray(x, dtype=float).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)

    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1

        # Zero-based average rank for the tied block.
        rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = rank
        i = j

    return ranks


def _correlation_pair(x: np.ndarray, y: np.ndarray) -> dict:
    """Return finite-sample Pearson and Spearman correlations."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return {"n": int(len(x)), "pearson": np.nan, "spearman": np.nan}

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])

    rx = _rankdata_average(x)
    ry = _rankdata_average(y)

    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        spearman = np.nan
    else:
        spearman = float(np.corrcoef(rx, ry)[0, 1])

    return {
        "n": int(len(x)),
        "pearson": pearson,
        "spearman": spearman,
    }


def _spatial_bin_mean(
    xy: np.ndarray,
    values: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    min_count: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean scalar value in each spatial bin.

    Returns
    -------
    mean_grid : (ny, nx)
        NaN in bins with fewer than min_count samples.
    count_grid : (ny, nx)
    """
    xy = np.asarray(xy, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)

    n = min(len(xy), len(values))
    xy = xy[:n]
    values = values[:n]

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    sums = np.zeros((ny, nx), dtype=float)
    counts = np.zeros((ny, nx), dtype=int)

    ix = np.digitize(xy[:, 0], x_edges) - 1
    iy = np.digitize(xy[:, 1], y_edges) - 1

    # Include samples exactly on the final right/top edge.
    ix[np.isclose(xy[:, 0], x_edges[-1])] = nx - 1
    iy[np.isclose(xy[:, 1], y_edges[-1])] = ny - 1

    valid = (
        np.isfinite(values)
        & np.isfinite(xy[:, 0])
        & np.isfinite(xy[:, 1])
        & (ix >= 0)
        & (ix < nx)
        & (iy >= 0)
        & (iy < ny)
    )

    for k in np.where(valid)[0]:
        sums[iy[k], ix[k]] += values[k]
        counts[iy[k], ix[k]] += 1

    means = np.full((ny, nx), np.nan, dtype=float)
    keep = counts >= int(min_count)
    means[keep] = sums[keep] / counts[keep]

    return means, counts

def _fit_standardized_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Standardized ridge regression with an unpenalized intercept.

    Scaling is learned on the training fold only.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    keep = std > 1e-12

    # If every feature is constant, predict training mean.
    if not np.any(keep):
        return np.full(X_test.shape[0], np.mean(y_train))

    Xtr = (X_train[:, keep] - mean[keep]) / std[keep]
    Xte = (X_test[:, keep] - mean[keep]) / std[keep]

    y_mean = float(np.mean(y_train))
    yc = y_train - y_mean

    p = Xtr.shape[1]
    A = Xtr.T @ Xtr + float(alpha) * np.eye(p)
    b = Xtr.T @ yc

    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]

    return y_mean + Xte @ beta


def _blocked_cv_prediction(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, dict]:
    """
    Contiguous blocked cross-validation.

    Each fold is one contiguous segment of the learning history. The model is
    trained on all other segments and predicts the held-out segment.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n = len(y)
    if n < max(20, n_folds * 4):
        raise ValueError(
            f"Only {n} learning samples are available; too few for "
            f"{n_folds}-fold blocked cross-validation."
        )

    n_folds = int(max(2, min(n_folds, n // 4)))
    folds = np.array_split(np.arange(n), n_folds)
    pred = np.full(n, np.nan, dtype=float)

    all_idx = np.arange(n)

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_idx[train_mask]

        pred[test_idx] = _fit_standardized_ridge(
            X[train_idx],
            y[train_idx],
            X[test_idx],
            alpha=alpha,
        )

    valid = np.isfinite(pred)
    return pred, _score(y[valid], pred[valid])


# =============================================================================
# Main analyzer
# =============================================================================

class FeaturePerformanceAnalyzer:

    def __init__(
        self,
        configs_base: str | Path,
        evaluation_path: Optional[str | Path] = None,
        training_path: Optional[str | Path] = None,
        output_dir: str | Path = "visualization/feature_performance",
        evaluation_group: str = "rl_learned_R",
        training_group: str = "learning",
    ):
        self.configs_base = Path(configs_base)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluation_group = evaluation_group
        self.training_group = training_group

        self.cfg_data = _load_json(self.configs_base / "config_data.json")
        self.cfg_cala = _load_json(self.configs_base / "config_cala.json")
        self.cfg_mpc = _load_json(self.configs_base / "config_mpc.json")

        plant_path = self.configs_base / "config_plant.json"
        self.cfg_plant = _load_json(plant_path) if plant_path.exists() else {}

        if evaluation_path is None:
            evaluation_path = self.cfg_data.get("rl_evaluation_path")
        if training_path is None:
            training_path = self.cfg_data.get("rl_training_path")

        self.evaluation_path = (
            _resolve_path(evaluation_path, self.configs_base)
            if evaluation_path is not None
            else None
        )
        self.training_path = (
            _resolve_path(training_path, self.configs_base)
            if training_path is not None
            else None
        )

        self.feature_map = RTFeatureMap(str(self.configs_base))
        self.d_max = float(self.cfg_cala["cala_mpc_nld"]["d_max"])

        hm_cfg = self.cfg_cala.get("horizon_manager", {})
        self.reward_mode = hm_cfg.get("reward_mode", None)

        self.rl_parameter = self._detect_rl_parameter()

        self.eval_data = None
        self.train_data = None

        if self.evaluation_path is not None and self.evaluation_path.exists():
            self.eval_data = _read_h5_group(
                self.evaluation_path, self.evaluation_group
            )

        if self.training_path is not None and self.training_path.exists():
            self.train_data = _read_h5_group(
                self.training_path, self.training_group
            )

        if self.eval_data is None and self.train_data is None:
            raise FileNotFoundError(
                "Neither evaluation nor training HDF5 data could be loaded."
            )

        self.current_feature_names = list(self.feature_map.names)
        self.current_group_indices = self._group_index_arrays()

    # -------------------------------------------------------------------------
    # Configuration / mode
    # -------------------------------------------------------------------------

    def _detect_rl_parameter(self) -> str:
        value = _find_key_recursive(self.cfg_mpc, "rl_parameter")

        if value is None:
            value = _find_key_recursive(self.cfg_cala, "rl_parameter")

        if value is not None:
            text = str(value).strip().lower()
            if "r_adjust" in text or text in {"r", "r_adjustment"}:
                return "R_adjustment"
            if "d_adjust" in text or text in {"d", "d_adjustment"}:
                return "d_adjustment"

        # Current code's for_r reward is specifically R learning.
        if self.reward_mode == "for_r":
            return "R_adjustment"

        return "unknown"

    def _group_index_arrays(self) -> Dict[str, np.ndarray]:
        groups = {}
        for name, slc in self.feature_map.names_index.items():
            groups[name] = np.arange(len(self.feature_map.names))[slc]
        return groups

    # -------------------------------------------------------------------------
    # Data extraction
    # -------------------------------------------------------------------------

    def _extract_disturbance_series(self) -> dict:
        """
        Prefer evaluation data when it contains both d_true and d_hat.
        Otherwise use the CALA training history, where cala.py explicitly stores
        both d_true and d_hat at every completed learning trial.
        """

        sources = []

        if self.eval_data is not None:
            sources.append(("evaluation", self.eval_data))
        if self.train_data is not None:
            sources.append(("training", self.train_data))

        for source_name, data in sources:
            d_true = _first_existing(
                data,
                ["d_true", "d", "disturbance", "disturbance_true"],
            )
            d_hat = _first_existing(
                data,
                ["d_hat", "disturbance_estimate", "estimated_disturbance"],
            )

            if d_true is None or d_hat is None:
                continue

            d_true = _as_2d(d_true)
            d_hat = _as_2d(d_hat)

            step = _first_existing(data, ["step", "time", "t"])
            if step is None:
                step = np.arange(min(len(d_true), len(d_hat)), dtype=float)
            else:
                step = np.asarray(step, dtype=float).reshape(-1)

            n = min(len(step), len(d_true), len(d_hat))
            step = step[:n]
            d_true = d_true[:n]
            d_hat = d_hat[:n]

            rl_mean = _first_existing(data, ["rl_mean"])
            rl_adjustment = _first_existing(data, ["rl_adjustment"])

            if rl_mean is not None:
                rl_mean = _as_2d(rl_mean)[:n]

            if rl_adjustment is not None:
                rl_adjustment = _as_2d(rl_adjustment)[:n]

            return {
                "source": source_name,
                "step": step,
                "d_true": d_true,
                "d_hat": d_hat,
                "rl_mean": rl_mean,
                "rl_adjustment": rl_adjustment,
            }

        eval_keys = sorted(self.eval_data.keys()) if self.eval_data is not None else []
        train_keys = sorted(self.train_data.keys()) if self.train_data is not None else []

        raise KeyError(
            "Could not find both true disturbance and disturbance estimate. "
            f"Evaluation keys: {eval_keys}. Training keys: {train_keys}."
        )

    def _training_arrays(self) -> Optional[dict]:
        if self.train_data is None:
            return None

        required = {"phi", "rl_adjustment", "reward"}
        missing = required - set(self.train_data)

        if missing:
            print(
                "FeaturePerformanceAnalyzer: training data is missing "
                f"{sorted(missing)}; skipping reward-representation analysis."
            )
            return None

        phi = np.asarray(self.train_data["phi"], dtype=float)
        action = _as_2d(self.train_data["rl_adjustment"])
        reward = np.asarray(self.train_data["reward"], dtype=float).reshape(-1)

        step = _first_existing(self.train_data, ["step", "time", "t"])
        if step is None:
            step = np.arange(len(reward), dtype=float)
        else:
            step = np.asarray(step, dtype=float).reshape(-1)

        advantage = _first_existing(
            self.train_data,
            ["advantage_scaled", "advantage"],
        )
        if advantage is not None:
            advantage = np.asarray(advantage, dtype=float).reshape(-1)

        d_true = _first_existing(self.train_data, ["d_true", "d"])
        d_hat = _first_existing(self.train_data, ["d_hat"])

        n_candidates = [len(phi), len(action), len(reward), len(step)]
        if advantage is not None:
            n_candidates.append(len(advantage))
        if d_true is not None:
            n_candidates.append(len(d_true))
        if d_hat is not None:
            n_candidates.append(len(d_hat))

        n = min(n_candidates)

        out = {
            "step": step[:n],
            "phi": phi[:n],
            "action": action[:n],
            "reward": reward[:n],
            "advantage": advantage[:n] if advantage is not None else None,
            "d_true": _as_2d(d_true)[:n] if d_true is not None else None,
            "d_hat": _as_2d(d_hat)[:n] if d_hat is not None else None,
        }

        if out["phi"].shape[1] != len(self.feature_map.names):
            raise ValueError(
                f"Stored phi has {out['phi'].shape[1]} features, but the "
                f"configured RTFeatureMap has {len(self.feature_map.names)}."
            )

        return out

    # -------------------------------------------------------------------------
    # Evaluation feature reconstruction / learned policy
    # -------------------------------------------------------------------------

    def _evaluation_context(self) -> Optional[dict]:
        """
        Reconstruct the context used for the evaluation policy.

        The evaluation logger used in this project stores post-step state. When
        Ts and x0 are available, shift state back one sample so the spatial
        context corresponds to the controller/disturbance decision that
        generated the stored row.
        """
        if self.eval_data is None:
            return None

        state = _first_existing(self.eval_data, ["state", "x"])
        step = _first_existing(self.eval_data, ["step", "time", "t"])

        if state is None or step is None:
            return None

        state = _as_2d(state)
        step = np.asarray(step, dtype=float).reshape(-1)

        n = min(len(state), len(step))
        state = state[:n]
        step = step[:n]

        Ts = _find_key_recursive(self.cfg_mpc, "Ts")
        x0 = _find_key_recursive(self.cfg_plant, "x0")

        if Ts is not None and x0 is not None and n > 0:
            Ts = float(Ts)
            x0 = np.asarray(x0, dtype=float).reshape(-1)

            context_state = np.empty_like(state)
            context_state[0] = x0[: state.shape[1]]
            if n > 1:
                context_state[1:] = state[:-1]

            # Keep both row time and the context time. The row index is what
            # aligns the context with d_true/d_hat in the same evaluation HDF5.
            context_time = step - Ts
        else:
            context_state = state
            context_time = step

        phi = np.vstack(
            [
                self.feature_map.build_features(x, t)
                for x, t in zip(context_state, context_time)
            ]
        )

        return {
            "row_time": step,
            "context_time": context_time,
            "context_state": context_state,
            "phi": phi,
        }

    def _evaluation_phi(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        context = self._evaluation_context()
        if context is None:
            return None
        return context["context_time"], context["phi"]

    def _load_final_mu(self) -> Optional[np.ndarray]:
        if self.train_data is None or "mu" not in self.train_data:
            return None

        mu_hist = np.asarray(self.train_data["mu"], dtype=float)
        if len(mu_hist) == 0:
            return None

        return mu_hist[-1].copy()

    def _policy_series(self) -> Optional[dict]:
        """
        Obtain mean CALA policy over an evaluation trajectory when possible.
        Falls back to the training trial-start contexts.
        """
        mu = self._load_final_mu()
        if mu is None:
            return None

        theta = 2.0 * self.d_max * (mu - 0.5)

        eval_context = self._evaluation_context()

        if eval_context is not None:
            step = eval_context["context_time"]
            row_time = eval_context["row_time"]
            context_state = eval_context["context_state"]
            phi = eval_context["phi"]
            source = "evaluation"
        else:
            training = self._training_arrays()
            if training is None:
                return None
            step = training["step"]
            row_time = training["step"]
            context_state = None
            phi = training["phi"]
            source = "training"

        if phi.shape[1] != theta.shape[0]:
            raise ValueError(
                f"Policy mu has {theta.shape[0]} features but phi has "
                f"{phi.shape[1]}."
            )

        # Per-feature, per-input mean policy contribution.
        contribution = phi[:, :, None] * theta[None, :, :]

        # Exact CALA mean adjustment.
        rl_mean = np.sum(contribution, axis=1)

        # Absolute group contribution, avoiding cancellation within a group.
        feature_magnitude = np.linalg.norm(contribution, axis=2)

        group_names = list(self.current_group_indices.keys())
        group_magnitude = np.column_stack(
            [
                np.sum(
                    feature_magnitude[:, self.current_group_indices[name]],
                    axis=1,
                )
                for name in group_names
            ]
        )

        total = np.sum(group_magnitude, axis=1, keepdims=True)
        share = group_magnitude / (total + _EPS)

        return {
            "source": source,
            "step": step,
            "row_time": row_time,
            "context_state": context_state,
            "phi": phi,
            "theta": theta,
            "rl_mean": rl_mean,
            "group_names": group_names,
            "group_magnitude": group_magnitude,
            "share": share,
            "mean_share": np.mean(share, axis=0),
            "mean_group_magnitude": np.mean(group_magnitude, axis=0),
        }

    # -------------------------------------------------------------------------
    # Spatial relationship: estimator residual vs learned R policy
    # -------------------------------------------------------------------------

    def _spatial_alignment(
        self,
        policy: dict,
        dseries: dict,
        spatial_bins: int = 10,
        min_bin_count: int = 3,
        warmup_seconds: float = 1.0,
    ) -> Optional[dict]:
        """
        Compare the spatial structure left by the disturbance estimator with
        the spatial structure of the learned CALA policy.

        For R learning, two relationships are useful:

        1. residual magnitude vs R/R0
           A negative correlation means larger residuals tend to coincide with
           more aggressive control (smaller R).

        2. residual magnitude vs |log(R/R0)|
           A positive correlation means larger residuals tend to coincide with
           a stronger departure from nominal R, regardless of direction.

        Both sample-wise and occupied-spatial-bin correlations are reported.
        """
        if self.rl_parameter != "R_adjustment":
            return None

        if policy is None or policy.get("source") != "evaluation":
            return None

        context_state = policy.get("context_state")
        if context_state is None:
            return None

        if dseries.get("source") != "evaluation":
            return None

        d_true = _as_2d(dseries["d_true"])
        d_hat = _as_2d(dseries["d_hat"])
        rl_mean = _as_2d(policy["rl_mean"])
        row_time = np.asarray(
            policy.get("row_time", policy["step"]),
            dtype=float,
        ).reshape(-1)
        context_state = _as_2d(context_state)

        n = min(
            len(d_true),
            len(d_hat),
            len(rl_mean),
            len(row_time),
            len(context_state),
        )

        d_true = d_true[:n]
        d_hat = d_hat[:n]
        rl_mean = rl_mean[:n]
        row_time = row_time[:n]
        context_state = context_state[:n]

        residual = d_true - d_hat
        residual_norm = np.linalg.norm(residual, axis=1)

        r_scale = np.exp(rl_mean)
        log_r_deviation = np.abs(rl_mean)
        policy_log_norm = np.linalg.norm(rl_mean, axis=1)

        # Exclude only the requested estimator startup interval. This avoids
        # letting the initial observer transient dominate the spatial result.
        if len(row_time) > 0:
            warmup_end = float(row_time[0] + warmup_seconds)
            mask = row_time >= warmup_end
        else:
            warmup_end = np.nan
            mask = np.ones(n, dtype=bool)

        mask &= np.all(np.isfinite(context_state[:, :2]), axis=1)
        mask &= np.isfinite(residual_norm)
        mask &= np.all(np.isfinite(rl_mean), axis=1)

        if np.count_nonzero(mask) < 10:
            return None

        xy = context_state[mask, :2]
        residual = residual[mask]
        residual_norm = residual_norm[mask]
        r_scale = r_scale[mask]
        log_r_deviation = log_r_deviation[mask]
        policy_log_norm = policy_log_norm[mask]
        row_time = row_time[mask]

        x_edges = np.linspace(
            self.feature_map.x_lims[0],
            self.feature_map.x_lims[1],
            int(spatial_bins) + 1,
        )
        y_edges = np.linspace(
            self.feature_map.y_lims[0],
            self.feature_map.y_lims[1],
            int(spatial_bins) + 1,
        )

        residual_norm_map, occupancy = _spatial_bin_mean(
            xy,
            residual_norm,
            x_edges,
            y_edges,
            min_count=min_bin_count,
        )
        policy_log_norm_map, _ = _spatial_bin_mean(
            xy,
            policy_log_norm,
            x_edges,
            y_edges,
            min_count=min_bin_count,
        )

        n_inputs = min(residual.shape[1], rl_mean.shape[1])

        residual_abs_maps = []
        r_scale_maps = []
        log_r_deviation_maps = []

        sample_corr = {}
        binned_corr = {}

        for j in range(n_inputs):
            residual_abs = np.abs(residual[:, j])

            residual_map, _ = _spatial_bin_mean(
                xy,
                residual_abs,
                x_edges,
                y_edges,
                min_count=min_bin_count,
            )
            scale_map, _ = _spatial_bin_mean(
                xy,
                r_scale[:, j],
                x_edges,
                y_edges,
                min_count=min_bin_count,
            )
            logdev_map, _ = _spatial_bin_mean(
                xy,
                log_r_deviation[:, j],
                x_edges,
                y_edges,
                min_count=min_bin_count,
            )

            residual_abs_maps.append(residual_map)
            r_scale_maps.append(scale_map)
            log_r_deviation_maps.append(logdev_map)

            sample_corr[f"channel_{j}"] = {
                "abs_residual_vs_R_over_R0": _correlation_pair(
                    residual_abs,
                    r_scale[:, j],
                ),
                "abs_residual_vs_abs_log_R_ratio": _correlation_pair(
                    residual_abs,
                    log_r_deviation[:, j],
                ),
            }

            binned_corr[f"channel_{j}"] = {
                "mean_abs_residual_vs_mean_R_over_R0": _correlation_pair(
                    residual_map.ravel(),
                    scale_map.ravel(),
                ),
                "mean_abs_residual_vs_mean_abs_log_R_ratio": _correlation_pair(
                    residual_map.ravel(),
                    logdev_map.ravel(),
                ),
            }

        sample_corr["total"] = {
            "residual_norm_vs_policy_log_norm": _correlation_pair(
                residual_norm,
                policy_log_norm,
            )
        }

        binned_corr["total"] = {
            "mean_residual_norm_vs_mean_policy_log_norm": _correlation_pair(
                residual_norm_map.ravel(),
                policy_log_norm_map.ravel(),
            )
        }

        return {
            "warmup_seconds": float(warmup_seconds),
            "warmup_end_time": warmup_end,
            "n_samples": int(len(xy)),
            "spatial_bins": int(spatial_bins),
            "min_bin_count": int(min_bin_count),
            "xy": xy,
            "time": row_time,
            "residual": residual,
            "residual_norm": residual_norm,
            "r_scale": r_scale,
            "log_r_deviation": log_r_deviation,
            "policy_log_norm": policy_log_norm,
            "x_edges": x_edges,
            "y_edges": y_edges,
            "occupancy": occupancy,
            "residual_norm_map": residual_norm_map,
            "policy_log_norm_map": policy_log_norm_map,
            "residual_abs_maps": residual_abs_maps,
            "r_scale_maps": r_scale_maps,
            "log_r_deviation_maps": log_r_deviation_maps,
            "sample_correlations": sample_corr,
            "binned_spatial_correlations": binned_corr,
        }

    # -------------------------------------------------------------------------
    # Candidate product features / performance models
    # -------------------------------------------------------------------------

    def _build_space_time_products(
        self,
        phi: np.ndarray,
    ) -> Tuple[np.ndarray, list[str]]:
        if "radial" not in self.current_group_indices:
            raise ValueError("No radial feature group found.")
        if "time" not in self.current_group_indices:
            raise ValueError(
                "No time feature group found; cannot build space x time products."
            )

        radial_idx = self.current_group_indices["radial"]
        time_idx = self.current_group_indices["time"]

        radial = phi[:, radial_idx]
        temporal = phi[:, time_idx]

        products = (
            radial[:, :, None] * temporal[:, None, :]
        ).reshape(len(phi), -1)

        names = []
        for r_i in radial_idx:
            for t_i in time_idx:
                names.append(
                    f"{self.current_feature_names[r_i]}*"
                    f"{self.current_feature_names[t_i]}"
                )

        return np.column_stack([phi, products]), names

    @staticmethod
    def _reward_design(phi: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Contextual reward model.

        Reward depends on both context and the sampled CALA adjustment. To test
        whether context features help explain the reward landscape, include:

            context
            action
            context x action

        This lets reward sensitivity to the CALA adjustment vary with context.
        """
        phi = np.asarray(phi, dtype=float)
        action = _as_2d(action)

        interaction = (
            phi[:, :, None] * action[:, None, :]
        ).reshape(len(phi), -1)

        return np.column_stack([phi, action, interaction])

    def _performance_representation(
        self,
        n_folds: int,
        ridge_alpha: float,
    ) -> Optional[dict]:
        training = self._training_arrays()
        if training is None:
            return None

        phi = training["phi"]
        action = training["action"]

        phi_aug, product_names = self._build_space_time_products(phi)

        X_current = self._reward_design(phi, action)
        X_augmented = self._reward_design(phi_aug, action)

        targets = {"reward": training["reward"]}
        if training["advantage"] is not None:
            targets["advantage"] = training["advantage"]

        results = {
            "step": training["step"],
            "n_current_context_features": int(phi.shape[1]),
            "n_product_context_features": int(len(product_names)),
            "n_augmented_context_features": int(phi_aug.shape[1]),
            "targets": {},
        }

        for name, y in targets.items():
            pred_current, score_current = _blocked_cv_prediction(
                X_current,
                y,
                n_folds=n_folds,
                alpha=ridge_alpha,
            )
            pred_augmented, score_augmented = _blocked_cv_prediction(
                X_augmented,
                y,
                n_folds=n_folds,
                alpha=ridge_alpha,
            )

            gain_pct = float(
                100.0
                * (score_current["rmse"] - score_augmented["rmse"])
                / (score_current["rmse"] + _EPS)
            )

            results["targets"][name] = {
                "actual": y,
                "pred_current": pred_current,
                "pred_augmented": pred_augmented,
                "current": score_current,
                "augmented": score_augmented,
                "space_time_rmse_improvement_pct": gain_pct,
            }

        # Group ablation on the actual CALA learning signal (advantage when
        # available, otherwise reward).
        target_name = "advantage" if "advantage" in targets else "reward"
        y = targets[target_name]

        baseline_pred, baseline_score = _blocked_cv_prediction(
            X_current,
            y,
            n_folds=n_folds,
            alpha=ridge_alpha,
        )
        baseline_rmse = baseline_score["rmse"]

        ablation = {}

        all_idx = np.arange(phi.shape[1])

        for group_name, group_idx in self.current_group_indices.items():
            keep = np.setdiff1d(all_idx, group_idx)

            # Do not create an empty context basis.
            if len(keep) == 0:
                ablation[group_name] = np.nan
                continue

            X_removed = self._reward_design(phi[:, keep], action)

            _, score_removed = _blocked_cv_prediction(
                X_removed,
                y,
                n_folds=n_folds,
                alpha=ridge_alpha,
            )

            ablation[group_name] = float(
                100.0
                * (score_removed["rmse"] - baseline_rmse)
                / (baseline_rmse + _EPS)
            )

        aug_gain = results["targets"][target_name][
            "space_time_rmse_improvement_pct"
        ]

        results["group_value_target"] = target_name
        results["current_group_ablation_rmse_change_pct"] = ablation
        results["space_time_addition_rmse_improvement_pct"] = aug_gain

        return results

    # -------------------------------------------------------------------------
    # Plots: estimator
    # -------------------------------------------------------------------------

    def _plot_disturbance_estimate(self, dseries: dict) -> None:
        t = dseries["step"]
        d_true = dseries["d_true"]
        d_hat = dseries["d_hat"]

        n_inputs = min(d_true.shape[1], d_hat.shape[1])

        fig, axes = plt.subplots(
            n_inputs,
            1,
            figsize=(11, max(3.2 * n_inputs, 4.5)),
            sharex=True,
        )
        axes = np.atleast_1d(axes)

        for j, ax in enumerate(axes):
            ax.plot(t, d_true[:, j], linewidth=2.0, label="True disturbance")
            ax.plot(t, d_hat[:, j], linewidth=1.7, label="Disturbance estimate")
            ax.set_ylabel(fr"$d_{j}$")
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel("Simulation time [s]")
        fig.suptitle(
            f"True disturbance vs estimator ({dseries['source']} data)"
        )
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "01_disturbance_true_vs_estimate.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_estimator_residual(self, dseries: dict) -> None:
        t = dseries["step"]
        residual = dseries["d_true"] - dseries["d_hat"]

        n_inputs = residual.shape[1]

        fig, axes = plt.subplots(
            n_inputs,
            1,
            figsize=(11, max(3.2 * n_inputs, 4.5)),
            sharex=True,
        )
        axes = np.atleast_1d(axes)

        for j, ax in enumerate(axes):
            ax.plot(t, residual[:, j], linewidth=1.8)
            ax.axhline(0.0, linewidth=1.0, linestyle="--")
            ax.set_ylabel(fr"$d_{{true,{j}}}-\hat{{d}}_{j}$")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Simulation time [s]")
        fig.suptitle("Residual left after disturbance estimation")
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "02_disturbance_estimator_residual.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_estimator_magnitude(self, dseries: dict) -> None:
        t = dseries["step"]
        d_true = dseries["d_true"]
        d_hat = dseries["d_hat"]
        residual = d_true - d_hat

        true_mag = np.linalg.norm(d_true, axis=1)
        hat_mag = np.linalg.norm(d_hat, axis=1)
        residual_mag = np.linalg.norm(residual, axis=1)

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(11, 7),
            sharex=True,
        )

        axes[0].plot(t, true_mag, label=r"$\|d_{true}\|$")
        axes[0].plot(t, hat_mag, label=r"$\|\hat d\|$")
        axes[0].set_ylabel("Magnitude")
        axes[0].set_title("Disturbance magnitude captured by estimator")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(t, residual_mag, label=r"$\|d_{true}-\hat d\|$")
        axes[1].set_ylabel("Residual magnitude")
        axes[1].set_xlabel("Simulation time [s]")
        axes[1].set_title("What remains after the disturbance estimate")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(
            self.output_dir / "03_disturbance_estimator_magnitude.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_d_adjustment_if_applicable(self, dseries: dict) -> None:
        if self.rl_parameter != "d_adjustment":
            return

        correction = dseries["rl_mean"]
        if correction is None:
            return

        t = dseries["step"]
        d_true = dseries["d_true"]
        d_hat = dseries["d_hat"]

        n = min(len(t), len(correction), len(d_true), len(d_hat))
        t = t[:n]
        correction = correction[:n]
        d_true = d_true[:n]
        d_hat = d_hat[:n]

        n_inputs = min(
            correction.shape[1],
            d_true.shape[1],
            d_hat.shape[1],
        )

        fig, axes = plt.subplots(
            n_inputs,
            1,
            figsize=(11, max(3.2 * n_inputs, 4.5)),
            sharex=True,
        )
        axes = np.atleast_1d(axes)

        for j, ax in enumerate(axes):
            residual_before = d_true[:, j] - d_hat[:, j]
            residual_after = residual_before - correction[:, j]

            ax.plot(
                t,
                residual_before,
                label=r"$d_{true}-\hat d$",
            )
            ax.plot(
                t,
                correction[:, j],
                label="CALA mean residual correction",
            )
            ax.plot(
                t,
                residual_after,
                label=r"$d_{true}-(\hat d+d_{CALA})$",
            )
            ax.axhline(0.0, linewidth=1.0, linestyle="--")
            ax.set_ylabel(f"Channel {j}")
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel("Simulation time [s]")
        fig.suptitle("CALA residual disturbance correction (zoomed)")
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "04_cala_disturbance_correction.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Plots: CALA policy
    # -------------------------------------------------------------------------

    def _plot_policy_adjustment(self, policy: dict) -> None:
        t = policy["step"]
        rl_mean = policy["rl_mean"]

        if self.rl_parameter == "R_adjustment":
            r_scale = np.exp(rl_mean)
            pct = 100.0 * (r_scale - 1.0)

            n_inputs = r_scale.shape[1]

            fig, axes = plt.subplots(
                2,
                1,
                figsize=(11, 7),
                sharex=True,
            )

            for j in range(n_inputs):
                axes[0].plot(
                    t,
                    r_scale[:, j],
                    label=fr"$R_{j}/R_{{0,{j}}}$",
                )
                axes[1].plot(
                    t,
                    pct[:, j],
                    label=fr"$100(R_{j}/R_{{0,{j}}}-1)$",
                )

            axes[0].axhline(1.0, linewidth=1.0, linestyle="--")
            axes[0].set_ylabel(r"$R/R_0$")
            axes[0].set_title("Learned CALA R scaling")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].axhline(0.0, linewidth=1.0, linestyle="--")
            axes[1].set_ylabel("Change from nominal [%]")
            axes[1].set_xlabel("Simulation time [s]")
            axes[1].set_title("Same CALA adjustment on a magnified scale")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            fig.tight_layout()
            fig.savefig(
                self.output_dir / "04_cala_R_adjustment.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        elif self.rl_parameter == "d_adjustment":
            n_inputs = rl_mean.shape[1]

            fig, axes = plt.subplots(
                n_inputs,
                1,
                figsize=(11, max(3.2 * n_inputs, 4.5)),
                sharex=True,
            )
            axes = np.atleast_1d(axes)

            for j, ax in enumerate(axes):
                ax.plot(t, rl_mean[:, j])
                ax.axhline(0.0, linewidth=1.0, linestyle="--")
                ax.set_ylabel(fr"$d_{{CALA,{j}}}$")
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Simulation time [s]")
            fig.suptitle("Mean CALA disturbance residual adjustment")
            fig.tight_layout()
            fig.savefig(
                self.output_dir / "04_cala_adjustment.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        else:
            fig, ax = plt.subplots(figsize=(11, 4.5))
            for j in range(rl_mean.shape[1]):
                ax.plot(t, rl_mean[:, j], label=f"Adjustment {j}")
            ax.axhline(0.0, linewidth=1.0, linestyle="--")
            ax.set_xlabel("Simulation time [s]")
            ax.set_ylabel("CALA mean adjustment")
            ax.set_title("Learned CALA adjustment")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                self.output_dir / "04_cala_adjustment.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

    def _matching_residual_for_policy(
        self,
        policy: dict,
        dseries: dict,
    ) -> Optional[np.ndarray]:
        """
        Interpolate disturbance-estimator residual magnitude onto the policy
        timebase when both use numerical simulation time.
        """
        t_policy = np.asarray(policy.get("row_time", policy["step"]), dtype=float)
        t_d = np.asarray(dseries["step"], dtype=float)

        if len(t_policy) < 2 or len(t_d) < 2:
            return None

        residual_mag = np.linalg.norm(
            dseries["d_true"] - dseries["d_hat"],
            axis=1,
        )

        # np.interp is meaningful only over the common time interval.
        out = np.full(len(t_policy), np.nan)
        mask = (t_policy >= np.min(t_d)) & (t_policy <= np.max(t_d))

        if np.any(mask):
            out[mask] = np.interp(
                t_policy[mask],
                t_d,
                residual_mag,
            )

        return out

    def _plot_policy_group_share(
        self,
        policy: dict,
        dseries: dict,
    ) -> None:
        t = policy["step"]
        share = policy["share"]
        names = policy["group_names"]

        residual_mag = self._matching_residual_for_policy(policy, dseries)

        if residual_mag is not None and np.any(np.isfinite(residual_mag)):
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(11, 7),
                sharex=True,
                gridspec_kw={"height_ratios": [1, 2]},
            )

            axes[0].plot(t, residual_mag)
            axes[0].set_ylabel(r"$\|d_{true}-\hat d\|$")
            axes[0].set_title(
                "Estimator residual alongside CALA feature use"
            )
            axes[0].grid(True, alpha=0.3)

            share_ax = axes[1]
        else:
            fig, share_ax = plt.subplots(figsize=(11, 5.5))

        share_ax.stackplot(
            t,
            *[share[:, i] for i in range(share.shape[1])],
            labels=names,
        )
        share_ax.set_ylim(0.0, 1.0)
        share_ax.set_ylabel("Share of |policy contribution|")
        share_ax.set_xlabel("Simulation time [s]")
        share_ax.set_title(
            "Relative use of configured feature groups"
        )
        share_ax.grid(True, alpha=0.3)
        share_ax.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(
            self.output_dir / "05_policy_feature_group_share.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_policy_group_absolute(self, policy: dict) -> None:
        """
        Absolute contribution is important because a 70% share can still
        correspond to a very small total CALA adjustment.
        """
        t = policy["step"]
        group_mag = policy["group_magnitude"]
        names = policy["group_names"]

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(11, 7),
            sharex=True,
        )

        for i, name in enumerate(names):
            axes[0].plot(
                t,
                group_mag[:, i],
                label=name,
            )

        axes[0].set_ylabel("Absolute contribution")
        axes[0].set_title(
            "Absolute learned-policy contribution by feature group"
        )
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        total_adjustment = np.linalg.norm(policy["rl_mean"], axis=1)
        axes[1].plot(t, total_adjustment)
        axes[1].set_ylabel(r"$\|\mathrm{CALA\ mean}\|$")
        axes[1].set_xlabel("Simulation time [s]")
        axes[1].set_title(
            "Total magnitude of the learned CALA adjustment"
        )
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            self.output_dir / "06_policy_feature_group_absolute.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        axes[0].bar(
            names,
            100.0 * policy["mean_share"],
        )
        axes[0].set_ylabel("Mean share [%]")
        axes[0].set_title("Relative feature-group contribution")
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].bar(
            names,
            policy["mean_group_magnitude"],
        )
        axes[1].set_ylabel("Mean absolute contribution")
        axes[1].set_title("Absolute feature-group contribution")
        axes[1].grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            self.output_dir / "07_policy_feature_group_average.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Context-learning diagnostics
    # -------------------------------------------------------------------------

    def _context_learning_diagnostics(
        self,
        policy: Optional[dict],
        delta_threshold: float = 0.02,
        shuffle_seed: int = 12345,
    ) -> Optional[dict]:
        """
        Diagnose whether CALA learned a context-specific policy.

        This is deliberately different from asking whether that context is
        *useful*.  The latter still requires a controller-level ablation run.

        Structural evidence of context learning is assessed using:

        1. feature-specific parameter learning:
              mu_final - mu_first_stored

        2. heterogeneity across features within a context group:
              if every radial feature moves identically, that is much closer
              to a global shift than a spatially differentiated policy.

        3. context-driven policy variation:
              signed bias/radial/time contributions to the learned log-R
              adjustment over evaluation.

        4. shuffled-context sensitivity:
              keep the learned theta fixed, but pair each evaluation sample
              with another sample's non-bias feature vector.  This is NOT a
              performance ablation; it asks whether the learned policy output
              actually depends on matching the policy to the present context.

        The returned evidence label is a heuristic diagnostic, not a
        statistical proof of improved closed-loop performance.
        """
        if policy is None:
            return None

        if self.train_data is None or "mu" not in self.train_data:
            return None

        mu_hist = np.asarray(self.train_data["mu"], dtype=float)
        if mu_hist.ndim != 3 or len(mu_hist) < 2:
            return None

        mu_first = mu_hist[0]
        mu_final = mu_hist[-1]
        delta_mu = mu_final - mu_first

        cfg_nld = self.cfg_cala.get("cala_mpc_nld", {})
        mu_init = float(cfg_nld.get("mu_init", 0.5))
        sigma_min = float(cfg_nld.get("sigma_min", 0.0))
        sigma_max = float(cfg_nld.get("sigma_max", 1.0))

        group_learning = {}
        for name, idx in self.current_group_indices.items():
            d = delta_mu[idx, :]
            final_group = mu_final[idx, :]
            group_learning[name] = {
                "n_features": int(len(idx)),
                "mean_abs_delta_mu_by_channel": np.mean(np.abs(d), axis=0),
                "rms_delta_mu_by_channel": np.sqrt(np.mean(d**2, axis=0)),
                "std_delta_mu_across_features_by_channel": np.std(d, axis=0),
                "range_delta_mu_across_features_by_channel": (
                    np.max(d, axis=0) - np.min(d, axis=0)
                ),
                "fraction_changed_more_than_threshold": float(
                    np.mean(np.abs(d) > float(delta_threshold))
                ),
                "fraction_mu_near_0_or_1": float(
                    np.mean((final_group < 0.05) | (final_group > 0.95))
                ),
            }

        # Learning/convergence behaviour.
        mu_step = np.diff(mu_hist, axis=0)
        n_step = len(mu_step)
        late_start = max(0, int(0.8 * n_step))
        late_step = mu_step[late_start:]
        late_mu_update_rms = float(
            np.sqrt(np.mean(late_step**2))
        ) if len(late_step) else np.nan

        sigma_summary = None
        if "sigma" in self.train_data:
            sigma_hist = np.asarray(self.train_data["sigma"], dtype=float)
            if sigma_hist.ndim == 3 and len(sigma_hist):
                sigma_final = sigma_hist[-1]
                tol = 1e-6
                sigma_summary = {
                    "final_mean": float(np.mean(sigma_final)),
                    "final_mean_by_channel": np.mean(sigma_final, axis=0),
                    "fraction_at_sigma_min": float(
                        np.mean(sigma_final <= sigma_min + tol)
                    ),
                    "fraction_at_sigma_max": float(
                        np.mean(sigma_final >= sigma_max - tol)
                    ),
                }

        phi = np.asarray(policy["phi"], dtype=float)
        theta = np.asarray(policy["theta"], dtype=float)
        full = np.asarray(policy["rl_mean"], dtype=float)

        signed_group = {}
        group_variation = {}
        for name, idx in self.current_group_indices.items():
            g = phi[:, idx] @ theta[idx, :]
            signed_group[name] = g
            group_variation[name] = {
                "mean_by_channel": np.mean(g, axis=0),
                "rms_by_channel": np.sqrt(np.mean(g**2, axis=0)),
                "std_by_channel": np.std(g, axis=0),
            }

        if "bias" in signed_group:
            bias_component = signed_group["bias"]
        else:
            bias_component = np.zeros_like(full)

        context_component = full - bias_component
        full_rms = np.sqrt(np.mean(full**2, axis=0))
        context_std = np.std(context_component, axis=0)
        context_rms = np.sqrt(np.mean(context_component**2, axis=0))

        # Context-collapsed comparator.  mean_R_over_R0 is the arithmetic mean
        # physical R scaling; constant_log_R is the geometric-mean equivalent.
        r_ratio = np.exp(full)
        constant_log_R = np.mean(full, axis=0)
        constant_R_geometric = np.exp(constant_log_R)
        constant_R_arithmetic = np.mean(r_ratio, axis=0)

        # Shuffle all non-bias context together by sample.  Keeping a single
        # permutation preserves physically co-occurring radial/time features
        # from another sample rather than constructing impossible combinations.
        rng = np.random.default_rng(shuffle_seed)
        perm = rng.permutation(len(phi))
        phi_shuffled = phi.copy()
        non_bias_idx = np.concatenate(
            [
                idx
                for name, idx in self.current_group_indices.items()
                if name != "bias"
            ]
        ) if any(name != "bias" for name in self.current_group_indices) else np.array([], dtype=int)

        if len(non_bias_idx):
            phi_shuffled[:, non_bias_idx] = phi[perm][:, non_bias_idx]

        shuffled = phi_shuffled @ theta
        shuffle_change = shuffled - full
        shuffle_change_rms = np.sqrt(np.mean(shuffle_change**2, axis=0))
        shuffle_change_normalized = shuffle_change_rms / (full_rms + _EPS)

        log_R_std = np.std(full, axis=0)
        R_ratio_cv = np.std(r_ratio, axis=0) / (np.mean(r_ratio, axis=0) + _EPS)
        context_modulation_fraction = context_std / (full_rms + _EPS)

        # Heuristic structural-evidence flags.  These are intentionally simple
        # and are reported explicitly so they are not confused with hypothesis
        # tests or with proof of performance benefit.
        radial_heterogeneity = 0.0
        if "radial" in group_learning:
            radial_heterogeneity = float(
                np.max(
                    group_learning["radial"][
                        "std_delta_mu_across_features_by_channel"
                    ]
                )
            )

        flags = {
            "feature_specific_radial_learning": bool(
                radial_heterogeneity >= float(delta_threshold)
            ),
            "nontrivial_policy_context_variation": bool(
                np.max(log_R_std) >= 0.10
            ),
            "policy_sensitive_to_context_shuffle": bool(
                np.max(shuffle_change_normalized) >= 0.10
            ),
            "context_component_material_relative_to_policy": bool(
                np.max(context_modulation_fraction) >= 0.10
            ),
        }

        score = int(sum(flags.values()))
        if score >= 3:
            evidence = "strong structural evidence of context-dependent learning"
        elif score == 2:
            evidence = "moderate structural evidence of context-dependent learning"
        elif score == 1:
            evidence = "weak structural evidence of context-dependent learning"
        else:
            evidence = "little structural evidence of context-dependent learning"

        return {
            "n_learning_records": int(len(mu_hist)),
            "delta_threshold": float(delta_threshold),
            "mu_init_config": mu_init,
            "mu_first_stored": mu_first,
            "mu_final": mu_final,
            "delta_mu": delta_mu,
            "group_learning": group_learning,
            "late_mu_update_rms": late_mu_update_rms,
            "sigma": sigma_summary,
            "group_signed_contribution": signed_group,
            "group_variation": group_variation,
            "full_log_R_adjustment": full,
            "context_component": context_component,
            "context_rms_by_channel": context_rms,
            "context_std_by_channel": context_std,
            "context_modulation_fraction_by_channel": context_modulation_fraction,
            "log_R_std_by_channel": log_R_std,
            "R_ratio_cv_by_channel": R_ratio_cv,
            "constant_log_R_adjustment": constant_log_R,
            "constant_R_over_R0_geometric": constant_R_geometric,
            "constant_R_over_R0_arithmetic": constant_R_arithmetic,
            "shuffled_log_R_adjustment": shuffled,
            "shuffle_change_rms_by_channel": shuffle_change_rms,
            "shuffle_change_normalized_by_channel": shuffle_change_normalized,
            "evidence_flags": flags,
            "evidence_score_0_to_4": score,
            "structural_evidence": evidence,
            "closed_loop_context_benefit_proven": False,
            "closed_loop_note": (
                "To prove useful context, evaluate the full contextual policy "
                "against a constant learned-R ablation on identical seeds."
            ),
        }

    def _configuration_recommendations(
        self,
        policy: Optional[dict],
        context: Optional[dict],
    ) -> Optional[dict]:
        """Generate conservative, diagnostic configuration recommendations."""
        if policy is None or self.rl_parameter != "R_adjustment":
            return None

        nld = self.cfg_cala.get("cala_mpc_nld", {})
        hm = self.cfg_cala.get("horizon_manager", {})

        d_max = float(nld.get("d_max", self.d_max))
        sigma_init = float(nld.get("sigma_init", np.nan))
        sigma_max = float(nld.get("sigma_max", np.nan))
        learning_rate = float(nld.get("learning_rate", np.nan))
        r_lambda = float(hm.get("r_lambda_effort", np.nan))
        reward_period = int(hm.get("reward_period", 1))
        disturbance_gain = float(hm.get("r_disturbance_gain", np.nan))

        log_adj = np.asarray(policy["rl_mean"], dtype=float)
        r_ratio = np.exp(log_adj)
        mean_r_ratio = np.mean(r_ratio, axis=0)
        aggressive_fraction = np.mean(r_ratio < 0.5, axis=0)
        boundary_fraction = np.mean(np.abs(log_adj) > 0.8 * d_max, axis=0)

        R_diag = _find_key_recursive(self.cfg_mpc, "R_diag")
        if R_diag is not None:
            R_diag = np.asarray(R_diag, dtype=float).reshape(-1)
            n = min(len(R_diag), len(mean_r_ratio))
            candidate_constant_R = R_diag[:n] * mean_r_ratio[:n]
        else:
            candidate_constant_R = None

        # In the normalized action space, one sigma maps to this local log-R
        # displacement before feature blending.
        local_log_R_sigma = (
            2.0 * d_max * sigma_init
            if np.isfinite(sigma_init)
            else np.nan
        )

        recommendations = []

        # Effort penalty: current result is aggressively below nominal in at
        # least one channel, so this is the first knob to turn.
        if np.any(mean_r_ratio < 0.5) or np.any(aggressive_fraction > 0.5):
            lambda_grid = sorted(
                set(
                    round(v, 6)
                    for v in [
                        r_lambda,
                        max(r_lambda * 1.5, 0.015),
                        max(r_lambda * 2.0, 0.020),
                        max(r_lambda * 3.0, 0.030),
                    ]
                    if np.isfinite(v)
                )
            )
            recommendations.append(
                {
                    "parameter": "horizon_manager.r_lambda_effort",
                    "current": r_lambda,
                    "recommended_screen": lambda_grid,
                    "priority": "high",
                    "reason": (
                        "The learned policy spends substantial time below "
                        "R/R0=0.5. Increase the effort penalty before judging "
                        "whether the large negative R shift is useful context "
                        "or simply an under-penalized aggressive solution."
                    ),
                }
            )

        # d_max: large search range, but do not reduce blindly if the current
        # policy is already close to its boundary.
        dmax_rec = {
            "parameter": "cala_mpc_nld.d_max",
            "current": d_max,
            "current_R_multiplier_range": [
                float(np.exp(-d_max)),
                float(np.exp(d_max)),
            ],
            "fraction_policy_near_80pct_bound_by_channel": boundary_fraction,
            "priority": "medium",
        }
        if np.any(boundary_fraction > 0.20):
            dmax_rec.update(
                {
                    "recommended_action": (
                        "Do not reduce d_max first. The present policy is "
                        "already using the edge of the available log-R range. "
                        "First increase r_lambda_effort and/or recenter nominal "
                        "R0. Then screen d_max in [0.75, 1.0, 1.25]."
                    ),
                    "recommended_screen_after_recentering": [0.75, 1.0, 1.25],
                }
            )
        else:
            dmax_rec.update(
                {
                    "recommended_action": (
                        "The current d_max gives a very broad multiplicative R "
                        "range. Screen a narrower contextual range."
                    ),
                    "recommended_screen": [0.75, 1.0, 1.25],
                }
            )
        recommendations.append(dmax_rec)

        # Exploration scale is coupled to d_max.
        if np.isfinite(local_log_R_sigma) and local_log_R_sigma > 0.8:
            recommendations.append(
                {
                    "parameter": "cala_mpc_nld.sigma_init",
                    "current": sigma_init,
                    "current_local_1sigma_log_R_step": local_log_R_sigma,
                    "current_local_1sigma_multiplier": [
                        float(np.exp(-local_log_R_sigma)),
                        float(np.exp(local_log_R_sigma)),
                    ],
                    "recommended_if_d_max_stays_2": 0.20,
                    "recommended_if_d_max_is_1": 0.25,
                    "priority": "medium",
                    "reason": (
                        "d_max and sigma jointly set exploration amplitude. "
                        "The current one-sigma local move is large for an R "
                        "policy, making early exploration highly aggressive."
                    ),
                }
            )

        # Nominal R centering / constant learned-R ablation.
        if candidate_constant_R is not None:
            recommendations.append(
                {
                    "parameter": "config_mpc.R_diag / nominal R0",
                    "current": R_diag,
                    "constant_learned_R_candidate": candidate_constant_R,
                    "mean_R_over_R0": mean_r_ratio,
                    "priority": "high",
                    "reason": (
                        "Before claiming useful context, run a static controller "
                        "at the same average learned R. If contextual CALA beats "
                        "this constant-R baseline on paired seeds, the spatial/"
                        "temporal modulation itself is providing value."
                    ),
                }
            )

        # Learning-rate recommendation only when the training history says the
        # policy is still moving materially near the end.
        if context is not None:
            late = context.get("late_mu_update_rms", np.nan)
            if np.isfinite(late) and late > 0.02 and np.isfinite(learning_rate):
                recommendations.append(
                    {
                        "parameter": "cala_mpc_nld.learning_rate",
                        "current": learning_rate,
                        "recommended_screen": [0.15, 0.20, learning_rate],
                        "priority": "medium",
                        "reason": (
                            "The final 20% of training still has relatively "
                            "large mu updates. A slightly lower learning rate "
                            "may give a more stable contextual policy."
                        ),
                    }
                )

        return {
            "current": {
                "d_max": d_max,
                "r_lambda_effort": r_lambda,
                "sigma_init": sigma_init,
                "sigma_max": sigma_max,
                "learning_rate": learning_rate,
                "reward_period": reward_period,
                "r_disturbance_gain": disturbance_gain,
                "group_weights": dict(self.feature_map.group_weights),
            },
            "observed_policy": {
                "mean_R_over_R0": mean_r_ratio,
                "fraction_R_below_0p5_by_channel": aggressive_fraction,
                "fraction_policy_near_80pct_dmax_bound_by_channel": boundary_fraction,
            },
            "recommended_tuning_order": [
                "1. Establish the constant learned-R / static Pareto baseline.",
                "2. Increase r_lambda_effort so the learner is not rewarded mainly for aggressiveness.",
                "3. Recenter nominal R0 near the static Pareto knee.",
                "4. Then narrow d_max to make CALA learn local contextual deviations.",
                "5. Tune sigma_init jointly with d_max; only then revisit learning_rate.",
            ],
            "recommendations": recommendations,
        }

    def _plot_context_mu_learning(self, context: dict) -> None:
        delta = np.asarray(context["delta_mu"], dtype=float)
        n_features, n_inputs = delta.shape
        x = np.arange(n_features)

        fig, axes = plt.subplots(
            n_inputs,
            1,
            figsize=(12, max(3.2 * n_inputs, 4.5)),
            sharex=True,
        )
        axes = np.atleast_1d(axes)

        for j, ax in enumerate(axes):
            ax.plot(x, delta[:, j], marker="o", linewidth=1.2)
            ax.axhline(0.0, linestyle="--", linewidth=1.0)
            ax.set_ylabel(fr"$\Delta\mu_{{i,{j}}}$")
            ax.grid(True, alpha=0.3)

            # Mark feature-group boundaries and names.
            for name, idx in self.current_group_indices.items():
                if len(idx):
                    ax.axvline(idx[0] - 0.5, linewidth=0.7, alpha=0.35)
                    center = 0.5 * (idx[0] + idx[-1])
                    ax.text(
                        center,
                        0.97,
                        name,
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=9,
                    )

        axes[-1].set_xlabel("Feature index")
        fig.suptitle(
            r"Feature-specific CALA learning: final $\mu$ minus first stored $\mu$"
        )
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "12_context_feature_mu_learning.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_context_policy_decomposition(self, policy: dict, context: dict) -> None:
        t = np.asarray(policy["step"], dtype=float)
        group_signed = context["group_signed_contribution"]
        full = np.asarray(context["full_log_R_adjustment"], dtype=float)
        n_inputs = full.shape[1]

        fig, axes = plt.subplots(
            n_inputs,
            1,
            figsize=(12, max(3.5 * n_inputs, 5.0)),
            sharex=True,
        )
        axes = np.atleast_1d(axes)

        for j, ax in enumerate(axes):
            ax.plot(t, full[:, j], linewidth=2.0, label="full learned policy")
            for name, values in group_signed.items():
                ax.plot(t, values[:, j], linewidth=1.2, label=name)
            ax.axhline(0.0, linestyle="--", linewidth=1.0)
            ax.set_ylabel(fr"$\log(R_{j}/R_{{0,{j}}})$")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=2)

        axes[-1].set_xlabel("Simulation time [s]")
        fig.suptitle("Signed feature-group contribution to the learned R policy")
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "13_context_policy_decomposition.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_context_ablation_proxy(self, policy: dict, context: dict) -> None:
        """
        Policy-output ablation only.  This does not claim closed-loop benefit.
        """
        t = np.asarray(policy["step"], dtype=float)
        full = np.asarray(context["full_log_R_adjustment"], dtype=float)
        shuffled = np.asarray(context["shuffled_log_R_adjustment"], dtype=float)
        constant = np.asarray(context["constant_log_R_adjustment"], dtype=float)

        n_inputs = full.shape[1]
        fig, axes = plt.subplots(
            n_inputs,
            1,
            figsize=(12, max(3.5 * n_inputs, 5.0)),
            sharex=True,
        )
        axes = np.atleast_1d(axes)

        for j, ax in enumerate(axes):
            ax.plot(t, np.exp(full[:, j]), linewidth=2.0, label="full context")
            ax.plot(t, np.exp(shuffled[:, j]), linewidth=1.0, alpha=0.7, label="shuffled context")
            ax.axhline(
                np.exp(constant[j]),
                linestyle="--",
                linewidth=1.5,
                label="constant learned R",
            )
            ax.axhline(1.0, linestyle=":", linewidth=1.0, label="nominal R")
            ax.set_ylabel(fr"$R_{j}/R_{{0,{j}}}$")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=2)

        axes[-1].set_xlabel("Simulation time [s]")
        fig.suptitle(
            "Context dependence of policy output (not a performance ablation)"
        )
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "14_context_policy_ablation_proxy.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_context_learning_summary(self, context: dict) -> None:
        names = list(context["group_learning"].keys())
        learned = [
            np.mean(
                context["group_learning"][name][
                    "mean_abs_delta_mu_by_channel"
                ]
            )
            for name in names
        ]
        heterogeneity = [
            np.mean(
                context["group_learning"][name][
                    "std_delta_mu_across_features_by_channel"
                ]
            )
            for name in names
        ]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].bar(names, learned)
        axes[0].set_ylabel(r"Mean $|\Delta\mu|$")
        axes[0].set_title("How much each feature group learned")
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].bar(names, heterogeneity)
        axes[1].set_ylabel(r"Std of $\Delta\mu$ across features")
        axes[1].set_title("Feature-specific differentiation")
        axes[1].grid(True, axis="y", alpha=0.3)

        fig.suptitle(
            f"Context-learning diagnostic: {context['structural_evidence']}"
        )
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "15_context_learning_summary.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Plots: spatial residual / learned R alignment
    # -------------------------------------------------------------------------

    def _plot_spatial_residual_maps(self, spatial: dict) -> None:
        x_edges = spatial["x_edges"]
        y_edges = spatial["y_edges"]
        n_inputs = len(spatial["residual_abs_maps"])

        fig, axes = plt.subplots(
            1,
            n_inputs + 1,
            figsize=(5.0 * (n_inputs + 1), 4.6),
            squeeze=False,
        )
        axes = axes.ravel()

        grids = [spatial["residual_norm_map"]] + spatial["residual_abs_maps"]
        titles = [r"Mean $\|d_{true}-\hat d\|$"] + [
            fr"Mean $|d_{{true,{j}}}-\hat d_{j}|$"
            for j in range(n_inputs)
        ]

        for ax, grid, title in zip(axes, grids, titles):
            im = ax.pcolormesh(
                x_edges,
                y_edges,
                grid,
                shading="auto",
            )
            fig.colorbar(im, ax=ax)
            ax.scatter(
                self.feature_map.centers[:, 0],
                self.feature_map.centers[:, 1],
                marker="x",
                s=30,
                label="RBF centres",
            )
            ax.set_title(title)
            ax.set_xlabel("$x_0$")
            ax.set_ylabel("$x_1$")
            ax.set_xlim(self.feature_map.x_lims)
            ax.set_ylim(self.feature_map.y_lims)
            ax.set_aspect("equal")
            ax.legend(loc="best", fontsize=8)

        fig.suptitle(
            "Spatial structure remaining after disturbance estimation"
        )
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "08_spatial_estimator_residual.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_spatial_R_maps(self, spatial: dict) -> None:
        x_edges = spatial["x_edges"]
        y_edges = spatial["y_edges"]
        n_inputs = len(spatial["r_scale_maps"])

        fig, axes = plt.subplots(
            1,
            n_inputs,
            figsize=(5.2 * n_inputs, 4.6),
            squeeze=False,
        )
        axes = axes.ravel()

        for j, ax in enumerate(axes):
            im = ax.pcolormesh(
                x_edges,
                y_edges,
                spatial["r_scale_maps"][j],
                shading="auto",
            )
            fig.colorbar(im, ax=ax, label=fr"$R_{j}/R_{{0,{j}}}$")
            ax.scatter(
                self.feature_map.centers[:, 0],
                self.feature_map.centers[:, 1],
                marker="x",
                s=30,
                label="RBF centres",
            )
            ax.set_title(fr"Mean learned $R_{j}/R_{{0,{j}}}$")
            ax.set_xlabel("$x_0$")
            ax.set_ylabel("$x_1$")
            ax.set_xlim(self.feature_map.x_lims)
            ax.set_ylim(self.feature_map.y_lims)
            ax.set_aspect("equal")
            ax.legend(loc="best", fontsize=8)

        fig.suptitle("Spatial structure of the learned CALA R policy")
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "09_spatial_learned_R.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_spatial_alignment(self, spatial: dict) -> None:
        """
        Put estimator residual and CALA policy departure directly above/below
        each other for visual comparison in each input channel.
        """
        x_edges = spatial["x_edges"]
        y_edges = spatial["y_edges"]
        n_inputs = len(spatial["r_scale_maps"])

        fig, axes = plt.subplots(
            2,
            n_inputs,
            figsize=(5.2 * n_inputs, 8.2),
            squeeze=False,
        )

        for j in range(n_inputs):
            residual_map = spatial["residual_abs_maps"][j]
            policy_map = spatial["log_r_deviation_maps"][j]

            ax = axes[0, j]
            im = ax.pcolormesh(
                x_edges,
                y_edges,
                residual_map,
                shading="auto",
            )
            fig.colorbar(
                im,
                ax=ax,
                label=fr"mean $|d_{{true,{j}}}-\hat d_{j}|$",
            )
            ax.scatter(
                self.feature_map.centers[:, 0],
                self.feature_map.centers[:, 1],
                marker="x",
                s=25,
            )
            ax.set_title(fr"Estimator residual: channel {j}")
            ax.set_xlabel("$x_0$")
            ax.set_ylabel("$x_1$")
            ax.set_xlim(self.feature_map.x_lims)
            ax.set_ylim(self.feature_map.y_lims)
            ax.set_aspect("equal")

            corr = spatial["binned_spatial_correlations"][
                f"channel_{j}"
            ]["mean_abs_residual_vs_mean_abs_log_R_ratio"]

            ax = axes[1, j]
            im = ax.pcolormesh(
                x_edges,
                y_edges,
                policy_map,
                shading="auto",
            )
            fig.colorbar(
                im,
                ax=ax,
                label=fr"mean $|\log(R_{j}/R_{{0,{j}}})|$",
            )
            ax.scatter(
                self.feature_map.centers[:, 0],
                self.feature_map.centers[:, 1],
                marker="x",
                s=25,
            )
            ax.set_title(
                fr"CALA departure from nominal: channel {j}"
                + "\n"
                + fr"occupied-bin Spearman $\rho$="
                + (
                    f"{corr['spearman']:.3f}"
                    if np.isfinite(corr["spearman"])
                    else "nan"
                )
            )
            ax.set_xlabel("$x_0$")
            ax.set_ylabel("$x_1$")
            ax.set_xlim(self.feature_map.x_lims)
            ax.set_ylim(self.feature_map.y_lims)
            ax.set_aspect("equal")

        fig.suptitle(
            "Do estimator-residual regions align with CALA R adaptation?"
        )
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "10_spatial_residual_R_alignment.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_residual_vs_R_scatter(self, spatial: dict) -> None:
        n_inputs = spatial["r_scale"].shape[1]

        fig, axes = plt.subplots(
            1,
            n_inputs + 1,
            figsize=(5.0 * (n_inputs + 1), 4.5),
            squeeze=False,
        )
        axes = axes.ravel()

        for j in range(n_inputs):
            x = np.abs(spatial["residual"][:, j])
            y = spatial["r_scale"][:, j]

            corr = spatial["sample_correlations"][
                f"channel_{j}"
            ]["abs_residual_vs_R_over_R0"]

            axes[j].scatter(x, y, s=12, alpha=0.45)
            axes[j].axhline(1.0, linestyle="--", linewidth=1.0)
            axes[j].set_xlabel(
                fr"$|d_{{true,{j}}}-\hat d_{j}|$"
            )
            axes[j].set_ylabel(fr"$R_{j}/R_{{0,{j}}}$")
            axes[j].set_title(
                fr"Channel {j}: Spearman $\rho$="
                + (
                    f"{corr['spearman']:.3f}"
                    if np.isfinite(corr["spearman"])
                    else "nan"
                )
            )
            axes[j].grid(True, alpha=0.3)

        total_corr = spatial["sample_correlations"][
            "total"
        ]["residual_norm_vs_policy_log_norm"]

        axes[-1].scatter(
            spatial["residual_norm"],
            spatial["policy_log_norm"],
            s=12,
            alpha=0.45,
        )
        axes[-1].set_xlabel(r"$\|d_{true}-\hat d\|$")
        axes[-1].set_ylabel(
            r"$\|\log(R/R_0)\|$"
        )
        axes[-1].set_title(
            r"Total: Spearman $\rho$="
            + (
                f"{total_corr['spearman']:.3f}"
                if np.isfinite(total_corr["spearman"])
                else "nan"
            )
        )
        axes[-1].grid(True, alpha=0.3)

        fig.suptitle(
            "Estimator residual versus learned CALA R response"
        )
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "11_residual_vs_R_scatter.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Plots: representation of CALA learning signal
    # -------------------------------------------------------------------------

    def _plot_performance_representation(self, perf: dict) -> None:
        target_names = list(perf["targets"].keys())

        fig, axes = plt.subplots(
            1,
            len(target_names),
            figsize=(5.3 * len(target_names), 4.5),
            squeeze=False,
        )
        axes = axes.ravel()

        for ax, target_name in zip(axes, target_names):
            result = perf["targets"][target_name]
            labels = ["Current", "+ space×time"]
            values = [
                result["current"]["rmse"],
                result["augmented"]["rmse"],
            ]

            ax.bar(labels, values)
            ax.set_ylabel("Blocked-CV RMSE")
            ax.set_title(
                f"{target_name.capitalize()} representation\n"
                f"space×time gain = "
                f"{result['space_time_rmse_improvement_pct']:.1f}%"
            )
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle(
            "Does space×time improve representation of CALA performance?"
        )
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "16_performance_representation_comparison.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Out-of-fold time-series, one figure per target to avoid scale overlap.
        for target_name, result in perf["targets"].items():
            fig, ax = plt.subplots(figsize=(11, 4.8))
            t = perf["step"]

            ax.plot(
                t,
                result["actual"],
                linewidth=1.4,
                label=f"Actual {target_name}",
            )
            ax.plot(
                t,
                result["pred_current"],
                linewidth=1.2,
                label="Current features",
            )
            ax.plot(
                t,
                result["pred_augmented"],
                linewidth=1.2,
                label="Current + space×time",
            )

            ax.set_xlabel("Trial start time [s]")
            ax.set_ylabel(target_name.capitalize())
            ax.set_title(
                f"Out-of-fold {target_name} prediction from context + CALA action"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()

            fig.tight_layout()
            fig.savefig(
                self.output_dir
                / f"17_{target_name}_out_of_fold_prediction.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

    def _plot_context_group_value(self, perf: dict) -> None:
        ablation = perf["current_group_ablation_rmse_change_pct"]
        target = perf["group_value_target"]

        labels = list(ablation.keys()) + ["space×time\n(addition)"]
        values = list(ablation.values()) + [
            perf["space_time_addition_rmse_improvement_pct"]
        ]

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(labels, values)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_ylabel("Blocked-CV RMSE importance / improvement [%]")
        ax.set_title(
            f"Feature-group value for CALA {target} representation"
        )
        ax.grid(True, axis="y", alpha=0.3)

        note = (
            "Existing group: positive = prediction worsens when removed.  "
            "space×time: positive = prediction improves when added."
        )
        fig.text(0.5, 0.01, note, ha="center", fontsize=9)

        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(
            self.output_dir / "18_context_feature_group_value.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Master
    # -------------------------------------------------------------------------

    def run(
        self,
        make_plots: bool = True,
        n_folds: int = 5,
        ridge_alpha: float = 1.0,
        spatial_bins: int = 10,
        min_bin_count: int = 3,
        warmup_seconds: float = 1.0,
        run_reward_representation: bool = False,
        context_delta_threshold: float = 0.02,
    ) -> dict:

        dseries = self._extract_disturbance_series()

        estimator_score = _score(
            dseries["d_true"],
            dseries["d_hat"],
        )

        residual = dseries["d_true"] - dseries["d_hat"]

        # Also report estimator performance after a short startup interval so
        # observer initialization does not dominate the summary statistic.
        d_time = np.asarray(dseries["step"], dtype=float).reshape(-1)
        if len(d_time) > 0:
            warmup_mask = d_time >= (d_time[0] + float(warmup_seconds))
        else:
            warmup_mask = np.ones(len(residual), dtype=bool)

        if np.count_nonzero(warmup_mask) >= 3:
            estimator_score_after_warmup = _score(
                dseries["d_true"][warmup_mask],
                dseries["d_hat"][warmup_mask],
            )
        else:
            estimator_score_after_warmup = None

        metrics = {
            "configs_base": str(self.configs_base),
            "evaluation_path": (
                str(self.evaluation_path)
                if self.evaluation_path is not None
                else None
            ),
            "training_path": (
                str(self.training_path)
                if self.training_path is not None
                else None
            ),
            "evaluation_group": self.evaluation_group,
            "training_group": self.training_group,
            "rl_parameter": self.rl_parameter,
            "reward_mode": self.reward_mode,
            "group_weights": dict(self.feature_map.group_weights),
            "disturbance_series_source": dseries["source"],
            "disturbance_estimator": estimator_score,
            "estimator_warmup_seconds": float(warmup_seconds),
            "disturbance_estimator_after_warmup": estimator_score_after_warmup,
            "disturbance_residual_rms": float(
                np.sqrt(np.mean(residual**2))
            ),
            "disturbance_residual_rms_per_channel": np.sqrt(
                np.mean(residual**2, axis=0)
            ),
        }

        # If CALA is itself a disturbance correction, quantify incremental
        # residual improvement. Never do this for R learning.
        if (
            self.rl_parameter == "d_adjustment"
            and dseries["rl_mean"] is not None
        ):
            n = min(
                len(dseries["d_true"]),
                len(dseries["d_hat"]),
                len(dseries["rl_mean"]),
            )
            d_true = dseries["d_true"][:n]
            d_eff = (
                dseries["d_hat"][:n]
                + dseries["rl_mean"][:n]
            )

            cala_corrected_score = _score(d_true, d_eff)

            before = estimator_score["rmse"]
            after = cala_corrected_score["rmse"]

            metrics["cala_disturbance_correction"] = {
                "corrected_estimate": cala_corrected_score,
                "rmse_improvement_pct": float(
                    100.0 * (before - after) / (before + _EPS)
                ),
            }
        else:
            metrics["cala_disturbance_correction"] = None

        policy = self._policy_series()

        if policy is not None:
            metrics["learned_policy"] = {
                "source": policy["source"],
                "mean_adjustment": np.mean(
                    policy["rl_mean"],
                    axis=0,
                ),
                "rms_adjustment": np.sqrt(
                    np.mean(policy["rl_mean"] ** 2, axis=0)
                ),
                "mean_group_share": {
                    name: float(value)
                    for name, value in zip(
                        policy["group_names"],
                        policy["mean_share"],
                    )
                },
                "mean_group_absolute_contribution": {
                    name: float(value)
                    for name, value in zip(
                        policy["group_names"],
                        policy["mean_group_magnitude"],
                    )
                },
            }

            if self.rl_parameter == "R_adjustment":
                r_scale = np.exp(policy["rl_mean"])
                metrics["learned_policy"]["mean_R_over_R0"] = np.mean(
                    r_scale,
                    axis=0,
                )
                metrics["learned_policy"]["rms_percent_R_change"] = np.sqrt(
                    np.mean((100.0 * (r_scale - 1.0)) ** 2, axis=0)
                )
        else:
            metrics["learned_policy"] = None

        context = self._context_learning_diagnostics(
            policy=policy,
            delta_threshold=context_delta_threshold,
        )

        if context is not None:
            metrics["context_learning"] = {
                "n_learning_records": context["n_learning_records"],
                "delta_threshold": context["delta_threshold"],
                "group_learning": context["group_learning"],
                "late_mu_update_rms": context["late_mu_update_rms"],
                "sigma": context["sigma"],
                "group_variation": context["group_variation"],
                "context_rms_by_channel": context["context_rms_by_channel"],
                "context_std_by_channel": context["context_std_by_channel"],
                "context_modulation_fraction_by_channel": context["context_modulation_fraction_by_channel"],
                "log_R_std_by_channel": context["log_R_std_by_channel"],
                "R_ratio_cv_by_channel": context["R_ratio_cv_by_channel"],
                "constant_R_over_R0_geometric": context["constant_R_over_R0_geometric"],
                "constant_R_over_R0_arithmetic": context["constant_R_over_R0_arithmetic"],
                "shuffle_change_rms_by_channel": context["shuffle_change_rms_by_channel"],
                "shuffle_change_normalized_by_channel": context["shuffle_change_normalized_by_channel"],
                "evidence_flags": context["evidence_flags"],
                "evidence_score_0_to_4": context["evidence_score_0_to_4"],
                "structural_evidence": context["structural_evidence"],
                "closed_loop_context_benefit_proven": context["closed_loop_context_benefit_proven"],
                "closed_loop_note": context["closed_loop_note"],
            }
        else:
            metrics["context_learning"] = None

        recommendations = self._configuration_recommendations(
            policy=policy,
            context=context,
        )
        metrics["configuration_recommendations"] = recommendations

        # Always save a compact ablation candidate when R learning is active.
        if recommendations is not None:
            with (self.output_dir / "configuration_recommendations.json").open("w") as f:
                json.dump(_jsonable(recommendations), f, indent=2)

        if context is not None and _find_key_recursive(self.cfg_mpc, "R_diag") is not None:
            R_diag_now = np.asarray(_find_key_recursive(self.cfg_mpc, "R_diag"), dtype=float).reshape(-1)
            scales = np.asarray(context["constant_R_over_R0_arithmetic"], dtype=float).reshape(-1)
            n_const = min(len(R_diag_now), len(scales))
            ablation = {
                "purpose": "Constant learned-R baseline for testing whether context modulation itself adds value.",
                "current_R_diag": R_diag_now[:n_const],
                "mean_learned_R_scale": scales[:n_const],
                "candidate_constant_R_diag": R_diag_now[:n_const] * scales[:n_const],
                "how_to_use": (
                    "Disable CALA during evaluation, set config_mpc.R_diag to candidate_constant_R_diag, "
                    "and rerun the exact same evaluation seeds/targets. Compare tracking and effort with "
                    "the full contextual learned policy."
                ),
            }
            with (self.output_dir / "context_ablation_candidate.json").open("w") as f:
                json.dump(_jsonable(ablation), f, indent=2)

        spatial = self._spatial_alignment(
            policy=policy,
            dseries=dseries,
            spatial_bins=spatial_bins,
            min_bin_count=min_bin_count,
            warmup_seconds=warmup_seconds,
        )

        if spatial is not None:
            metrics["spatial_residual_policy_alignment"] = {
                "warmup_seconds": spatial["warmup_seconds"],
                "n_samples": spatial["n_samples"],
                "spatial_bins": spatial["spatial_bins"],
                "min_bin_count": spatial["min_bin_count"],
                "sample_correlations": spatial["sample_correlations"],
                "binned_spatial_correlations": spatial[
                    "binned_spatial_correlations"
                ],
            }
        else:
            metrics["spatial_residual_policy_alignment"] = None

        perf = None
        if run_reward_representation:
            try:
                perf = self._performance_representation(
                    n_folds=n_folds,
                    ridge_alpha=ridge_alpha,
                )
            except (ValueError, KeyError) as exc:
                print(
                    "FeaturePerformanceAnalyzer: skipping performance "
                    f"representation test: {exc}"
                )

        if perf is not None:
            metrics["performance_representation"] = {
                "n_current_context_features": perf[
                    "n_current_context_features"
                ],
                "n_product_context_features": perf[
                    "n_product_context_features"
                ],
                "n_augmented_context_features": perf[
                    "n_augmented_context_features"
                ],
                "n_folds": int(n_folds),
                "ridge_alpha": float(ridge_alpha),
                "targets": {
                    name: {
                        "current": result["current"],
                        "augmented": result["augmented"],
                        "space_time_rmse_improvement_pct": result[
                            "space_time_rmse_improvement_pct"
                        ],
                    }
                    for name, result in perf["targets"].items()
                },
                "group_value_target": perf["group_value_target"],
                "current_group_ablation_rmse_change_pct": perf[
                    "current_group_ablation_rmse_change_pct"
                ],
                "space_time_addition_rmse_improvement_pct": perf[
                    "space_time_addition_rmse_improvement_pct"
                ],
            }
        else:
            metrics["performance_representation"] = None

        with (
            self.output_dir / "feature_performance_metrics.json"
        ).open("w") as f:
            json.dump(_jsonable(metrics), f, indent=2)

        if make_plots:
            self._plot_disturbance_estimate(dseries)
            self._plot_estimator_residual(dseries)
            self._plot_estimator_magnitude(dseries)
            self._plot_d_adjustment_if_applicable(dseries)

            if policy is not None:
                self._plot_policy_adjustment(policy)
                self._plot_policy_group_share(policy, dseries)
                self._plot_policy_group_absolute(policy)

            if context is not None and policy is not None:
                self._plot_context_mu_learning(context)
                self._plot_context_policy_decomposition(policy, context)
                self._plot_context_ablation_proxy(policy, context)
                self._plot_context_learning_summary(context)

            if spatial is not None:
                self._plot_spatial_residual_maps(spatial)
                self._plot_spatial_R_maps(spatial)
                self._plot_spatial_alignment(spatial)
                self._plot_residual_vs_R_scatter(spatial)

            if perf is not None:
                self._plot_performance_representation(perf)
                self._plot_context_group_value(perf)

        self._print_summary(metrics)

        return metrics

    def _print_summary(self, metrics: dict) -> None:
        est = metrics["disturbance_estimator"]

        print()
        print("=" * 72)
        print("FEATURE / DISTURBANCE PERFORMANCE")
        print("=" * 72)
        print(f"RL parameter:               {self.rl_parameter}")
        print(f"Reward mode:                {self.reward_mode}")
        print(
            f"Disturbance data source:    "
            f"{metrics['disturbance_series_source']}"
        )
        print(f"Estimator RMSE:              {est['rmse']:.6f}")
        print(f"Estimator normalized RMSE:   {est['nrmse']:.6f}")
        print(f"Estimator R^2:               {est['r2']:.6f}")
        print(
            f"RMS fraction captured:       "
            f"{100.0 * est['rms_fraction_captured']:.2f}%"
        )

        est_warm = metrics.get("disturbance_estimator_after_warmup")
        if est_warm is not None:
            print(
                f"After {metrics['estimator_warmup_seconds']:.2f}s warm-up:"
            )
            print(
                f"  estimator RMSE:            {est_warm['rmse']:.6f}"
            )
            print(
                f"  estimator R^2:             {est_warm['r2']:.6f}"
            )
            print(
                f"  RMS fraction captured:     "
                f"{100.0 * est_warm['rms_fraction_captured']:.2f}%"
            )

        if metrics["cala_disturbance_correction"] is not None:
            c = metrics["cala_disturbance_correction"]
            print(
                f"CALA residual RMSE gain:      "
                f"{c['rmse_improvement_pct']:.2f}%"
            )

        policy = metrics["learned_policy"]
        if policy is not None:
            print()
            print("Learned CALA policy:")
            for name, share in policy["mean_group_share"].items():
                absolute = policy[
                    "mean_group_absolute_contribution"
                ][name]
                print(
                    f"  {name:>10s}: "
                    f"{100.0 * share:6.2f}% share, "
                    f"mean |contribution|={absolute:.6f}"
                )

            if "mean_R_over_R0" in policy:
                print(
                    "  mean R/R0: "
                    + np.array2string(
                        np.asarray(policy["mean_R_over_R0"]),
                        precision=4,
                    )
                )
                print(
                    "  RMS R change [%]: "
                    + np.array2string(
                        np.asarray(
                            policy["rms_percent_R_change"]
                        ),
                        precision=3,
                    )
                )

        context_summary = metrics.get("context_learning")
        if context_summary is not None:
            print()
            print("Context-learning diagnostics:")
            print(
                f"  evidence: {context_summary['structural_evidence']} "
                f"({context_summary['evidence_score_0_to_4']}/4 flags)"
            )
            print(
                "  log-R std by channel: "
                + np.array2string(
                    np.asarray(context_summary["log_R_std_by_channel"]),
                    precision=4,
                )
            )
            print(
                "  context modulation fraction: "
                + np.array2string(
                    np.asarray(context_summary["context_modulation_fraction_by_channel"]),
                    precision=4,
                )
            )
            print(
                "  shuffled-context policy change / full RMS: "
                + np.array2string(
                    np.asarray(context_summary["shuffle_change_normalized_by_channel"]),
                    precision=4,
                )
            )
            print(
                "  NOTE: structural context learning does not by itself prove "
                "closed-loop performance benefit."
            )

        rec_summary = metrics.get("configuration_recommendations")
        if rec_summary is not None:
            print()
            print("Configuration recommendations:")
            for rec in rec_summary["recommendations"]:
                print(
                    f"  [{rec.get('priority', 'info')}] "
                    f"{rec['parameter']}: {rec.get('reason', rec.get('recommended_action', 'see JSON'))}"
                )

        spatial_summary = metrics.get("spatial_residual_policy_alignment")
        if spatial_summary is not None:
            print()
            print("Spatial estimator-residual / R-policy alignment:")
            for key, value in spatial_summary[
                "sample_correlations"
            ].items():
                if key == "total":
                    corr = value["residual_norm_vs_policy_log_norm"]
                    print(
                        f"  total residual vs |log R shift|: "
                        f"Spearman={corr['spearman']:.3f}, "
                        f"Pearson={corr['pearson']:.3f}"
                    )
                else:
                    corr = value["abs_residual_vs_R_over_R0"]
                    print(
                        f"  {key}: |residual| vs R/R0: "
                        f"Spearman={corr['spearman']:.3f}, "
                        f"Pearson={corr['pearson']:.3f}"
                    )

            print("  occupied spatial-bin correlations:")
            for key, value in spatial_summary[
                "binned_spatial_correlations"
            ].items():
                if key == "total":
                    corr = value[
                        "mean_residual_norm_vs_mean_policy_log_norm"
                    ]
                    print(
                        f"    total: Spearman={corr['spearman']:.3f}, "
                        f"Pearson={corr['pearson']:.3f}"
                    )
                else:
                    corr = value[
                        "mean_abs_residual_vs_mean_R_over_R0"
                    ]
                    print(
                        f"    {key}: Spearman={corr['spearman']:.3f}, "
                        f"Pearson={corr['pearson']:.3f}"
                    )

        perf = metrics["performance_representation"]
        if perf is not None:
            print()
            print("Performance representation:")
            for target, result in perf["targets"].items():
                print(
                    f"  {target:>10s}: "
                    f"current RMSE={result['current']['rmse']:.6f}, "
                    f"+space×time RMSE={result['augmented']['rmse']:.6f}, "
                    f"gain={result['space_time_rmse_improvement_pct']:.2f}%"
                )

        print()
        print(f"Plots / metrics saved to: {self.output_dir}")
        print("=" * 72)


# =============================================================================
# Convenience function for master scripts
# =============================================================================

def analyze_feature_performance(
    configs_base: str | Path,
    evaluation_path: Optional[str | Path] = None,
    training_path: Optional[str | Path] = None,
    output_dir: str | Path = "visualization/feature_performance",
    evaluation_group: str = "rl_learned_R",
    training_group: str = "learning",
    make_plots: bool = True,
    spatial_bins: int = 10,
    min_bin_count: int = 3,
    warmup_seconds: float = 1.0,
    run_reward_representation: bool = False,
    n_folds: int = 5,
    ridge_alpha: float = 1.0,
    context_delta_threshold: float = 0.02,
) -> dict:
    """
    Run the complete feature-performance analysis.

    Typical use
    -----------
    metrics = analyze_feature_performance(
        configs_base="configs/generated/R0_validation/R_x1p00/trial_000",
        output_dir="visualization/feature_performance/R_x1p00/trial_000",
    )
    """

    analyzer = FeaturePerformanceAnalyzer(
        configs_base=configs_base,
        evaluation_path=evaluation_path,
        training_path=training_path,
        output_dir=output_dir,
        evaluation_group=evaluation_group,
        training_group=training_group,
    )

    return analyzer.run(
        make_plots=make_plots,
        n_folds=n_folds,
        ridge_alpha=ridge_alpha,
        spatial_bins=spatial_bins,
        min_bin_count=min_bin_count,
        warmup_seconds=warmup_seconds,
        run_reward_representation=run_reward_representation,
        context_delta_threshold=context_delta_threshold,
    )