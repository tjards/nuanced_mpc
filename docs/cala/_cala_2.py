"""
cala_horizon.py

Independent CALA horizon learner for residual disturbance correction.

This module is designed to sit around your existing MPC implementation.
It does not use the true nonlinear disturbance model. In particular, it does
not read vortex centres, vortex sigmas, vortex amplitudes, or the true field
value d(p,t).

Intended architecture
---------------------
Your MPC already estimates a local input-channel disturbance:

    d_hat[k]

from one-step prediction error. CALA learns an additive correction:

    d_eff[k] = d_hat[k] + d_cala[k]

where d_cala is sampled from feature-conditioned continuous action
distributions and held fixed for one MPC horizon.

Learning mode
-------------
This implements low-frequency non-overlapping trials:

    one CALA trial = one MPC horizon = h timesteps

At trial start:
    1. Compute independent features phi(x, t).
    2. Sample d_cala.
    3. MPC solves using d_hat + d_cala.
    4. Save the trial-start predicted horizon.

During the trial:
    5. Keep d_cala fixed for h controller steps.
    6. Record the actual closed-loop states.

At trial end:
    7. Compare the trial-start prediction to the actual closed-loop states.
    8. Use negative horizon prediction error as the reward.
    9. Update CALA action means and variances.

Frame convention
----------------
If your MPC solves in target-relative coordinates:

    controller.solve(x - xr, u, ...)

then the CALA reward should also use target-relative states:

    predicted = controller.result_state_sequence.reshape(h, nx)
    actual    = x - xr

The feature map should usually use the global state x, because the nonlinear
field is spatial in the world frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """
    Configuration for an independent RBF feature map.

    These feature centres are learner basis centres, not physical disturbance
    centres. They should be chosen without using knowledge of the true nonlinear
    disturbance generator.
    """

    # Workspace used to auto-generate a grid of feature centres when
    # feature_centers is None.
    xlim: tuple[float, float] = (-5.0, 5.0)
    ylim: tuple[float, float] = (-5.0, 5.0)

    # Number of grid centres in each direction.
    n_x: int = 5
    n_y: int = 5

    # Optional explicit feature centres. If None, a regular grid is used.
    feature_centers: np.ndarray | None = None

    # RBF width. If None, it is chosen from the grid spacing.
    sigma: float | None = None

    # A constant feature allows CALA to learn a workspace-wide correction.
    include_bias: bool = True

    # Time features allow periodic corrections without using the true field.
    include_time_features: bool = True
    omega: float = 0.2

    # Normalizing makes phi behave like blending weights.
    normalize: bool = True


@dataclass
class CALAConfig:
    """
    Configuration for the continuous-action learner.
    """

    # Maximum magnitude of CALA's disturbance correction.
    #
    # Normalized action a in [0, 1] maps to:
    #
    #     delta_d = 2 * d_corr_max * (a - 0.5)
    #
    # Therefore:
    #     a = 0.0 -> -d_corr_max
    #     a = 0.5 ->  0
    #     a = 1.0 -> +d_corr_max
    d_corr_max: float = 0.50

    # Initial action distribution.
    mu_init: float = 0.5
    sigma_init: float = 0.25

    # Exploration limits.
    sigma_min: float = 0.03
    sigma_max: float = 0.60

    # CALA update rates.
    lr_mu: float = 0.12
    lr_sigma: float = 0.06

    # Running reward baseline update.
    reward_beta: float = 0.05

    # Advantage scaling before tanh bounding.
    advantage_gain: float = 2.0

    # Reproducibility.
    seed: int | None = 1


@dataclass
class HorizonRewardConfig:
    """
    Configuration for the horizon prediction-error reward.
    """

    # If None, defaults to [1, 1, 0.25, 0.25] for nx=4.
    state_weights: np.ndarray | None = None

    # Discount within the horizon. Use 1.0 for no discount.
    discount: float = 0.98

    # Additional weight on the final prediction error.
    terminal_weight: float = 1.0

    # Penalizes large learned disturbance corrections.
    effort_weight: float = 0.02


@dataclass
class CALAStepInfo:
    """
    Stores one sampled CALA action for delayed reward update.
    """

    phi: np.ndarray
    action_normalized: np.ndarray
    local_delta_d: np.ndarray
    d_cala: np.ndarray


@dataclass
class CALATrialResult:
    """
    Diagnostics returned when a horizon trial completes.
    """

    reward: float
    advantage: float
    cost: float
    prediction_error: float
    terminal_error: float
    effort: float
    d_cala: np.ndarray
    exploration: float
    start_time: float | None


# -----------------------------------------------------------------------------
# Independent nonlinear feature map
# -----------------------------------------------------------------------------

class RBFFeatureMap:
    """
    Independent nonlinear feature map for CALA.

    Each RBF centre defines a local learning region. CALA maintains one local
    action distribution per feature and per input dimension.

    Important distinction:
        - RBF feature centres do not create the true disturbance.
        - They do not need to match the disturbance/vortex centres.
        - They decide which CALA local residual actions are active.

    The features are:
        [global_bias, spatial RBFs..., optional sin_time_gate, cos_time_gate]

    The output phi is normalized by default, so the final CALA correction is a
    weighted blend of local action samples.
    """

    def __init__(self, cfg: FeatureConfig | None = None):
        self.cfg = cfg or FeatureConfig()

        if self.cfg.feature_centers is None:
            xs = np.linspace(self.cfg.xlim[0], self.cfg.xlim[1], self.cfg.n_x)
            ys = np.linspace(self.cfg.ylim[0], self.cfg.ylim[1], self.cfg.n_y)
            X, Y = np.meshgrid(xs, ys)
            self.centres = np.column_stack([X.reshape(-1), Y.reshape(-1)])
        else:
            self.centres = np.asarray(self.cfg.feature_centers, dtype=float)

        if self.centres.ndim != 2 or self.centres.shape[1] != 2:
            raise ValueError("feature_centers must have shape (n_centres, 2).")

        if self.cfg.sigma is None:
            # Choose a default width from the grid spacing, independent of the
            # disturbance model.
            dx = (self.cfg.xlim[1] - self.cfg.xlim[0]) / max(self.cfg.n_x - 1, 1)
            dy = (self.cfg.ylim[1] - self.cfg.ylim[0]) / max(self.cfg.n_y - 1, 1)
            self.sigma = 1.25 * max(dx, dy)
        else:
            self.sigma = float(self.cfg.sigma)

        self.names: list[str] = []

        if self.cfg.include_bias:
            self.names.append("global_bias")

        self.names += [
            f"rbf_{i:02d}_at_({c[0]:.2f},{c[1]:.2f})"
            for i, c in enumerate(self.centres)
        ]

        if self.cfg.include_time_features:
            self.names += ["sin_time_gate", "cos_time_gate"]

    @property
    def n_features(self) -> int:
        return len(self.names)

    def __call__(self, x_or_p: np.ndarray, t: float) -> np.ndarray:
        """
        Compute feature activations.

        Parameters
        ----------
        x_or_p:
            Either a full state [px, py, ...] or a position [px, py].
            Only the first two entries are used.

        t:
            Current simulation time.

        Returns
        -------
        phi:
            Feature vector of length n_features.
        """

        arr = np.asarray(x_or_p, dtype=float).reshape(-1)
        p = arr[:2]

        values: list[float] = []

        # Global feature lets CALA learn a workspace-wide offset.
        if self.cfg.include_bias:
            values.append(1.0)

        # Generic spatial RBF features. These are learner basis functions only.
        for c in self.centres:
            rel = p - c
            values.append(float(np.exp(-np.dot(rel, rel) / (self.sigma**2))))

        # Generic time gates. Kept nonnegative so phi can be normalized as
        # blending weights.
        if self.cfg.include_time_features:
            values.append(0.5 * (1.0 + np.sin(self.cfg.omega * t)))
            values.append(0.5 * (1.0 + np.cos(self.cfg.omega * t)))

        phi = np.asarray(values, dtype=float)

        if self.cfg.normalize:
            phi = phi / (np.sum(phi) + 1e-12)

        return phi

    def activation_grid(
        self,
        feature_index: int,
        xlim=(-5.0, 5.0),
        ylim=(-5.0, 5.0),
        n: int = 70,
        t: float = 0.0,
    ):
        """
        Evaluate a single feature activation over a grid.

        Useful for visualizing where each CALA local learner is active.
        """

        if feature_index < 0 or feature_index >= self.n_features:
            raise IndexError(
                f"feature_index must be in [0, {self.n_features - 1}], "
                f"got {feature_index}."
            )

        xs = np.linspace(xlim[0], xlim[1], n)
        ys = np.linspace(ylim[0], ylim[1], n)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)

        for row in range(n):
            for col in range(n):
                phi = self(np.array([X[row, col], Y[row, col]]), t)
                Z[row, col] = phi[feature_index]

        return X, Y, Z


# -----------------------------------------------------------------------------
# CALA residual compensator
# -----------------------------------------------------------------------------

class CALAResidualCompensator:
    """
    Continuous-action learner for residual disturbance correction.

    Each feature i and input dimension j has a continuous-action distribution:

        a_ij ~ Normal(mu_ij, sigma_ij^2), clipped to [0, 1]

    The normalized action is mapped to a physical disturbance correction:

        delta_d_ij = 2 * d_corr_max * (a_ij - 0.5)

    The final correction is the feature-weighted blend:

        d_cala = sum_i phi_i * delta_d_i

    In the MPC integration, use:

        d_eff = d_hat + d_cala

    and keep d_hat and d_cala separate.
    """

    def __init__(self, n_features: int, nu: int = 2, cfg: CALAConfig | None = None):
        self.cfg = cfg or CALAConfig()
        self.n_features = int(n_features)
        self.nu = int(nu)

        self.mu = np.full((self.n_features, self.nu), self.cfg.mu_init, dtype=float)
        self.sigma = np.full((self.n_features, self.nu), self.cfg.sigma_init, dtype=float)

        self.rng = np.random.default_rng(self.cfg.seed)
        self.reward_bar: float | None = None
        self.last: CALAStepInfo | None = None

    def sample(self, phi: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Sample a CALA disturbance correction.

        Parameters
        ----------
        phi:
            Feature vector from RBFFeatureMap.

        explore:
            If True, sample from Normal(mu, sigma).
            If False, use mu deterministically.

        Returns
        -------
        d_cala:
            Disturbance correction with shape (nu,).
        """

        phi = np.asarray(phi, dtype=float).reshape(-1)

        if phi.shape[0] != self.n_features:
            raise ValueError(f"Expected phi length {self.n_features}, got {phi.shape[0]}.")

        if explore:
            action_normalized = self.rng.normal(self.mu, self.sigma)
        else:
            action_normalized = self.mu.copy()

        action_normalized = np.clip(action_normalized, 0.0, 1.0)

        local_delta_d = 2.0 * self.cfg.d_corr_max * (action_normalized - 0.5)
        d_cala = phi @ local_delta_d

        self.last = CALAStepInfo(
            phi=phi.copy(),
            action_normalized=action_normalized.copy(),
            local_delta_d=local_delta_d.copy(),
            d_cala=d_cala.copy(),
        )

        return d_cala

    def update(self, reward: float) -> float:
        """
        Reward/penalty update.

        Positive advantage:
            sampled action was better than recent average, so move mu toward it
            and reduce variance.

        Negative advantage:
            sampled action was worse than recent average, so move mu away from it
            and increase variance.

        Feature activation controls credit assignment.
        """

        if self.last is None:
            return 0.0

        if self.reward_bar is None:
            self.reward_bar = float(reward)
            return 0.0

        raw_advantage = float(reward - self.reward_bar)
        self.reward_bar += self.cfg.reward_beta * raw_advantage

        # Rewards are usually negative costs, so scale relative to reward
        # magnitude before bounding the update pressure.
        denom = abs(self.reward_bar) + 1e-9
        scaled_advantage = raw_advantage / denom
        adv = float(np.tanh(self.cfg.advantage_gain * scaled_advantage))

        phi = self.last.phi
        action = self.last.action_normalized

        for i in range(self.n_features):
            activation = float(phi[i])
            if activation < 1e-9:
                continue

            for j in range(self.nu):
                sampled = action[i, j]
                mean = self.mu[i, j]

                if adv >= 0.0:
                    # Good trial: reinforce sampled action.
                    self.mu[i, j] += self.cfg.lr_mu * activation * adv * (sampled - mean)
                    self.sigma[i, j] *= 1.0 - self.cfg.lr_sigma * activation * adv
                else:
                    # Bad trial: move away and explore more.
                    self.mu[i, j] -= self.cfg.lr_mu * activation * (-adv) * (sampled - mean)
                    self.sigma[i, j] *= 1.0 + self.cfg.lr_sigma * activation * (-adv)

        self.mu = np.clip(self.mu, 0.0, 1.0)
        self.sigma = np.clip(self.sigma, self.cfg.sigma_min, self.cfg.sigma_max)

        return raw_advantage

    def compensation_from_mean(self, phi: np.ndarray) -> np.ndarray:
        """
        Deterministic learned correction using current action means.
        """

        phi = np.asarray(phi, dtype=float).reshape(-1)
        local_delta_d = 2.0 * self.cfg.d_corr_max * (self.mu - 0.5)
        return phi @ local_delta_d

    def compensation_grid(
        self,
        feature_map: RBFFeatureMap,
        xlim=(-5.0, 5.0),
        ylim=(-5.0, 5.0),
        n: int = 35,
        t: float = 0.0,
    ):
        """
        Evaluate the learned mean correction field over a grid.

        Useful for plotting what CALA has learned. This does not use the true
        disturbance field.
        """

        xs = np.linspace(xlim[0], xlim[1], n)
        ys = np.linspace(ylim[0], ylim[1], n)
        X, Y = np.meshgrid(xs, ys)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for row in range(n):
            for col in range(n):
                phi = feature_map(np.array([X[row, col], Y[row, col]]), t)
                u = self.compensation_from_mean(phi)
                U[row, col] = u[0]
                V[row, col] = u[1]

        speed = np.sqrt(U**2 + V**2)
        return X, Y, U, V, speed

    def exploration_level(self) -> float:
        """
        Mean standard deviation across all feature/action distributions.
        """

        return float(np.mean(self.sigma))


# -----------------------------------------------------------------------------
# Low-frequency horizon trial manager
# -----------------------------------------------------------------------------

class CALAHorizonResidual:
    """
    Low-frequency horizon wrapper around CALAResidualCompensator.

    This class handles:
        - starting a trial
        - holding d_cala fixed for h MPC steps
        - saving the trial-start MPC prediction
        - recording actual closed-loop states
        - computing horizon prediction-error reward
        - updating CALA after h steps

    It implements:
        Option A: non-overlapping trials
        Option 1: closed-loop prediction error
    """

    def __init__(
        self,
        feature_map: RBFFeatureMap,
        learner: CALAResidualCompensator,
        h: int,
        nx: int,
        reward_cfg: HorizonRewardConfig | None = None,
    ):
        self.feature_map = feature_map
        self.learner = learner
        self.h = int(h)
        self.nx = int(nx)
        self.nu = int(learner.nu)
        self.reward_cfg = reward_cfg or HorizonRewardConfig()

        if self.reward_cfg.state_weights is None:
            state_weights = np.ones(self.nx)
            if self.nx >= 4:
                state_weights[:2] = 1.0
                state_weights[2:] = 0.25
            self.state_weights = state_weights
        else:
            self.state_weights = np.asarray(self.reward_cfg.state_weights, dtype=float).reshape(self.nx)

        self.active = False
        self.trial_step = 0
        self.trial_prediction: np.ndarray | None = None
        self.trial_actual: list[np.ndarray] = []
        self.trial_d_cala = np.zeros((self.nu, 1))
        self.trial_start_time: float | None = None

        self.reward_history: list[float] = []
        self.advantage_history: list[float] = []
        self.cost_history: list[float] = []
        self.prediction_error_history: list[float] = []
        self.terminal_error_history: list[float] = []
        self.effort_history: list[float] = []
        self.d_cala_history: list[np.ndarray] = []
        self.exploration_history: list[float] = []

    def should_start_trial(self) -> bool:
        """
        Return True when no trial is active.
        """

        return not self.active

    def begin_trial(self, x_global: np.ndarray, t: float, explore: bool = True) -> np.ndarray:
        """
        Begin a new horizon-length CALA trial.

        Parameters
        ----------
        x_global:
            Current global state. Used only to compute independent features.

        t:
            Current time.

        explore:
            If True, sample actions. If False, use mean corrections.

        Returns
        -------
        d_cala:
            Disturbance correction with shape (nu, 1). Pass this into MPC.
        """

        if self.active:
            raise RuntimeError("Cannot begin a new trial while one is active.")

        phi = self.feature_map(x_global, t)
        d_cala = self.learner.sample(phi, explore=explore).reshape(self.nu, 1)

        self.active = True
        self.trial_step = 0
        self.trial_prediction = None
        self.trial_actual = []
        self.trial_d_cala = d_cala.copy()
        self.trial_start_time = float(t)

        return d_cala

    def current_correction(self) -> np.ndarray:
        """
        Return the correction currently held during the active trial.
        """

        if not self.active:
            return np.zeros((self.nu, 1))

        return self.trial_d_cala.copy()

    def attach_prediction(self, predicted_states: np.ndarray) -> None:
        """
        Attach the MPC predicted horizon from trial start.

        If your MPC solves on x - xr, pass the reference-frame prediction:

            controller.result_state_sequence.reshape(h, nx)

        Do not pass the global visualization plan if actual states are recorded
        in the target-relative frame.
        """

        if not self.active:
            raise RuntimeError("No active CALA trial.")

        predicted_states = np.asarray(predicted_states, dtype=float)
        expected = (self.h, self.nx)

        if predicted_states.shape != expected:
            raise ValueError(f"Expected predicted_states shape {expected}, got {predicted_states.shape}.")

        # Only the first plan of the trial is evaluated. Subsequent MPC replans
        # during the same trial should not overwrite this saved prediction.
        if self.trial_prediction is None:
            self.trial_prediction = predicted_states.copy()

    def record_actual(self, actual_state_for_reward: np.ndarray) -> CALATrialResult | None:
        """
        Record one actual closed-loop state.

        If the MPC prediction was made in target-relative coordinates, pass:

            x - xr

        after plant and target are advanced to the same time.

        Returns None until h states have been collected. When the trial ends,
        returns CALATrialResult.
        """

        if not self.active:
            return None

        x = np.asarray(actual_state_for_reward, dtype=float).reshape(-1)

        if x.shape[0] != self.nx:
            raise ValueError(f"Expected actual state length {self.nx}, got {x.shape[0]}.")

        self.trial_actual.append(x.copy())
        self.trial_step += 1

        if self.trial_step >= self.h:
            return self.end_trial()

        return None

    def end_trial(self) -> CALATrialResult:
        """
        End the current trial, compute reward, update CALA, and reset buffers.
        """

        if self.trial_prediction is None:
            raise RuntimeError("Cannot end CALA trial: no prediction was attached.")

        actual = np.asarray(self.trial_actual, dtype=float)
        expected = (self.h, self.nx)

        if actual.shape != expected:
            raise RuntimeError(f"Expected actual history shape {expected}, got {actual.shape}.")

        reward, info = self.compute_reward(
            predicted=self.trial_prediction,
            actual=actual,
            d_cala=self.trial_d_cala,
        )

        advantage = self.learner.update(reward)
        exploration = self.learner.exploration_level()

        result = CALATrialResult(
            reward=float(reward),
            advantage=float(advantage),
            cost=float(info["cost"]),
            prediction_error=float(info["prediction_error"]),
            terminal_error=float(info["terminal_error"]),
            effort=float(info["effort"]),
            d_cala=self.trial_d_cala.flatten().copy(),
            exploration=float(exploration),
            start_time=self.trial_start_time,
        )

        self.reward_history.append(result.reward)
        self.advantage_history.append(result.advantage)
        self.cost_history.append(result.cost)
        self.prediction_error_history.append(result.prediction_error)
        self.terminal_error_history.append(result.terminal_error)
        self.effort_history.append(result.effort)
        self.d_cala_history.append(result.d_cala.copy())
        self.exploration_history.append(result.exploration)

        self.reset_trial()
        return result

    def reset_trial(self) -> None:
        """
        Clear active trial storage.
        """

        self.active = False
        self.trial_step = 0
        self.trial_prediction = None
        self.trial_actual = []
        self.trial_d_cala = np.zeros((self.nu, 1))
        self.trial_start_time = None

    def compute_reward(self, predicted: np.ndarray, actual: np.ndarray, d_cala: np.ndarray):
        """
        Compute reward from closed-loop prediction error over the MPC horizon.

        Cost:
            discounted state prediction error
            + terminal prediction error
            + learned-correction effort penalty

        Reward:
            -cost
        """

        predicted = np.asarray(predicted, dtype=float)
        actual = np.asarray(actual, dtype=float)
        d_cala = np.asarray(d_cala, dtype=float).reshape(self.nu)

        error = actual - predicted
        discounts = self.reward_cfg.discount ** np.arange(self.h)

        weighted_error_sq = (error**2) * self.state_weights.reshape(1, self.nx)
        step_error = np.sum(weighted_error_sq, axis=1)
        prediction_error = float(np.sum(discounts * step_error))

        terminal_error = float(np.sum((error[-1] ** 2) * self.state_weights))
        effort = float(np.dot(d_cala, d_cala))

        cost = (
            prediction_error
            + self.reward_cfg.terminal_weight * terminal_error
            + self.reward_cfg.effort_weight * effort
        )

        reward = -cost

        info = {
            "cost": cost,
            "prediction_error": prediction_error,
            "terminal_error": terminal_error,
            "effort": effort,
        }

        return reward, info

    def get_logs(self):
        """
        Return learning logs as arrays.
        """

        return {
            "reward": np.asarray(self.reward_history),
            "advantage": np.asarray(self.advantage_history),
            "cost": np.asarray(self.cost_history),
            "prediction_error": np.asarray(self.prediction_error_history),
            "terminal_error": np.asarray(self.terminal_error_history),
            "effort": np.asarray(self.effort_history),
            "d_cala": np.asarray(self.d_cala_history),
            "exploration": np.asarray(self.exploration_history),
        }
