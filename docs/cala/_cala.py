"""
CALA horizon residual learner

Purpose
-------
This module learns a low-frequency residual correction to the MPC
disturbance estimate.

Your MPC already estimates the local input-channel disturbance:

    d_hat[k]

from the one-step prediction error.

CALA learns an additive correction:

    d_eff[k] = d_hat[k] + d_cala[k]

where d_cala is held fixed for one MPC horizon.

Learning trial
--------------
Option A: low-frequency, non-overlapping trials.

If h = 20 and Ts = 0.1, one CALA trial lasts:

    h * Ts = 2 seconds

At the beginning of the trial:
    1. CALA samples d_cala.
    2. MPC solves using d_hat + d_cala.
    3. The predicted state sequence is saved.

Over the next h steps:
    4. The receding-horizon controller continues operating.
    5. Actual closed-loop states are recorded.

At the end of h steps:
    6. CALA compares the original predicted horizon against the actual
       closed-loop states.
    7. That horizon prediction error becomes the reward signal.

Important convention
--------------------
Because your MPC solves in the target-relative frame:

    controller.solve(x - xr, u)

the prediction error used for CALA should also usually be computed in
the same target-relative frame:

    predicted: controller.result_state_sequence
    actual:    x - xr

This avoids mixing disturbance-learning error with target-motion error.
"""

import numpy as np


# ------------------------------------------------------------------
# Feature map
# ------------------------------------------------------------------

class VortexFeatureMap:
    """
    Feature map for CALA based on your existing VortexField.

    This class uses the vortex centers and sigmas from your implemented
    nonlinear field as RBF feature locations. It does NOT call
    field.compute_disturbance(), so it is not directly using the true
    disturbance vector as a training target.

    It only answers:

        "Which nonlinear region of the workspace am I in?"

    The CALA learner then associates each active region with a local
    correction to the MPC disturbance estimate.

    Parameters
    ----------
    field:
        Your nonlinear_field.VortexField instance.

    sigma_scale:
        Multiplier applied to the field's vortex sigmas when constructing
        RBF feature widths. Larger values create smoother, more overlapping
        features.

    include_bias:
        Adds a feature that is always active. This lets CALA learn a global
        correction.

    include_time:
        Adds sin/cos time features. Useful because your field has a rotating
        background and time-varying vortex gains.

    normalize:
        If True, features are normalized to sum to one. This makes the CALA
        correction a weighted blend of local corrections.
    """

    def __init__(
        self,
        field,
        sigma_scale=1.25,
        include_bias=True,
        include_time=True,
        normalize=True,
    ):
        self.field = field
        self.sigma_scale = float(sigma_scale)
        self.include_bias = bool(include_bias)
        self.include_time = bool(include_time)
        self.normalize = bool(normalize)

        self.base_centers = np.asarray(self.field.config.vortex_centers, dtype=float)
        self.base_sigmas = np.asarray(self.field.config.vortex_sigmas, dtype=float)

        if len(self.base_centers) != len(self.base_sigmas):
            raise ValueError("vortex_centers and vortex_sigmas must have the same length.")

    @property
    def n_features(self):
        n = len(self.base_centers)

        if self.include_bias:
            n += 1

        if self.include_time:
            n += 2

        return n

    def __call__(self, x_global, t):
        """
        Compute feature activations phi(x,t).

        Parameters
        ----------
        x_global:
            The true/global plant state. Expected shape is compatible with:

                x = [px, py, vx, vy]

            Only position x[:2] is used.

        t:
            Current simulation time.

        Returns
        -------
        phi:
            Feature vector with shape (n_features,).
        """

        x_global = np.asarray(x_global, dtype=float).reshape(-1)
        p = x_global[:2]

        phi = []

        # --------------------------------------------------------------
        # Bias feature
        # --------------------------------------------------------------
        # Always active. Lets CALA learn a global correction.
        if self.include_bias:
            phi.append(1.0)

        # --------------------------------------------------------------
        # Spatial RBF features
        # --------------------------------------------------------------
        # Use your field's moving centers if available.
        #
        # These centres are physical disturbance centres in the simulation.
        # The feature map uses them only as local basis centres.
        # It does not use the true disturbance vector d(p,t).
        centers = self.field.evolve_centers(t)

        for c, sigma in zip(centers, self.base_sigmas):
            diff = p - c
            width = self.sigma_scale * sigma
            rbf = np.exp(-np.dot(diff, diff) / (width**2))
            phi.append(rbf)

        # --------------------------------------------------------------
        # Time features
        # --------------------------------------------------------------
        # Your field has:
        #
        #     background ~ [cos(omega t), sin(omega t)]
        #
        # so these give CALA a simple phase/context signal.
        if self.include_time:
            omega = self.field.config.omega
            phi.append(0.5 * (1.0 + np.sin(omega * t)))
            phi.append(0.5 * (1.0 + np.cos(omega * t)))

        phi = np.asarray(phi, dtype=float)

        # Normalized features make the final d_cala a convex-like blend.
        if self.normalize:
            phi = phi / (np.sum(phi) + 1e-12)

        return phi


# ------------------------------------------------------------------
# CALA learner
# ------------------------------------------------------------------

class CALAHorizonResidual:
    """
    CALA learner for horizon-level MPC prediction error.

    The action is an additive disturbance correction:

        d_cala in R^nu

    The MPC should plan with:

        d_eff = d_hat + d_cala

    Each feature owns a continuous action distribution for each input
    dimension:

        a_ij ~ Normal(mu_ij, sigma_ij^2)

    where a_ij is clipped to [0, 1] and mapped to a physical correction:

        delta_d_ij = 2 * d_corr_max_j * (a_ij - 0.5)

    Therefore:

        a = 0.0  -> maximum negative correction
        a = 0.5  -> zero correction
        a = 1.0  -> maximum positive correction

    The final correction is the feature-weighted blend:

        d_cala = sum_i phi_i(x,t) * delta_d_i
    """

    def __init__(
        self,
        feature_map,
        nu,
        nx,
        h,
        d_corr_max=0.5,
        mu_init=0.5,
        sigma_init=0.25,
        sigma_min=0.03,
        sigma_max=0.60,
        lr_mu=0.12,
        lr_sigma=0.06,
        reward_beta=0.05,
        advantage_gain=2.0,
        state_weights=None,
        terminal_weight=1.0,
        effort_weight=0.02,
        discount=0.98,
        seed=None,
    ):
        self.feature_map = feature_map

        self.nu = int(nu)
        self.nx = int(nx)
        self.h = int(h)

        # Allow scalar or per-input correction bounds.
        if np.isscalar(d_corr_max):
            self.d_corr_max = np.full(self.nu, float(d_corr_max))
        else:
            self.d_corr_max = np.asarray(d_corr_max, dtype=float).reshape(self.nu)

        self.n_features = self.feature_map.n_features

        # CALA action distributions.
        #
        # mu[i, j] and sigma[i, j] define the distribution for:
        #     feature i, input channel j
        self.mu = np.full((self.n_features, self.nu), float(mu_init))
        self.sigma = np.full((self.n_features, self.nu), float(sigma_init))

        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        self.lr_mu = float(lr_mu)
        self.lr_sigma = float(lr_sigma)

        # Running reward baseline.
        # The first completed trial initializes this baseline.
        self.reward_bar = None
        self.reward_beta = float(reward_beta)
        self.advantage_gain = float(advantage_gain)

        # Horizon prediction error weights.
        if state_weights is None:
            # Default for [px, py, vx, vy]:
            # prioritize position error, but include velocity error.
            state_weights = np.ones(self.nx)
            if self.nx >= 4:
                state_weights[:2] = 1.0
                state_weights[2:] = 0.25

        self.state_weights = np.asarray(state_weights, dtype=float).reshape(self.nx)

        self.terminal_weight = float(terminal_weight)
        self.effort_weight = float(effort_weight)
        self.discount = float(discount)

        self.rng = np.random.default_rng(seed)

        # Active trial variables.
        self.active = False
        self.trial_step = 0

        self.trial_phi = None
        self.trial_action = None
        self.trial_local_delta = None
        self.trial_d_cala = self._zero_correction()

        self.trial_prediction = None
        self.trial_actual = []

        self.trial_start_time = None
        self.trial_start_global_state = None

        # Logs.
        self.reward_history = []
        self.cost_history = []
        self.prediction_error_history = []
        self.terminal_error_history = []
        self.effort_history = []
        self.d_cala_history = []
        self.exploration_history = []

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def _zero_correction(self):
        return np.zeros((self.nu, 1))

    def should_start_trial(self):
        """
        Low-frequency Option A:

        Start a new trial only when no trial is active.

        This creates non-overlapping trials:
            trial 1: k = 0 ... h-1
            trial 2: k = h ... 2h-1
            etc.
        """

        return not self.active

    def current_correction(self):
        """
        Return the correction currently being held for the active trial.

        If no trial is active, return zero.
        """

        if not self.active:
            return self._zero_correction()

        return self.trial_d_cala.copy()

    # ------------------------------------------------------------------
    # Trial lifecycle
    # ------------------------------------------------------------------

    def begin_trial(self, x_global, t):
        """
        Start a new CALA trial.

        This should be called before controller.solve(...), because the
        sampled correction should be injected into the MPC prediction.

        Parameters
        ----------
        x_global:
            Current true/global plant state.

        t:
            Current time.

        Returns
        -------
        d_cala:
            Disturbance correction to pass into MPC.
        """

        if self.active:
            raise RuntimeError("A CALA trial is already active.")

        # Compute nonlinear context features.
        phi = self.feature_map(x_global, t)

        # Sample normalized CALA actions in [0, 1].
        action = self.rng.normal(self.mu, self.sigma)
        action = np.clip(action, 0.0, 1.0)

        # Map normalized action to physical disturbance correction.
        #
        # local_delta shape:
        #     (n_features, nu)
        local_delta = 2.0 * (action - 0.5) * self.d_corr_max.reshape(1, self.nu)

        # Blend local corrections using feature activations.
        #
        # d_cala shape:
        #     (nu, 1)
        d_cala = phi @ local_delta
        d_cala = d_cala.reshape(self.nu, 1)

        # Store trial information for delayed reward/update.
        self.active = True
        self.trial_step = 0

        self.trial_phi = phi
        self.trial_action = action
        self.trial_local_delta = local_delta
        self.trial_d_cala = d_cala

        self.trial_prediction = None
        self.trial_actual = []

        self.trial_start_time = float(t)
        self.trial_start_global_state = np.asarray(x_global, dtype=float).reshape(-1).copy()

        return d_cala

    def attach_prediction(self, predicted_states):
        """
        Attach the MPC predicted horizon from the beginning of the trial.

        For your current reference-frame MPC, pass:

            controller.result_state_sequence.reshape(controller.h, controller.nx)

        Do NOT add xr for the CALA reward if you are going to record actual
        states as x - xr.

        Parameters
        ----------
        predicted_states:
            Shape (h, nx). These should be target-relative predicted states
            if your controller was solved using x - xr.
        """

        if not self.active:
            raise RuntimeError("Cannot attach prediction because no trial is active.")

        predicted_states = np.asarray(predicted_states, dtype=float)

        expected = (self.h, self.nx)
        if predicted_states.shape != expected:
            raise ValueError(
                f"predicted_states must have shape {expected}, "
                f"but got {predicted_states.shape}"
            )

        # Only attach the first prediction made at trial start.
        # Later MPC replans should not overwrite this prediction, because
        # this trial evaluates the closed-loop error against the original
        # horizon prediction.
        if self.trial_prediction is None:
            self.trial_prediction = predicted_states.copy()

    def record_actual(self, actual_state_for_reward):
        """
        Record one actual closed-loop state for the active trial.

        For your current reference-frame MPC, pass:

            x - xr

        after the plant and target have been evolved to the same time.

        Returns
        -------
        None:
            If the trial is still running.

        result dict:
            If the trial has just completed and CALA was updated.
        """

        if not self.active:
            return None

        x = np.asarray(actual_state_for_reward, dtype=float).reshape(-1)

        if x.shape[0] != self.nx:
            raise ValueError(
                f"actual_state_for_reward must have length {self.nx}, "
                f"but got {x.shape[0]}"
            )

        self.trial_actual.append(x.copy())
        self.trial_step += 1

        if self.trial_step >= self.h:
            return self.end_trial()

        return None

    def end_trial(self):
        """
        End the active trial, compute the delayed reward, update CALA,
        log results, and reset the trial state.
        """

        if not self.active:
            return None

        if self.trial_prediction is None:
            raise RuntimeError("Cannot end CALA trial: no prediction was attached.")

        actual = np.asarray(self.trial_actual, dtype=float)

        expected = (self.h, self.nx)
        if actual.shape != expected:
            raise RuntimeError(
                f"Trial actual state history has shape {actual.shape}; "
                f"expected {expected}"
            )

        reward, info = self.compute_reward(
            predicted=self.trial_prediction,
            actual=actual,
            d_cala=self.trial_d_cala,
        )

        self.update(reward)

        result = {
            "reward": float(reward),
            "cost": float(info["cost"]),
            "prediction_error": float(info["prediction_error"]),
            "terminal_error": float(info["terminal_error"]),
            "effort": float(info["effort"]),
            "d_cala": self.trial_d_cala.flatten().copy(),
            "start_time": self.trial_start_time,
            "exploration": self.exploration_level(),
        }

        self.reward_history.append(result["reward"])
        self.cost_history.append(result["cost"])
        self.prediction_error_history.append(result["prediction_error"])
        self.terminal_error_history.append(result["terminal_error"])
        self.effort_history.append(result["effort"])
        self.d_cala_history.append(result["d_cala"])
        self.exploration_history.append(result["exploration"])

        self.reset_trial()

        return result

    def reset_trial(self):
        """
        Clear active trial storage.
        """

        self.active = False
        self.trial_step = 0

        self.trial_phi = None
        self.trial_action = None
        self.trial_local_delta = None
        self.trial_d_cala = self._zero_correction()

        self.trial_prediction = None
        self.trial_actual = []

        self.trial_start_time = None
        self.trial_start_global_state = None

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_reward(self, predicted, actual, d_cala):
        """
        Compute closed-loop horizon prediction-error reward.

        The prediction error is:

            e_i = actual_i - predicted_i

        over the h states in the trial.

        Cost:
            discounted state prediction error
            + terminal prediction error
            + small penalty on d_cala magnitude

        Reward:
            negative cost
        """

        predicted = np.asarray(predicted, dtype=float)
        actual = np.asarray(actual, dtype=float)
        d_cala = np.asarray(d_cala, dtype=float).reshape(self.nu, 1)

        error = actual - predicted

        # Discount later errors if desired.
        discounts = self.discount ** np.arange(self.h)

        # Weighted state error at each horizon index.
        weighted_error_sq = (error**2) * self.state_weights.reshape(1, self.nx)
        step_error = np.sum(weighted_error_sq, axis=1)

        prediction_error = float(np.sum(discounts * step_error))

        # Extra terminal penalty.
        terminal_error = float(np.sum((error[-1] ** 2) * self.state_weights))

        # Regularize the learned correction so CALA does not simply push hard.
        effort = float(np.dot(d_cala.flatten(), d_cala.flatten()))

        cost = (
            prediction_error
            + self.terminal_weight * terminal_error
            + self.effort_weight * effort
        )

        reward = -cost

        info = {
            "cost": cost,
            "prediction_error": prediction_error,
            "terminal_error": terminal_error,
            "effort": effort,
        }

        return reward, info

    # ------------------------------------------------------------------
    # CALA update
    # ------------------------------------------------------------------

    def update(self, reward):
        """
        CALA-style continuous action update.

        If reward is better than the running baseline:
            - move means toward the sampled actions
            - reduce variances

        If reward is worse than the running baseline:
            - move means away from the sampled actions
            - increase variances

        Feature activation controls credit assignment.
        Strongly active features get larger updates.
        """

        if self.trial_phi is None or self.trial_action is None:
            return

        # Initialize baseline on first trial.
        if self.reward_bar is None:
            self.reward_bar = float(reward)
            return

        # Advantage: positive means better than expected.
        raw_advantage = float(reward) - self.reward_bar

        # Relative scaling prevents large costs from causing violent updates.
        denom = abs(self.reward_bar) + 1e-9
        scaled_advantage = raw_advantage / denom

        # Smooth bounded advantage.
        advantage = np.tanh(self.advantage_gain * scaled_advantage)

        # Update running baseline after computing advantage.
        self.reward_bar += self.reward_beta * raw_advantage

        phi = self.trial_phi
        action = self.trial_action

        for i in range(self.n_features):
            activation = phi[i]

            if activation < 1e-10:
                continue

            for j in range(self.nu):
                sampled = action[i, j]
                mean = self.mu[i, j]

                if advantage >= 0.0:
                    # Good trial:
                    # reinforce sampled action.
                    self.mu[i, j] += (
                        self.lr_mu
                        * activation
                        * advantage
                        * (sampled - mean)
                    )

                    # reduce exploration.
                    self.sigma[i, j] *= (
                        1.0
                        - self.lr_sigma * activation * advantage
                    )

                else:
                    # Bad trial:
                    # move away from sampled action.
                    self.mu[i, j] -= (
                        self.lr_mu
                        * activation
                        * (-advantage)
                        * (sampled - mean)
                    )

                    # increase exploration.
                    self.sigma[i, j] *= (
                        1.0
                        + self.lr_sigma * activation * (-advantage)
                    )

        self.mu = np.clip(self.mu, 0.0, 1.0)
        self.sigma = np.clip(self.sigma, self.sigma_min, self.sigma_max)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def mean_correction(self, x_global, t):
        """
        Deterministic correction using current action means.

        Useful later for plotting the learned correction field.
        """

        phi = self.feature_map(x_global, t)

        local_delta = 2.0 * (self.mu - 0.5) * self.d_corr_max.reshape(1, self.nu)

        d_cala = phi @ local_delta
        return d_cala.reshape(self.nu, 1)

    def exploration_level(self):
        """
        Average action standard deviation.

        High value:
            CALA is still exploring.

        Low value:
            CALA has become confident.
        """

        return float(np.mean(self.sigma))

    def get_logs(self):
        """
        Return learning logs as numpy arrays.
        """

        return {
            "reward": np.asarray(self.reward_history),
            "cost": np.asarray(self.cost_history),
            "prediction_error": np.asarray(self.prediction_error_history),
            "terminal_error": np.asarray(self.terminal_error_history),
            "effort": np.asarray(self.effort_history),
            "d_cala": np.asarray(self.d_cala_history),
            "exploration": np.asarray(self.exploration_history),
        }