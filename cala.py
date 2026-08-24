import json
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py


# ---------------------------------
# master calls
# ---------------------------------

def cala_suite(configs_base, controller, data_path="data/cala_learning.h5", overwrite=True):

    feature_map     = RTFeatureMap(configs_base)
    cala_nld        = CALA_NLD(configs_base, feature_map, controller.nu)
    cala_data       = CALADataset(filepath=data_path, overwrite=overwrite)
    horizon_manager = HorizonManager(configs_base, feature_map, cala_nld, controller, data=cala_data)

    return horizon_manager

def pre_controller(horizon_manager, x, t, x_error = None):

    if not horizon_manager.enable:
        return np.zeros(horizon_manager.n_inputs)

    if not horizon_manager.active:
        rl_adjustment = horizon_manager.begin_trial(x, t, x_error=x_error, explore=True)
    else:
        rl_adjustment = horizon_manager.rl_adjustment.copy()

    return rl_adjustment

def post_controller(horizon_manager, predicted_reference, x, xr):

    #predicted_reference = controller.result_state_sequence.reshape(controller.h, controller.nx).copy()
    reward, prediction_error, terminal_error, advantage = horizon_manager.update(predicted_reference, x - xr)
    if reward is not None:
        print(f"CALA reward: {reward:.4f}, advantage: {advantage:.4f}")


# ---------------------------------
# Radial-Temporal feature map
# ---------------------------------

class RTFeatureMap():
    
    def __init__(self, configs_base):

        with open(f'{configs_base}/config_cala.json') as f:
            cfg = json.load(f)
            cfg_fm = cfg["feature_map"]

        # bring stuff in
        self.x_lims     = cfg_fm["x_lims"] 
        self.y_lims     = cfg_fm["y_lims"]
        self.x_n        = cfg_fm["x_n"]
        self.y_n        = cfg_fm["y_n"]
        self.feature_centers    = cfg_fm["feature_centers"]
        self.sigma_factor       = cfg_fm["sigma_factor"]
        self.include_bias       = cfg_fm["include_bias"]
        self.include_time       = cfg_fm["include_time"]
        self.omega              = cfg_fm["omega"]
        self.normalize          = cfg_fm["normalize"]
        self.normalize_type     = cfg_fm["normalize_type"] 
        self.group_weights      = cfg_fm["group_weights"]
        if abs(sum(self.group_weights.values()) - 1.0) > 1e-12:
            raise ValueError("Feature group weights must sum to 1.0") 
  
        # define the feature locations
        if self.feature_centers is None:
            xs = np.linspace(self.x_lims[0], self.x_lims[1], self.x_n)
            ys = np.linspace(self.y_lims[0], self.y_lims[1], self.y_n)
            X, Y = np.meshgrid(xs, ys)
            self.centers = np.column_stack([X.reshape(-1), Y.reshape(-1)])
        else:
            self.centers = np.asarray(self.feature_centers, dtype=float)

        # width of spacing 
        dx = (self.x_lims[1] - self.x_lims[0]) / max(self.x_n - 1, 1)
        dy = (self.y_lims[1] - self.y_lims[0]) / max(self.y_n - 1, 1)
        self.sigma = self.sigma_factor* max(dx, dy)

        # names for the features 
        self.names = []
        self.names_index = {} 
        _names_index = 0
            
        # bias
        if self.include_bias:
            self.names += ["bias"]
            _start = _names_index
            _names_index += 1
            _end = _names_index
            self.names_index["bias"] = slice(_start, _end)

        # space
        _start = _names_index
        for i,c in enumerate(self.centers):
            self.names += [f"radial_{i:02d}({c[0]:.1f},{c[1]:.1f})"]
            _names_index += 1
            _end = _names_index
        self.names_index["radial"] = slice(_start, _end)

        # time
        if self.include_time:
            _start = _names_index
            self.names += ["sin(t)"]
            self.names += ["cos(t)"]
            _names_index += 2
            _end = _names_index
            self.names_index["time"] = slice(_start, _end)
    
    # compute spatial (e.g., radial) basis function
    def _compute_spatial_feature(self, d, basis_type = 'radial'):
        if basis_type == 'radial':
            return float(np.exp(-np.dot(d, d) / (self.sigma**2)))
        else:
            raise ValueError(f"Unsupported spacial basis {basis_type}")

    # compute temporal (e.g., sines and cosines) basis function
    def _compute_time_features(self, t, basis_type = 'sin'):
        if basis_type == 'sin':
            return 0.5 * (1.0 + np.sin(self.omega * t))
        if basis_type == 'cos':
            return 0.5 * (1.0 + np.cos(self.omega * t))
        else:
            raise ValueError(f"Unsupported time basis {basis_type}")

    # compute the feature activations 
    def build_features(self, x, t):

        # initial values
        values = []

        # bias (captures bias in the signal)
        if self.include_bias:
            values.append(1.0)

        # space (just use x,y positions)
        x = np.asarray(x).reshape(-1)[:2]
        for c in self.centers:
            rel = x - c
            values.append(self._compute_spatial_feature(d = rel, basis_type = 'radial'))

        # time
        if self.include_time:
            values.append(self._compute_time_features(t, basis_type = 'sin'))
            values.append(self._compute_time_features(t, basis_type = 'cos'))

        # make feature set
        phi = np.asarray(values)

        if self.normalize:
            if self.normalize_type == 'local':
                # normalize within each feature set
                for name, index in self.names_index.items():
                    phi[index] = (phi[index]/ (np.sum(phi[index]) + 1e-12))
                    phi[index] *= self.group_weights[name]

            elif self.normalize_type == 'softmax':
                tau = 0.3
                phi = np.exp(phi / tau) / np.sum(np.exp(phi / tau))
            else:
                phi = phi / (np.sum(phi) + 1e-12) # global is default
                

        self.phi = phi.copy()

        return phi

    # evaluate one feature over the workspace for plotting
    def activation_grid(self, feature_index, t, resolution = 100):

        xs = np.linspace(self.x_lims[0], self.x_lims[1], resolution)
        ys = np.linspace(self.y_lims[0], self.y_lims[1], resolution)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)

        for row in range(resolution):
            for col in range(resolution):
                phi = self.build_features([X[row, col], Y[row, col]], t)
                Z[row, col] = phi[feature_index]

        return X, Y, Z

    def plot_feature(self, feature_index = 0, t = 0.0):

        X, Y, Z = self.activation_grid(feature_index=feature_index, t=t)
        fig, ax = plt.subplots(figsize=(7, 7))
        contour = ax.contourf(X,Y,Z,levels=30)
        fig.colorbar(contour,ax=ax,label="Feature activation")
        ax.scatter(self.centers[:, 0], self.centers[:, 1], marker="x",label="Feature centres")
        ax.set_title(self.names[feature_index])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(self.x_lims)
        ax.set_ylim(self.y_lims)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.legend()

        plt.show()

    # plot activation across x,y,t (fixing one)
    def plot_fixed_axis(self, feature_index = 0, fixed_axis = 1, fixed_at = 0.0, resolution  =100):

        xs = np.linspace(self.x_lims[0], self.x_lims[1], resolution)
        ys = np.linspace(self.y_lims[0], self.y_lims[1], resolution)
        times = np.linspace(0.0, 2.0 * np.pi / self.omega, resolution)

        # select which axis
        if fixed_axis == 0:
            grid_0 = ys
            grid_1 = times
            axis_names = ["x_1", "time", "x_0"]
        elif fixed_axis == 1:
            grid_0 = xs
            grid_1 = times
            axis_names = ["x_0", "time", "x_1"]
        else:
            grid_0 = xs
            grid_1 = ys
            axis_names = ["x_0", "x_1", "time"]

        X, T = np.meshgrid(grid_0, grid_1)
        Z = np.zeros_like(X)

        for row in range(X.shape[0]):
            for col in range(X.shape[1]):

                if fixed_axis == 0:
                    x = [fixed_at, X[row, col]]
                    t = T[row, col]
                elif fixed_axis == 1:
                    x = [X[row, col], fixed_at]
                    t = T[row, col]
                else:
                    x = [X[row, col], T[row, col]]
                    t = fixed_at
                #phi = self.build_features([X[row, col], fixed_at],T[row, col])
                phi = self.build_features(x, t)
                Z[row, col] = phi[feature_index]

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        surface = ax.plot_surface(X,T,Z, cmap="viridis",vmin=0.0,vmax=1.0,rcount=resolution,ccount=resolution,)
        fig.colorbar(surface,ax=ax,label="Feature activation",shrink=0.7)

        ax.set_title(f"{self.names[feature_index]} at ${axis_names[2]}={fixed_at}$")
        ax.set_xlabel(f"${axis_names[0]}$")
        ax.set_ylabel(f"${axis_names[1]}$")
        ax.set_zlabel("Feature activation")

        plt.show()


# --------------------------
# CALA modeller
# --------------------------

"""
Continuous Action Learning Automata for modelling nonlinear residual disturbances.

For each feature i and input dimension j, CALA maintains a Gaussian action
distribution: a_ij ~ N(mu_ij, sigma_ij^2), where:
    mu       : (n_features, n_inputs)
    sigma    : (n_features, n_inputs)
    action   : (n_features, n_inputs)

The sampled normalized action is clipped to [0, 1] and mapped to a local
physical residual disturbance: d_local = 2 * d_max * (a_ij - 0.5),
where: d_local  : (n_features, n_inputs)

The current feature activations are stored as phi : (n_features,)

The final CALA residual disturbance is the feature-weighted sum of the local
residual disturbances: rl_adjustment = phi @ d_local, with dimensions:

    (n_inputs,) = (n_features,) @ (n_features, n_inputs)

Therefore: rl_adjustment   : (n_inputs,)

This residual can then be used as follows:

1. if mpc "rl_parameter" is "d_adjustment", it simply adds to the d directly:

    adds to the existing local disturbance estimate used by MPC: d_eff = d_hat + rl_adjustment 

2. if mpc  "rl_parameter" is "R_adjustment", it biases the control effort components of the optimization:

    adjusts weights of R matrix: R = diag[Ro * exp(rl_adjustment[0]), Ro * exp(rl_adjustment[1])]

"""

class CALA_NLD():

    def __init__(self, configs_base, feature_map, n_inputs):

        # enforce formats for passed in variables
        self.feature_map    = feature_map
        self.n_features     =  len(self.feature_map.names)
        self.n_inputs       =  int(n_inputs)

        # bring in configs 
        with open(f'{configs_base}/config_cala.json') as f:
            cfg = json.load(f)
            cfg_cala = cfg["cala_mpc_nld"]

        self.d_max          = cfg_cala["d_max"]
        self.mu_init        = cfg_cala["mu_init"]
        self.mu_init_noise  = cfg_cala["mu_init_noise"]
        self.sigma_init     = cfg_cala["sigma_init"]
        self.sigma_min      = cfg_cala["sigma_min"]
        self.sigma_max      = cfg_cala["sigma_max"]
        self.learning_rate  = cfg_cala["learning_rate"]
        self.variance_rate  = cfg_cala["variance_rate"]
        self.reward_rate    = cfg_cala["reward_rate"]
        self.advantage_gain = cfg_cala["advantage_gain"]
        self.seed           = cfg_cala["seed"]

        # initialize 
        self.rng        = np.random.default_rng(self.seed)
        self.mu         = np.full((self.n_features, self.n_inputs), self.mu_init, dtype=float)              # means

        if self.mu_init_noise is not None:
            self.mu = self.rng.normal(self.mu_init, self.mu_init_noise, size=(self.n_features, self.n_inputs))
            self.mu = np.clip(self.mu, 0.0, 1.0)

        self.sigma      = np.full((self.n_features, self.n_inputs), self.sigma_init, dtype=float)      # stds
        self.action     = np.zeros((self.n_features, self.n_inputs))

        # rl adjustment 
        self.rl_adjustment     = np.zeros((self.n_inputs))

        # phi (n_features,) @ d_local (n_features, n_inputs) c        
        self._d_local   = np.zeros((self.n_features, self.n_inputs)) # use _ because it's kind of just an internal param

        # note: phi comes from feature map - don't duplicate
        #self.phi        = np.zeros((self.n_features)) 

        # reward updates 
        self.reward_mean = None         # will take value of first reward signal (i.e., not zeros)
        self.phi = None                 # stores activations
        self.reward_scale = None
        self.last_advantage_scaled = 0.0

    # sample an action from the distributions (exploit or explore)
    def _sample(self, explore = True):

        # default is to explore the distribution
        if explore:
            action = self.rng.normal(self.mu, self.sigma)
        # else, exploit the mean
        else:
            action = self.mu.copy()

        # ensure between 0 and 1
        self.action = np.clip(action, 0.0, 1.0)

        # note: action size should be (self.n_features, self.n_inputs)
        return self.action

    # map sample to disturbance correction 
    def _map_sample_to_disturbance(self, phi):

        # reshape to allow either list or array
        phi = np.asarray(phi, dtype=float).reshape(-1)

        # translate acton to local disturbance
        self._d_local = 2.0 * self.d_max * (self.action - 0.5)

        # blend local features into one input signal
        self.rl_adjustment = phi @ self._d_local

        # note: this should be the size of the inputs now
        return self.rl_adjustment

    # combine two above
    def sample_map(self, phi, explore = True):

        # reshape to allow either list or array, test dims
        phi = np.asarray(phi, dtype=float).reshape(-1)
        if phi.shape[0] != self.n_features:
            raise ValueError(f"dimensions of phi: {phi.shape} do not match feature length {self.n_features}") 

        self.phi = phi.copy()
        action = self._sample(explore = explore)
        rl_adjustment = self._map_sample_to_disturbance(phi = phi)

        return action, rl_adjustment

    # update the distribution based on a received reward signal
    def update(self, reward):

        # we need to initialize the reward mean 
        if self.reward_mean is None:
            self.reward_mean = float(reward)
            return 0.0 

        # update reward mean (uses low pass filter, reward_mean = (1-B)reward_mean + B*reward)
        reward_error = reward - self.reward_mean
        self.reward_mean += self.reward_rate * reward_error

        # compute the advantage
        abs_error = abs(reward_error)
        if self.reward_scale is None:
            self.reward_scale = max(abs_error, 1e-3)
        else:
            self.reward_scale += (self.reward_rate * (abs_error- self.reward_scale))
        advantage = (reward_error/max(self.reward_scale, 1e-3))
        advantage_scaled = np.tanh(self.advantage_gain* advantage)
        self.last_advantage_scaled = float(advantage_scaled)

        # for each feature
        for i in range(self.n_features):

            # pull out the activation
            activation = self.phi[i]

            # ignore low values
            if abs(activation) < 1e-9:
                continue

            # for each input
            for j in range(self.n_inputs):

                # good trial
                if advantage_scaled >= 0.0:

                    # move the mean toward the sampled action
                    self.mu[i, j] += (self.learning_rate * activation * advantage_scaled * (self.action[i, j] - self.mu[i, j]))

                    # reduce exploration in proportion
                    self.sigma[i, j] *= (1.0 - self.variance_rate * activation * advantage_scaled)

                # bad trial
                else:

                    # move the mean away from the sampled action
                    self.mu[i, j] -= (self.learning_rate * activation * (-advantage_scaled) * (self.action[i, j] - self.mu[i, j]))

                    # increase exploration in proportion
                    self.sigma[i, j] *= (1.0 + self.variance_rate * activation * (-advantage_scaled))

        # clip stats
        self.mu     = np.clip(self.mu, 0.0, 1.0)
        self.sigma  = np.clip(self.sigma, self.sigma_min, self.sigma_max)

        # return
        return advantage_scaled

    # returns rl_adjustment, based on current distro
    def get_correction(self, phi):

        phi = np.asarray(phi, dtype=float).reshape(-1)
        if phi.shape[0] != self.n_features:
            raise ValueError(f"dimensions of phi: {phi.shape} do not match feature length: {self.n_features}")

        return phi @ (2.0 * self.d_max * (self.mu - 0.5))


    # evaluate the learned corrections over searchspace
    def _correction_grid(self, t=0.0, resolution=100):

        xs = np.linspace(self.feature_map.x_lims[0], self.feature_map.x_lims[1], resolution)
        ys = np.linspace(self.feature_map.y_lims[0], self.feature_map.y_lims[1], resolution)
        X, Y = np.meshgrid(xs, ys)
        D = np.zeros((self.n_inputs, resolution, resolution))

        for row in range(resolution):
            for col in range(resolution):
                phi = self.feature_map.build_features([X[row, col], Y[row, col]], t)
                D[:, row, col] = self.get_correction(phi)

        magnitude = np.sqrt(np.sum(D**2, axis=0))

        return X, Y, D, magnitude

    # plots a nice map of the computed correction (mean value)
    def plot_correction(self, t=0.0, resolution=100, folder = 'visualization/cala/'):

        X, Y, D, magnitude = self._correction_grid(t=t, resolution=resolution)

        if self.n_inputs == 2:

            r_scale_0 = np.exp(D[0])
            r_scale_1 = np.exp(D[1])

            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

            contour_0 = axes[0].contourf(X, Y, r_scale_0, levels=30)
            fig.colorbar(contour_0, ax=axes[0], label=r"$R_0/R_{0,0}$")
            axes[0].scatter(self.feature_map.centers[:, 0], self.feature_map.centers[:, 1], marker="x", label="Feature centres")
            axes[0].set_title(r"$R_0$ scaling")
            axes[0].set_xlabel("$x_0$")
            axes[0].set_ylabel("$x_1$")
            axes[0].set_xlim(self.feature_map.x_lims)
            axes[0].set_ylim(self.feature_map.y_lims)
            axes[0].set_aspect("equal")
            axes[0].grid(True)
            axes[0].legend()

            contour_1 = axes[1].contourf(X, Y, r_scale_1, levels=30)
            fig.colorbar(contour_1, ax=axes[1], label=r"$R_1/R_{0,1}$")
            axes[1].scatter(self.feature_map.centers[:, 0], self.feature_map.centers[:, 1], marker="x", label="Feature centres")
            axes[1].set_title(r"$R_1$ scaling")
            axes[1].set_xlabel("$x_0$")
            axes[1].set_xlim(self.feature_map.x_lims)
            axes[1].set_ylim(self.feature_map.y_lims)
            axes[1].set_aspect("equal")
            axes[1].grid(True)
            axes[1].legend()

            fig.suptitle(f"Learning - Learned R Adjustment Field at $t={t}$")
            fig.tight_layout()
            fig.savefig(os.path.join(folder, f"Learning - Learned R Adjustment Field at t -{int(t)}.png"), dpi=200, bbox_inches="tight")
            #fig.savefig(f"visualization/cala/Learning - Learned R Adjustment Field at t -{int(t)}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        else:

            fig, ax = plt.subplots(figsize=(7, 7))
            r_scale = np.exp(D[0])

            contour = ax.contourf(X, Y, r_scale, levels=30)
            fig.colorbar(contour, ax=ax, label=r"$R/R_0$")

            ax.scatter(self.feature_map.centers[:, 0], self.feature_map.centers[:, 1], marker="x", label="Feature centres")
            ax.set_title(f"Learning - Learned R Adjustment Field at $t={t}$")
            ax.set_xlabel("$x_0$")
            ax.set_ylabel("$x_1$")
            ax.set_xlim(self.feature_map.x_lims)
            ax.set_ylim(self.feature_map.y_lims)
            ax.set_aspect("equal")
            ax.grid(True)
            ax.legend()

            fig.savefig(f"visualization/cala/Learning - Learned R Adjustment Field at t -{int(t)}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

# --------------------------------
# Manage the learning over horizon
# --------------------------------

class HorizonManager():

    def __init__(self, configs_base, feature_map, cala, mpc, data=None):

        with open(f'{configs_base}/config_cala.json') as f:

            cfg = json.load(f)
            cfg_hm = cfg["horizon_manager"]

        # pull from feature map
        self.feature_map    = feature_map

        # pull from cala
        self.cala           = cala

        # pull from mpc
        self.n_states           = int(mpc.nx)
        self.n_inputs           = int(mpc.nu)
        self.prediction_h       = int(mpc.h)
        self.replan_trigger     = int(mpc.replan_trigger) 
        self.state_weights      = np.diag(mpc.Q) 
        #self.effort_weights     = np.diag(mpc.R)
        self.terminal_weights   = np.diag(mpc.P)
        self.B                  = np.asarray(mpc.B).copy()
        self.Ts                 = float(getattr(mpc, "Ts", np.nan))

        # benchmark tolerances
        self.success_tolerance  = 0.25
        self.settling_tolerance = self.success_tolerance


        # reward configs
        self.reward_mode = cfg_hm["reward_mode"]
        self.reward_period = cfg_hm["reward_period"]

        # constraints on configs (debugging)
        '''
        if self.reward_period > 1:
           raise ValueError(f"reward_period must be set to 1 (for now)")
        if mpc.replan_mode != 'receding_horizon':
           raise ValueError(f"mpc must be set to receding horizon control (for now)")
        '''

        self.compensation_weight = cfg_hm["compensation_weight"]
        self.discount           = cfg_hm["discount"]
        self.enable             = cfg_hm["enable"]

        # things required for trial tracking 
        self.active         = False         # is it actively collecting eviidence
        self.trial_step     = 0
        self.predicted      = None
        self.actual         = []
        self.rl_mean        = np.zeros(self.n_inputs)
        self.rl_adjustment  = np.zeros(self.n_inputs)
        self.d_true         = np.zeros(self.n_inputs) 
        self.d_hat          = np.zeros(self.n_inputs) 
        self.start_time     = None

        # r-learning params
        self.r_lambda_effort        = cfg_hm["r_lambda_effort"]
        self.r_disturbance_gain     = cfg_hm["r_disturbance_gain"]
        self.r_progress_floor       = cfg_hm["r_progress_floor"]
        self.u_trial                = []
        self.d_hat_trial            = []
        self.progress               = 0.0
        self.command_effort         = 0.0
        self.net_effort             = 0.0
        self.disturbance_energy     = 0.0

        self.data = data

        # storage (probably storing too much)
        '''
        self.history = {

            # basic
            "step": [],
            "reward": [],
            "advantage": [],
            "prediction_error": [],
            "terminal_error": [],
            "rl_adjustment": [],
            "d_true": [],
            "d_hat": [],
            "rl_mean": [],
            "mu": [],
            "sigma": [],

            # performance
            "tracking_rmse": [],
            "position_rmse": [],
            "final_distance": [],
            "success": [],
            "settling_steps": [],
            "settling_time": [],
            "state_cost": [],

            # prediction
            "prediction_rmse": [],
            "prediction_position_rmse": [],

            # control
            "control_effort": [],
            "effort_ratio": [],

            # disturbance rejection
            "d_hat_error": [],
            "d_residual_sampled": [],
            "d_residual_mean": [],

            # r- learning terms
            "progress": [],
            "command_effort": [],
            "net_effort": [],
            "disturbance_energy": [],
            "advantage_scaled": [],
            "phi": []
            }
        '''
    def begin_trial(self, x, t, x_error = None, explore = True):

        # build feature map for this time/space
        phi = self.feature_map.build_features(x, t)

        # get the means
        rl_mean = self.cala.get_correction(phi)

        # compute the corresponding disturbance
        _, rl_adjustment = self.cala.sample_map(phi, explore = explore)

        # re-initialize the things
        self.active = True
        self.trial_step = 0
        self.predicted = None
        self.actual = []
        self.rl_mean = rl_mean.copy()
        self.rl_adjustment = rl_adjustment.copy()

        if x_error is not None:
            self.x_start = np.asarray(x_error).reshape(-1).copy()
        else:
            self.x_start = None 

        self.u_trial = []
        self.d_hat_trial = []

        self.start_time = float(t)

        return self.rl_adjustment

    def _update_prediction(self, prediction):

        prediction      = np.asarray(prediction, dtype=float).reshape(-1, self.n_states)
        # in case there is a len mismatch, take smallest (shouldn't happen anymore)
        n_keep          = min(self.reward_period,self.prediction_h,prediction.shape[0])
        self.predicted  = prediction[:n_keep,:].copy()

    def _update_actual(self, x_new):

        # accumulate a list of actual states
        x_new = np.asarray(x_new, dtype=float).reshape(-1)
        self.actual.append(x_new.copy())
        self.trial_step += 1

    def _end_trial(self):

        # compute reward and update CALA
        reward, prediction_error, terminal_error    = self._compute_reward()
        advantage                                   = self.cala.update(reward)

        # metrics ( probably too many)
        actual              = np.asarray(self.actual, dtype=float).reshape(-1, self.n_states)
        predicted           = np.asarray(self.predicted, dtype=float).reshape(-1, self.n_states)
        n                   = min(actual.shape[0], predicted.shape[0])
        prediction_residual = actual[:n] - predicted[:n]
        tracking_rmse       = float(np.sqrt(np.mean(actual**2)))
        position_norm       = np.linalg.norm(actual[:, :2], axis=1)
        position_rmse       = float(np.sqrt(np.mean(position_norm**2)))
        final_distance      = float(position_norm[-1])

        success             = float(final_distance <= self.success_tolerance)
        state_cost          = float(np.mean(np.sum(self.state_weights.reshape(1, -1) * actual**2, axis=1)))

        prediction_rmse             = float(np.sqrt(np.mean(prediction_residual**2)))
        prediction_position_norm    = np.linalg.norm(prediction_residual[:, :2], axis=1)
        prediction_position_rmse    = float(np.sqrt(np.mean(prediction_position_norm**2)))

        settled             = np.where([np.all(position_norm[k:] <= self.settling_tolerance) for k in range(len(position_norm))])[0]
        settling_steps      = float(settled[0] + 1) if len(settled) > 0 else np.nan
        settling_time       = float(settling_steps * self.Ts) if np.isfinite(settling_steps) and np.isfinite(self.Ts) else np.nan

        u_trial         = np.asarray(self.u_trial, dtype=float).reshape(-1, self.n_inputs)
        control_effort  = float(np.mean(np.sum(u_trial**2, axis=1))) if len(u_trial) > 0 else np.nan
        effort_ratio    = float(self.net_effort / (self.command_effort + 1e-12)) if self.reward_mode == "for_r" else np.nan

        d_hat_error = float(np.linalg.norm(self.d_true - self.d_hat))
        if self.reward_mode == "for_r":
            d_residual_sampled = np.nan
            d_residual_mean = np.nan
        else:
            d_residual_sampled = float(np.linalg.norm(self.d_true - self.d_hat - self.rl_adjustment))
            d_residual_mean = float(np.linalg.norm(self.d_true - self.d_hat - self.rl_mean))


        # ---------------------------------------------
        # store completed CALA trial
        # ---------------------------------------------

        if self.data is not None:

            self.data.stage(

                phase="learning",

                # basic
                step=self.start_time,
                reward=reward,
                advantage=advantage,
                prediction_error=prediction_error,
                terminal_error=terminal_error,
                rl_adjustment=self.rl_adjustment,
                d_true=self.d_true,
                d_hat=self.d_hat,
                rl_mean=self.rl_mean,
                mu=self.cala.mu,
                sigma=self.cala.sigma,

                # performance
                tracking_rmse=tracking_rmse,
                position_rmse=position_rmse,
                final_distance=final_distance,
                success=success,
                settling_steps=settling_steps,
                settling_time=settling_time,
                state_cost=state_cost,

                # prediction
                prediction_rmse=prediction_rmse,
                prediction_position_rmse=prediction_position_rmse,

                # control
                control_effort=control_effort,
                effort_ratio=effort_ratio,

                # disturbance rejection
                d_hat_error=d_hat_error,
                d_residual_sampled=d_residual_sampled,
                d_residual_mean=d_residual_mean,

                # R-learning
                progress=self.progress,
                command_effort=self.command_effort,
                net_effort=self.net_effort,
                disturbance_energy=self.disturbance_energy,
                advantage_scaled=self.cala.last_advantage_scaled,

                # feature activation
                phi=self.cala.phi,
            )

            # write this completed trial immediately
            self.data.store()

        self.active = False

        return reward, prediction_error, terminal_error, advantage

    def _compute_reward(self):

        actual_all      = np.asarray(self.actual,dtype=float).reshape(-1, self.n_states)
        predicted_all   = np.asarray(self.predicted,dtype=float).reshape(-1, self.n_states)
        n               = min(actual_all.shape[0],predicted_all.shape[0])
        actual_pred     = actual_all[:n, :]
        predicted       = predicted_all[:n, :]
        error           = actual_pred - predicted

        discounts           = self.discount ** np.arange(n)
        weighted_error      = (self.state_weights.reshape(1, self.n_states)* error**2)
        step_error          = np.sum(weighted_error,axis=1)
        prediction_error    = float(np.sum(discounts * step_error)/(np.sum(discounts) + 1e-12))
        terminal_error      = float(np.sum(error[-1]**2* self.terminal_weights))

        if self.reward_mode == "prediction":

            cost = prediction_error 

        elif self.reward_mode == 'idealized':

            disturbance_error = self.rl_adjustment - self.d_true
            cost = (np.dot(disturbance_error, disturbance_error)+ self.compensation_weight* np.dot(self.rl_adjustment, self.rl_adjustment))

        # infer the equivalent input-channel prediction residual B*e_x = d_true - rl_adjustment
        elif self.reward_mode == 'prediction_infer':

            disturbance_error = np.linalg.lstsq(self.B, error.T, rcond=None)[0].T
            step_error = np.sum(disturbance_error**2,axis=1)
            cost = float(np.sum(discounts * step_error)/ (np.sum(discounts) + 1e-12) )

        elif self.reward_mode == 'for_r':

            '''
            Custom R-learning reward
                reward = progress - lambda * disturbance-normalized control effort  
                where:  progress = V_start - V_end, command_effort = ||u||^2, net_effort = ||u||^2 /(1 + beta * ||d_hat||^2)
            '''
  
            # ----------------------------------------------------------
            # 1. Tracking progress: +ve = got better, -ve = got worse
            # ----------------------------------------------------------

            x_start = np.asarray(self.x_start).reshape(-1)
            x_end = np.asarray(actual_all[-1],dtype=float).reshape(-1)
            V_start = float(np.sum(self.state_weights* x_start**2))
            V_end = float(np.sum(self.state_weights* x_end**2))

            # I've tried for few progress ideas
            #progress_denominator = max(V_start, self.r_progress_floor)
            #self.progress = (V_start - V_end)
            #self.progress = ((V_start - V_end)/progress_denominator)

            # latest
            self.progress = float(np.log((V_start + self.r_progress_floor)/(V_end + self.r_progress_floor)))

            # ----------------------------------------------------------
            # 2. Control/disturbance histories
            # ----------------------------------------------------------

            u_trial         = np.asarray(self.u_trial).reshape(-1, self.n_inputs)
            d_hat_trial     = np.asarray(self.d_hat_trial).reshape(-1, self.n_inputs)
            n_effort        = min(u_trial.shape[0], d_hat_trial.shape[0])
            u_trial         = u_trial[:n_effort]
            d_hat_trial     = d_hat_trial[:n_effort]

            # ----------------------------------------------------------
            # 3. Raw command effort
            # ----------------------------------------------------------

            command_effort_step = np.sum(u_trial**2, axis=1)

            # ----------------------------------------------------------
            # 4. Estimated disturbance energy
            # ----------------------------------------------------------

            disturbance_energy_step = np.sum(d_hat_trial**2, axis=1)

            # ----------------------------------------------------------
            # 5. Disturbance-normalized effort: big = encourages more control
            # ----------------------------------------------------------

            net_effort_step = (command_effort_step/(1.0+ self.r_disturbance_gain * disturbance_energy_step))

            # ----------------------------------------------------------
            # 6. Average over trial
            # ----------------------------------------------------------

            effort_discounts        = (self.discount** np.arange(n_effort))
            effort_norm             = (np.sum(effort_discounts)+ 1e-12)
            self.command_effort     = float(np.sum(effort_discounts* command_effort_step)/ effort_norm)
            self.disturbance_energy = float(np.sum(effort_discounts * disturbance_energy_step)/ effort_norm)
            self.net_effort         = float(np.sum(effort_discounts* net_effort_step) / effort_norm)

            # ----------------------------------------------------------
            # 7. Final R-learning cost
            # ----------------------------------------------------------

            cost = (-self.progress + self.r_lambda_effort * self.net_effort) # 0.015 is a good r_lambda_effort
            #cost = -self.progress

        else:

            raise ValueError(f"invalid reward mode: {self.reward_mode}")

        reward = -cost
        #reward = np.exp(-cost / 10)
        #reward = -np.log1p(cost)

        return reward, prediction_error, terminal_error

    def update(self, prediction, x):

        if not self.active:
            return None, None, None, None

        # if no prediction loaded (clears beginning of trial) - only needed for prediction mode
        if self.predicted is None: #and self.reward_mode == "prediction":
            self._update_prediction(prediction)

        # if active (always, unless between trials)
        self._update_actual(x)

        # if at end of trial (when period reached)
        if self.trial_step >= self.reward_period:
            return self._end_trial() # returns reward and last advantage

        return None, None, None, None

    def plot_learning(self, folder="visualization/cala", phase="learning"):

        os.makedirs(folder, exist_ok=True)

        if self.data is None:
            print("No CALA dataset attached.")
            return

        if len(self.data) == 0:
            print("No completed CALA trials to plot.")
            return

        # ---------------------------------------------
        # pull learning history FROM DISK
        # ---------------------------------------------

        history = self.data.read(phase)

        steps       = history["step"]
        rewards     = history["reward"]
        advantages  = history["advantage"]

        prediction_errors   = history["prediction_error"]
        terminal_errors     = history["terminal_error"]

        rl_adjustment   = history["rl_adjustment"]
        d_true          = history["d_true"]
        d_hat           = history["d_hat"]
        rl_mean         = history["rl_mean"]

        sigma_hist      = history["sigma"]
        phi_hist        = history["phi"]

        progress        = history["progress"]
        command_effort  = history["command_effort"]
        net_effort      = history["net_effort"]
        disturbance_energy = history["disturbance_energy"]

        tracking_rmse   = history["tracking_rmse"]
        position_rmse   = history["position_rmse"]
        final_distance  = history["final_distance"]
        success         = history["success"]
        settling_time   = history["settling_time"]
        state_cost      = history["state_cost"]

        prediction_rmse             = history["prediction_rmse"]
        prediction_position_rmse    = history["prediction_position_rmse"]

        control_effort              = history["control_effort"]
        effort_ratio_hist           = history["effort_ratio"]

        d_hat_error             = history["d_hat_error"]
        d_residual_sampled      = history["d_residual_sampled"]
        d_residual_mean         = history["d_residual_mean"]

        # reward and advantage history
        # ----------------------------
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(steps, rewards, marker="o", label="Reward")
        ax.plot(steps, advantages, marker="o", label="Advantage")
        ax.set_title("CALA learning history")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("Reward")
        ax.grid(True)
        ax.legend()
        fig.savefig(os.path.join(folder, "reward_history.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # reward components
        # -----------------
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(steps, prediction_errors, marker="o", label="Prediction error")
        ax.plot(steps, terminal_errors, marker="o", label="Terminal error")
        ax.set_title("CALA reward components")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("Weighted squared error")
        ax.grid(True)
        ax.legend()
        fig.savefig(os.path.join(folder, "reward_components.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # correction history
        # ------------------
        fig, ax = plt.subplots(figsize=(9, 6))
        for j in range(self.n_inputs):
            ax.plot(steps, rl_adjustment[:, j], marker="o", label=f"$rl_{{cala,{j}}}$")
        ax.axhline(0.0, linestyle="--")
        ax.set_title("CALA correction history")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("Correction")
        ax.grid(True)
        ax.legend()
        fig.savefig(os.path.join(folder, "correction_history.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # empirical reward surface
        # ------------------------
        if self.n_inputs == 2:
            fig, ax = plt.subplots(figsize=(7, 7))

            if len(rewards) >= 3:
                try:
                    contour = ax.tricontourf(rl_adjustment[:, 0], rl_adjustment[:, 1], rewards, levels=30)
                    fig.colorbar(contour, ax=ax, label="Reward")
                except RuntimeError:
                    points = ax.scatter(rl_adjustment[:, 0], rl_adjustment[:, 1], c=rewards)
                    fig.colorbar(points, ax=ax, label="Reward")
            else:
                points = ax.scatter(rl_adjustment[:, 0], rl_adjustment[:, 1], c=rewards)
                fig.colorbar(points, ax=ax, label="Reward")

            ax.scatter(rl_adjustment[-1, 0], rl_adjustment[-1, 1], marker="x", s=100, label="Latest trial")
            ax.set_title("Empirical reward over CALA corrections")
            ax.set_xlabel("$rl_{0}$")
            ax.set_ylabel("$rl_{1}$")
            ax.set_xlim(-self.cala.d_max, self.cala.d_max)
            ax.set_ylim(-self.cala.d_max, self.cala.d_max)
            ax.set_aspect("equal")
            ax.grid(True)
            ax.legend()
            fig.savefig(os.path.join(folder, "reward_surface.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

        # exploration history
        # -------------------
        sigma_mean = np.mean(sigma_hist, axis=(1, 2))
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(steps, sigma_mean, marker="o")
        ax.set_title("CALA exploration level")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("Mean sigma")
        ax.grid(True)
        fig.savefig(os.path.join(folder, "exploration_history.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # disturbance comparison
        # ----------------------
        fig, axes = plt.subplots(self.n_inputs, 1, figsize=(9, 3.5 * self.n_inputs), sharex=True)
        if self.n_inputs == 1:
            axes = [axes]

        for j, ax in enumerate(axes):
            ax.plot(steps, d_hat[:, j], linestyle="-", label=f"$d_{{hat,{j}}}$")
            ax.plot(steps, d_true[:, j], linestyle=":", label=f"$d_{{true,{j}}}$")
            ax.plot(steps, rl_mean[:, j], linestyle="--", label=f"$rl_{{{j}}}$")
            ax.set_ylabel("Disturbance")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Trial start time")
        fig.suptitle("CALA compensation tracking")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "compensation_tracking.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # R-learning reward components
        # ----------------------------
        effort_ratio = net_effort / (command_effort + 1e-12)
        print(f"mean effort penalty ratio: {np.mean(effort_ratio):.3f}")

        fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        axes[0].plot(steps, progress)
        axes[0].axhline(0.0, linestyle="--", linewidth=1)
        axes[0].set_ylabel("Progress")
        axes[0].grid(True)

        axes[1].plot(steps, command_effort)
        axes[1].set_ylabel(r"$||u||^2$")
        axes[1].grid(True)

        axes[2].plot(steps, disturbance_energy)
        axes[2].set_ylabel(r"$||\hat{d}||^2$")
        axes[2].grid(True)

        axes[3].plot(steps, net_effort)
        axes[3].set_ylabel("Normalized effort")
        axes[3].set_xlabel("Trial start time")
        axes[3].grid(True)

        fig.suptitle("R-learning Reward Components")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "r_reward_components.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # learned R mean
        # --------------
        r_scale_mean = np.exp(rl_mean)
        fig, ax = plt.subplots(figsize=(9, 6))

        for j in range(self.n_inputs):
            ax.plot(steps, r_scale_mean[:, j], linewidth=2, label=f"$R_{j}/R_{{0,{j}}}$")

        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_title("Learned MPC R scaling")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("$R/R_0$")
        ax.grid(True)
        ax.legend()
        fig.savefig(os.path.join(folder, "r_mean_history.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # active exploration level
        # ------------------------
        sigma_feature = np.mean(sigma_hist, axis=2)
        sigma_active = np.sum(phi_hist * sigma_feature, axis=1)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(steps, sigma_active, linewidth=2, label="Active-feature sigma")
        ax.plot(steps, sigma_mean, linewidth=1.5, alpha=0.6, label="Global mean sigma")
        ax.set_title("CALA exploration level")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("Sigma")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "cala_exploration_level.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # sigma histories
        # ---------------
        if len(sigma_hist) == 0:
            print("No sigma history available.")
            return

        sigma_feat = sigma_hist.mean(axis=2)
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(sigma_feat.T, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_title("CALA feature sigma heatmap")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("Feature index")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Sigma")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "all_sigmas.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


        # performance metrics
        # -------------------
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        axes[0].plot(steps, tracking_rmse, label="State RMSE")
        axes[0].plot(steps, position_rmse, label="Position RMSE")
        axes[0].set_ylabel("RMSE")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(steps, final_distance, label="Final distance")
        axes[1].axhline(self.success_tolerance, linestyle="--", linewidth=1, label="Success tolerance")
        axes[1].set_ylabel("Distance")
        axes[1].grid(True)
        axes[1].legend()

        success_rate = np.cumsum(success) / np.arange(1, len(success) + 1)
        axes[2].plot(steps, success_rate)
        axes[2].set_ylabel("Success rate")
        axes[2].set_xlabel("Trial start time")
        axes[2].set_ylim(0.0, 1.05)
        axes[2].grid(True)

        fig.suptitle("Performance - Tracking Metrics")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "performance_tracking_metrics.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # settling time
        # -------------
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(steps, settling_time, marker="o")
        ax.set_title("Performance - Settling Time")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("Settling time")
        ax.grid(True)
        fig.savefig(os.path.join(folder, "performance_settling_time.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # prediction RMSE
        # ---------------
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(steps, prediction_rmse, label="State prediction RMSE")
        ax.plot(steps, prediction_position_rmse, label="Position prediction RMSE")
        ax.set_title("Control - Prediction RMSE")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("RMSE")
        ax.grid(True)
        ax.legend()
        fig.savefig(os.path.join(folder, "control_prediction_rmse.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # state cost and control effort
        # -----------------------------
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(steps, state_cost)
        axes[0].set_ylabel(r"$J_x$")
        axes[0].grid(True)
        axes[1].plot(steps, control_effort)
        axes[1].set_ylabel(r"$||u||^2$")
        axes[1].set_xlabel("Trial start time")
        axes[1].grid(True)
        fig.suptitle("Control - State Cost and Effort")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "control_state_cost_effort.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # disturbance rejection error
        # ---------------------------
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(steps, d_hat_error, label=r"$||d_{true}-\hat{d}||$")
        if np.any(np.isfinite(d_residual_sampled)):
            ax.plot(steps, d_residual_sampled, label="Sampled residual")
        if np.any(np.isfinite(d_residual_mean)):
            ax.plot(steps, d_residual_mean, label="Mean-policy residual")
        ax.set_title("Robustness - Disturbance Rejection Error")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("Residual magnitude")
        ax.grid(True)
        ax.legend()
        fig.savefig(os.path.join(folder, "robustness_disturbance_rejection_error.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


        # performance - position RMSE
        # ---------------------------
        window = 10
        kernel = np.ones(window) / window
        position_rmse_smooth = np.convolve(position_rmse, kernel, mode="valid")

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(steps, position_rmse, alpha=0.25, label="Trial RMSE")
        ax.plot(steps[window - 1:], position_rmse_smooth, linewidth=2, label=f"{window}-trial moving mean")
        ax.set_title("Performance - Position RMSE")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("Position RMSE")
        ax.grid(True)
        ax.legend()

        fig.savefig(os.path.join(folder, "performance_position_rmse.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # effort ratio history
        # --------------------
        if np.any(np.isfinite(effort_ratio_hist)):
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(steps, effort_ratio_hist)
            ax.axhline(np.nanmean(effort_ratio_hist), linestyle="--", linewidth=1, label=f"Mean = {np.nanmean(effort_ratio_hist):.3f}")
            ax.set_title("Control - Effort Penalty Ratio")
            ax.set_xlabel("Trial start time")
            ax.set_ylabel("Effort ratio")
            ax.grid(True)
            ax.legend()
            fig.savefig(os.path.join(folder, "control_effort_penalty_ratio.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

        print(f"CALA plots saved to: {folder}")

    # load a learned policy (default: -1 as the last trial)
    def load_policy(self, phase="learning", trial=-1):

        if self.data is None:
            raise RuntimeError("Cannot load CALA policy: no dataset found.")

        history = self.data.read(phase)

        if "mu" not in history:
            raise KeyError("No learned parameter means stored in CALA dataset.")

        # load learned policy
        self.cala.mu = np.asarray(history["mu"][trial], dtype=float).copy()

        # sigma, if needed later 
        if "sigma" in history:
            self.cala.sigma = np.asarray(history["sigma"][trial],dtype=float).copy()

        print(f"Loaded CALA policy from {self.data.filepath}, trial {trial}")

        return self.cala.mu



# ---------------------------------
# A custom CALA dataset
# ---------------------------------

class CALADataset:

    def __init__(self, filepath="data/cala_learning.h5", overwrite=True):

        self.filepath = filepath

        # remove existing learning file for a new run
        if overwrite and os.path.exists(self.filepath):
            os.remove(self.filepath)

        # one row/sample corresponds to one completed CALA trial
        self.columns = [

            # basic learning terms
            "step",
            "reward",
            "advantage",
            "prediction_error",
            "terminal_error",
            "rl_adjustment",
            "d_true",
            "d_hat",
            "rl_mean",
            "mu",
            "sigma",

            # performance
            "tracking_rmse",
            "position_rmse",
            "final_distance",
            "success",
            "settling_steps",
            "settling_time",
            "state_cost",

            # prediction
            "prediction_rmse",
            "prediction_position_rmse",

            # control
            "control_effort",
            "effort_ratio",

            # disturbance rejection
            "d_hat_error",
            "d_residual_sampled",
            "d_residual_mean",

            # R-learning
            "progress",
            "command_effort",
            "net_effort",
            "disturbance_energy",
            "advantage_scaled",

            # feature information
            "phi",
        ]

        self.phase = "learning"

        # staging area
        for col in self.columns:
            setattr(self, col, [])

    # -------------------------------------------------
    # stage one or more completed CALA trial results
    # -------------------------------------------------

    def stage(self, phase="learning", **kwargs):

        self.phase = phase

        for key, value in kwargs.items():

            if key not in self.columns:
                raise KeyError(f"Unknown CALA data column: {key}")

            if value is not None:
                getattr(self, key).append(np.asarray(value).copy())

    # -------------------------------------------------
    # write staged data to HDF5
    # -------------------------------------------------

    def store(self, flush_after=True):

        folder = os.path.dirname(self.filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)

        with h5py.File(self.filepath, "a") as f:

            if self.phase not in f:
                group = f.create_group(self.phase)
            else:
                group = f[self.phase]

            for key in self.columns:

                staged = getattr(self, key)

                if len(staged) == 0:
                    continue

                data = self._make_batch(staged)

                if key not in group:

                    group.create_dataset(
                        key,
                        data=data,
                        maxshape=(None,) + data.shape[1:],
                        chunks=True,
                    )

                else:

                    dataset = group[key]

                    # protect against shape changes during a run
                    if dataset.shape[1:] != data.shape[1:]:
                        raise ValueError(
                            f"CALA dataset shape mismatch for '{key}': "
                            f"stored {dataset.shape[1:]}, new {data.shape[1:]}"
                        )

                    old_len = dataset.shape[0]
                    new_len = old_len + data.shape[0]

                    dataset.resize(new_len, axis=0)
                    dataset[old_len:new_len] = data

        if flush_after:
            self.clear()


    # -------------------------------------------------
    # read complete learning phase or individual column
    # -------------------------------------------------

    def read(self, phase="learning", key=None):

        with h5py.File(self.filepath, "r") as f:

            if phase not in f:
                raise KeyError(
                    f"CALA phase '{phase}' not found in {self.filepath}"
                )

            group = f[phase]

            if key is not None:

                if key not in group:
                    raise KeyError(
                        f"CALA column '{key}' not found in phase '{phase}'"
                    )

                return group[key][:]

            return {
                name: group[name][:]
                for name in group.keys()
            }


    # -------------------------------------------------
    # clear staging area
    # -------------------------------------------------

    def clear(self):

        for col in self.columns:
            setattr(self, col, [])


    # -------------------------------------------------
    # number of stored trials
    # -------------------------------------------------

    def __len__(self):

        if not os.path.exists(self.filepath):
            return 0

        with h5py.File(self.filepath, "r") as f:

            if self.phase not in f:
                return 0

            group = f[self.phase]

            if "reward" not in group:
                return 0

            return group["reward"].shape[0]


    # -------------------------------------------------
    # convert staged samples to HDF5 batch
    # -------------------------------------------------

    def _make_batch(self, staged):

        arrays = [np.asarray(item) for item in staged]

        # every staged item represents ONE completed trial
        #
        # scalar:
        #     () -> (N,)
        #
        # vector:
        #     (nu,) -> (N, nu)
        #
        # matrix:
        #     (n_features, nu)
        #       -> (N, n_features, nu)

        if arrays[0].ndim == 0:
            return np.asarray(
                [item.item() for item in arrays]
            )

        return np.stack(arrays, axis=0)


#-----------
# testing     
# ---------

# import matplotlib.pyplot as plt
# import mpc 

# test_mpc = mpc.MPC([0,0,0,0])

# test_map = RTFeatureMap()

# #test.plot_feature(feature_index = 0, t = 0.0)
# #test.plot_fixed_axis(feature_index = -1, fixed_axis = 0, fixed_at = 0.0)
# test_phi = test_map.build_features([0.0, 0.0], 0.0)
# print(f"phi is type: {type(test_phi)} and shape: {test_phi.shape}")

# test_cala = CALA_NLD(test_map, 2)
# action, rl_adjustment = test_cala.sample_map(test_phi)

# print(f"selected disturbance:  {rl_adjustment}")

# print(test_cala.get_correction(test_phi))
# test_cala.plot_correction(t=0.0)

# test_hm = HorizonManager(test_map, test_cala, test_mpc)
