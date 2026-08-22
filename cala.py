import json
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------
# master calls
# ---------------------------------

def cala_suite(controller):

    feature_map     = RTFeatureMap()
    cala_nld        = CALA_NLD(feature_map, controller.nu)
    horizon_manager = HorizonManager(feature_map, cala_nld, controller)

    return horizon_manager

def pre_controller(horizon_manager, x, t):

    if not horizon_manager.enable:
        return np.zeros(horizon_manager.n_inputs)

    if not horizon_manager.active:
        rl_adjustment = horizon_manager.begin_trial(x, t, explore=True)
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
    
    def __init__(self):

        with open('configs/config_cala.json') as f:
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
# Residual NL Disturbance CALA
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



Note: 
- treating as a residual disturbance (beyond what is being modelled linearly) creates some dependency on agents velocity, maybe? 
- I am thinking the direction of travel matters...



"""

class CALA_NLD():

    def __init__(self, feature_map, n_inputs):

        # enforce formats for passed in variables
        self.feature_map    = feature_map
        self.n_features     =  len(self.feature_map.names)
        self.n_inputs       =  int(n_inputs)

        # bring in configs 
        with open('configs/config_cala.json') as f:
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

        # rl_adjustment (n_inputs,) = 
        self.rl_adjustment     = np.zeros((self.n_inputs))

        # phi (n_features,) @ d_local (n_features, n_inputs) c        
        self._d_local   = np.zeros((self.n_features, self.n_inputs)) # use _ because it's kind of just an internal param

        # note: phi comes from feature map - don't duplicate
        #self.phi        = np.zeros((self.n_features)) 

        # reward updates 
        self.reward_mean = None         # will take value of first reward signal (i.e., not zeros)
        self.phi = None                 # stores activations



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
        advantage = reward_error / (abs(self.reward_mean) + 1e-8)
        self.reward_mean += self.reward_rate * reward_error

        # bound/scale the advantage for cleaner updates
        advantage_scaled  = np.tanh(self.advantage_gain * advantage)

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
        return advantage

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

    # plot the learned corrections
    def plot_correction(self, t=0.0, resolution=100):

        X, Y, D, magnitude = self._correction_grid(t=t, resolution=resolution)

        fig, ax = plt.subplots(figsize=(7, 7))
        contour = ax.contourf(X, Y, magnitude, levels=30)
        fig.colorbar(contour, ax=ax, label="CALA correction magnitude")

        if self.n_inputs == 2:
            spacing = max(resolution // 15, 1)
            ax.quiver(X[::spacing, ::spacing], Y[::spacing, ::spacing], D[0, ::spacing, ::spacing], D[1, ::spacing, ::spacing])

        ax.scatter(self.feature_map.centers[:, 0], self.feature_map.centers[:, 1], marker="x", label="Feature centres")
        ax.set_title(f"Learned CALA correction at $t={t}$")
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$x_1$")
        ax.set_xlim(self.feature_map.x_lims)
        ax.set_ylim(self.feature_map.y_lims)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.legend()

        plt.show()


# --------------------------
# Horizon learning manager 
# --------------------------
'''
Manages actions/rewards over finite prediction horizon

Typical order or ops:

        feature_map = FeatureMap()
        cala = CALA_NLD(feature_map, n_inputs)
        horizon_learning_manager = HorizonLearner(feature_map, cala, mpc)

        # before solving MPC
        if learner.should_start_trial():
            rl_adjustment = learner.begin_trial(x, t)
        else:
            rl_adjustment = learner.current_correction()

        controller.solve(x - xr, u, rl_adjustment=rl_adjustment)

        # save only the first predicted horizon of the CALA trial
        learner.attach_prediction(
            controller.result_state_sequence.reshape(controller.h, controller.nx)
        )

        # after advancing plant/target one step
        result = learner.record_actual(x - xr)

'''

class HorizonManager():

    def __init__(self, feature_map, cala, mpc):

        with open('configs/config_cala.json') as f:

            cfg = json.load(f)
            cfg_hm = cfg["horizon_manager"]

        # pull from feature map
        self.feature_map    = feature_map

        # pull from cala
        self.cala           = cala

        # pull from mpc
        self.n_states           = int(mpc.nx)
        self.n_inputs           = int(mpc.nu)
        #self.h                  = int(mpc.h)
        self.h                  = int(mpc.replan_trigger)
        self.replan_trigger     = int(mpc.replan_trigger) 
        self.state_weights      = np.diag(mpc.Q) 
        #self.effort_weights     = np.diag(mpc.R)
        self.terminal_weights   = np.diag(mpc.P)
        self.B                  = np.asarray(mpc.B).copy()


        # reward configs
        self.reward_mode = cfg_hm["reward_mode"]
        # if self.reward_mode == 'prediction':
        #     self.reward_period = self.replan_trigger 
        # else:
        #     self.reward_period = cfg_hm["reward_period"]

        #self.reward_period = self.replan_trigger 
        self.reward_period = cfg_hm["reward_period"]
        if self.reward_period > 1:
           raise ValueError(f"reward_period must be set to 1 (for now)")
        if mpc.replan_mode != 'receding_horizon':
           raise ValueError(f"mpc must be set to receding horizon control (for now)")
   

        #if self.reward_mode == 'prediction' and mpc.replan_mode == 'receding_horizon':
        #    raise ValueError(f"Cannot use prediction-error based RL reward when MPC replan_mode is receding horizon. Select horizon (h) or control (m) horizon.")
        


        self.compensation_weight = cfg_hm["compensation_weight"]

        # pull from configs
        self.discount           = cfg_hm["discount"]
        self.enable             = cfg_hm["enable"]
        # things required for trial tracking 
        self.active = False         # is it actively collecting eviidence
        self.trial_step = 0
        self.predicted = None
        self.actual = []
        self.rl_mean = np.zeros(self.n_inputs)
        self.rl_adjustment = np.zeros(self.n_inputs)
        self.d_true = np.zeros(self.n_inputs) # used for some rewards
        self.d_hat = np.zeros(self.n_inputs) # the linear assumption
        self.start_time = None

        # storage
        self.history = {
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
        }


    def begin_trial(self, x, t, explore = True):

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
        self.start_time = float(t)

        return self.rl_adjustment

    def _update_prediction(self, prediction):

        #prediction= np.asarray(prediction, dtype=float)
        #self.predicted = prediction.reshape(self.h, self.n_states).copy()
        prediction = np.asarray(prediction, dtype=float).reshape(-1, self.n_states)
        self.predicted = prediction[:self.h, :].copy()

    def _update_actual(self, x_new):

        # accumulate a list of actual states
        x_new = np.asarray(x_new, dtype=float).reshape(-1)
        self.actual.append(x_new.copy())
        self.trial_step += 1


    def _end_trial(self):

        #add stuff here
        reward, prediction_error, terminal_error = self._compute_reward()
        advantage = self.cala.update(reward)

        # store stuff
        self.history["step"].append(self.start_time)
        self.history["reward"].append(reward)
        self.history["advantage"].append(advantage)
        self.history["prediction_error"].append(prediction_error)
        self.history["terminal_error"].append(terminal_error)
        self.history["rl_adjustment"].append(self.rl_adjustment.copy())
        self.history["d_true"].append(self.d_true.copy())
        self.history["d_hat"].append(self.d_hat.copy())
        self.history["rl_mean"].append(self.rl_mean.copy())
        self.history["mu"].append(self.cala.mu.copy())
        self.history["sigma"].append(self.cala.sigma.copy())

        self.active = False

        return reward, prediction_error, terminal_error, advantage

    def _compute_reward(self):

        # actual = np.asarray(self.actual, dtype=float)
        # predicted = np.asarray(self.predicted, dtype=float)

        # tighter
        actual = np.asarray(self.actual, dtype=float).reshape(-1, self.n_states)
        predicted = np.asarray(self.predicted,dtype=float).reshape(-1, self.n_states)
        # number of valid state comparisons in this trial
        n = min(actual.shape[0], predicted.shape[0])
        if n == 0:
            raise RuntimeError("Cannot compute reward: no actual/predicted states available.")
        actual = actual[:n, :]
        predicted = predicted[:n, :]


        error = actual - predicted
        discounts = self.discount ** np.arange(n)

        weighted_error = self.state_weights.reshape(1, self.n_states) * error**2
        step_error = np.sum(weighted_error, axis=1)
        prediction_error = float(np.sum(discounts * step_error)/ (np.sum(discounts) + 1e-12))
        terminal_error = float(np.sum(error[-1]**2 * self.terminal_weights))

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

            pass
            cost = 0

            # --------------------------------------------------------------
            # R-learning reward
            #
            # Goal:
            #   1. reward progress toward target
            #   2. penalize excessive net control action
            #   3. mildly penalize actual commanded control effort
            #
            # Since:
            #
            #       x+ = A x + B (u + d)
            #
            # and d_hat ~= d_true, then:
            #
            #       u + d_hat
            #
            # approximates the nominal control action seen by the plant after
            # disturbance compensation.
            # --------------------------------------------------------------

            '''
            x_start = np.asarray(self.x_start, dtype=float).reshape(-1)
            x_end   = np.asarray(actual[-1], dtype=float).reshape(-1)

            u       = np.asarray(self.u_applied, dtype=float).reshape(-1)
            d_hat   = np.asarray(self.d_hat, dtype=float).reshape(-1)

            # state cost before and after this CALA trial
            V_start = float(np.sum(self.state_weights * x_start**2))
            V_end   = float(np.sum(self.state_weights * x_end**2))

            # positive if state improved
            progress = V_start - V_end

            # effective nominal plant input after disturbance cancellation
            u_net = u + d_hat

            # fixed evaluation costs -- do NOT use the CALA-adjusted R here
            net_effort     = float(np.dot(u_net, u_net))
            command_effort = float(np.dot(u, u))

            # reward weights
            lambda_net = 0.01
            lambda_u   = 0.001

            # convert to cost because reward = -cost below
            cost = (
                -progress
                + lambda_net * net_effort
                + lambda_u * command_effort
            )
            '''

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

    # plot useful CALA learning results (note: this autogenerated by ChatGPT 5.5 with minor revisions)
    def plot_learning(self, folder="visualization/cala"):

        import os
        os.makedirs(folder, exist_ok=True)

        if len(self.history["reward"]) == 0:
            print("No completed CALA trials to plot.")
            return

        steps = np.asarray(self.history["step"])
        rewards = np.asarray(self.history["reward"])
        advantages = np.asarray(self.history["advantage"])
        prediction_errors = np.asarray(self.history["prediction_error"])
        terminal_errors = np.asarray(self.history["terminal_error"])
        rl_adjustment = np.asarray(self.history["rl_adjustment"], dtype=float).reshape(-1, self.n_inputs)
        d_true = np.asarray(self.history["d_true"], dtype=float).reshape(-1, self.n_inputs)
        d_hat  = np.asarray(self.history["d_hat"],  dtype=float).reshape(-1, self.n_inputs)
        rl_mean = np.asarray(self.history["rl_mean"], dtype=float).reshape(-1, self.n_inputs)

        # linear disturbance rejection error
        error_linear = (d_true - d_hat)**2
        # learned CALA mean added to linear rejection
        error_combinerl_mean = (d_true - d_hat - rl_mean)**2
        # actual sampled CALA correction used during training
        #error_combined_sample = d_true - d_hat - rl_adjustment


        sigma = np.asarray(self.history["sigma"])

        # reward and advantage history
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(steps, rewards, marker="o", label="Reward")
        ax.plot(steps, advantages, marker="o", label="Advantage")
        ax.set_title("CALA learning history")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("Reward")
        ax.grid(True)
        ax.legend()
        fig.savefig(f"{folder}/reward_history.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # reward components
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(steps, prediction_errors, marker="o", label="Prediction error")
        ax.plot(steps, terminal_errors, marker="o", label="Terminal error")
        ax.set_title("CALA reward components")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("Weighted squared error")
        ax.grid(True)
        ax.legend()
        fig.savefig(f"{folder}/reward_components.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # correction history
        fig, ax = plt.subplots(figsize=(9, 6))
        for j in range(self.n_inputs):
            ax.plot(steps, rl_adjustment[:, j], marker="o", label=f"$d_{{cala,{j}}}$")
        ax.axhline(0.0, linestyle="--")
        ax.set_title("CALA correction history")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("Disturbance correction")
        ax.grid(True)
        ax.legend()
        fig.savefig(f"{folder}/correction_history.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # empirical reward surface / scatter in correction space
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

            #ax.plot(rl_adjustment[:, 0], rl_adjustment[:, 1], linestyle="--", alpha=0.5)
            ax.scatter(rl_adjustment[-1, 0], rl_adjustment[-1, 1], marker="x", s=100, label="Latest trial")
            ax.set_title("Empirical reward over CALA corrections")
            ax.set_xlabel("$d_{cala,0}$")
            ax.set_ylabel("$d_{cala,1}$")
            ax.set_xlim(-self.cala.d_max, self.cala.d_max)
            ax.set_ylim(-self.cala.d_max, self.cala.d_max)
            ax.set_aspect("equal")
            ax.grid(True)
            ax.legend()
            fig.savefig(f"{folder}/reward_surface.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        # exploration history
        sigma_mean = np.mean(sigma, axis=(1, 2))
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(steps, sigma_mean, marker="o")
        ax.set_title("CALA exploration level")
        ax.set_xlabel("Trial start time")
        ax.set_ylabel("Mean sigma")
        ax.grid(True)
        fig.savefig(f"{folder}/exploration_history.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"CALA plots saved to: {folder}")

        # compare d's 
        #d_target = d_true / (1.0 + self.compensation_weight)
        fig, axes = plt.subplots(
            self.n_inputs,
            1,
            figsize=(9, 3.5 * self.n_inputs),
            sharex=True
        )
        if self.n_inputs == 1:
            axes = [axes]
        for j, ax in enumerate(axes):

            # # actual learned / sampled CALA correction
            # ax.plot(
            #     steps,
            #     rl_adjustment[:, j],
            #     marker="o",
            #     markersize=3,
            #     label=f"$d_{{cala,{j}}}$"
            # )

            # linear-assumed disturbance
            ax.plot(
                steps,
                d_hat[:, j],
                linestyle="-",
                label=f"$d_{{hat,{j}}}$"
            )

            # true disturbance
            ax.plot(
                steps,
                d_true[:, j],
                linestyle=":",
                label=f"$d_{{true,{j}}}$"
            )
            # theoretically optimal compensation
            ax.plot(
                steps,
                rl_mean[:, j],
                linestyle="--",
                label=f"$d_{{mean,{j}}}$"
            )

            # # linear-assumed disturbance
            # ax.plot(
            #     steps,
            #     error_linear[:, j],
            #     linestyle="-",
            #     label=f"just dhat"
            # )

            # # true disturbance
            # ax.plot(
            #     steps,
            #     error_combined_mean[:,j],
            #     linestyle=":",
            #     label=f"combined"
            # )

            #ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.set_ylabel("Disturbance")
            #ax.set_ylabel("Error")
            #ax.set_ylim([0, 0.5])
            ax.grid(True)
            ax.legend()
        axes[-1].set_xlabel("Trial start time")
        fig.suptitle("CALA compensation tracking")
        fig.tight_layout()
        fig.savefig(
            f"{folder}/compensation_tracking.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(fig)




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
