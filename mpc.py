'''
Implementation of Convex Data-based Predictive Control (Convex DPC) with:

-  data-driven modelling 
-  disturbance rejection


Standard Form:

x(k+1) = Ax(k) + B (u(k) + d) 
J = sum_{i=0}^{h-1} x(k+i)'Qx(k+i) + u(k+i)'Ru(k+i) + x(k+h)'Px(k+h)

Constraints:

    x_min <= x(k) <= x_max
    u_min <= u(k) <= u_max

    or 

    Mx <= bx
    Mu <= bu

'''

import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are
import json


# online modeller
class Modeller():
    
    def __init__(self, configs_base):

        with open(f'{configs_base}/config_mpc.json') as f:
            cfg = json.load(f)

        A   = 0*np.array(cfg['A'])
        B   = 0*np.array(cfg['B'])  

        # assign 
        self.A_hat  = np.array(A, ndmin=2)
        self.B_hat  = np.array(B, ndmin=2)
        self.nx     = int(A.shape[0])
        self.nu     = int(B.shape[1])
        
        self.constraints    = cfg['constraints'] #constraints
        self.window_size    = cfg['window_size'] #int(window_size)        
        self.Phi            = np.zeros((self.nx, self.nx + self.nu))  
        self.learning_rate  = cfg['learning_rate'] #learning_rate
        self.optimizer      = cfg['optimizer'] #optimizer
        self.random_seed    = cfg['random_seed'] #random_seed
        
        self.update_parameters_rate     = int(cfg['update_parameters_rate']) 
        self.update_parameters_count    = 0

        self.X = np.zeros((self.nx, self.window_size))
        self.U = np.zeros((self.nu, self.window_size))

        self.scale_x  = np.ones((self.nx, 1))
        self.scale_u  = np.ones((self.nu, 1))
        self.scale_dx = np.ones((self.nx, 1))

        self.viable = False

        self.Ts = cfg['Ts']  #Ts
        self.u0 = np.array(cfg['u0'], dtype=float) #u0

        #self.logging = logging 

    def excite(self, plant, disturbor, x, t, state_history =[], input_history =[], A_hat_history =[], B_hat_history =[], step_history =[]):

        # t is initial time 

        # bounds
        u_max          = np.array(self.constraints['u_max'])
        u_min          = np.array(self.constraints['u_min'])

        # excitation parameters
        excite_hold         = 0.3                       # seconds to hold each random input
        excite_time         = 15
        excite_hold_steps   = int(excite_hold / self.Ts)              
        excite_max_steps    = int(excite_time / self.Ts) # excitation duration 
        
        # initialize excitation 
        rng             = np.random.default_rng(seed=self.random_seed)
        u_exc           = np.zeros_like(u_min)    # random input 
        excite_count    = excite_hold_steps       # trigger switch immediately 
        rockback        = False                   # set to False initially, avoids biasing the model in one direction
        k               = 0

        self.excite_time = excite_time

        # initialize log
        #with open('logs/model_log.txt', 'w') as model_log:
        #    model_log.write('step,A_hat,B_hat\n')

        # excite the modes of the plant for a while 
        while k < excite_max_steps:

            # determine if rockback needed 
            if excite_count >= excite_hold_steps and not rockback:
                excite_count = 0
                u_exc = rng.uniform(u_min, u_max)
                rockback = True
            elif excite_count >= excite_hold_steps and rockback:
                excite_count = 0
                u_exc = -u_exc
                rockback = False
            excite_count += 1

            # update the model
            self.update(x, u_exc)

            # evolve the disturbance
            d = disturbor.evolve(field = None, x = x, t = k*self.Ts)

            # evolve the plant
            x = plant.evolve(x, u_exc, d, disturb = False)
                    
            # store
            state_history.append(x.copy())
            input_history.append(u_exc.copy())
            A_hat_history.append(self.A_hat.tolist())
            B_hat_history.append(self.B_hat.tolist())
            step_history.append(t)

            # log
            #model_log.write(f'exc_{k + 1},{self.A_hat.tolist()},{self.B_hat.tolist()}\n')
            
            k += 1
            t += self.Ts
        
        return x, t, state_history, input_history, A_hat_history, B_hat_history, step_history

    def update(self, x, u):

        self.accumulate_data(x, u)
        self.update_parameters_count += 1
        if self.update_parameters_count >= self.update_parameters_rate:
            self.fit()
            self.update_parameters_count = 0

    # accumulate data over window 
    def accumulate_data(self, x, u):

        # shift data to the left and add new data to the right
        self.X[:, :-1] = self.X[:, 1:]
        self.U[:, :-1] = self.U[:, 1:]
        self.X[:, -1] = x.flatten()
        self.U[:, -1] = u.flatten()

    # fit the residuals: dx = x(k+1) - x(k) = A_tilde*x(k) + B*u(k) + bias               
    def fit(self):

        if self.optimizer == 'least_squares':

            # pull out states and inputs from the window
            X_curr = self.X[:, :-1]
            U_curr = self.U[:, :-1]
            X_next = self.X[:, 1:]
            N = X_curr.shape[1]

            # factor out identity
            dX = X_next - X_curr                                                

            # scales (per feature) for normalization 
            self.scale_x  = np.maximum(np.std(X_curr, axis=1, keepdims=True), 1e-8)  
            self.scale_u  = np.maximum(np.std(U_curr, axis=1, keepdims=True), 1e-8)  
            self.scale_dx = np.maximum(np.std(dX,     axis=1, keepdims=True), 1e-8)  

            # normalize
            X_n  = X_curr / self.scale_x
            U_n  = U_curr / self.scale_u
            dX_n = dX     / self.scale_dx

            # stack 
            Z_n = np.vstack([X_n, U_n, np.ones((1, N))])

            # solve dX_n ≈ Phi_n * Z_n using least squares (normalized)
            Phi_n, _, _, _ = np.linalg.lstsq(Z_n.T, dX_n.T, rcond=None)
            Phi_n = Phi_n.T                                                     

            # un-normalize
            S_dx = np.diag(self.scale_dx.flatten())
            A_tilde = S_dx @ Phi_n[:, :self.nx] @ np.diag(1.0 / self.scale_x.flatten())
            B_new   = S_dx @ Phi_n[:, self.nx:self.nx+self.nu] @ np.diag(1.0 / self.scale_u.flatten())

            # add back the identity
            A_new = np.eye(self.nx) + A_tilde

            # apply to model using learning rate
            self.A_hat = (1-self.learning_rate) * self.A_hat + self.learning_rate * A_new
            self.B_hat = (1-self.learning_rate) * self.B_hat + self.learning_rate * B_new

            # check stability 
            is_stable = np.all(np.abs(np.linalg.eigvals(self.A_hat)) <= 1.0)

            # check controllability
            C = np.hstack([np.linalg.matrix_power(self.A_hat, i) @ self.B_hat for i in range(self.nx)])
            is_controllable = np.linalg.matrix_rank(C) == self.nx

            # if both stable and controllable, consider viable
            if is_stable and is_controllable:
                self.viable = True
                #print('stable and controllable model found')
            else:
                self.viable = False

        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

# controller
class MPC():

    def __init__(self, configs_base, x0):

        with open(f'{configs_base}/config_mpc.json') as f:
            cfg = json.load(f)
        
        # model
        self.A = np.array(np.array(cfg['A']), ndmin=2)           # state matrix  
        self.B = np.array(np.array(cfg['B']), ndmin=2)           # input matrix
        self.Q = np.array(np.diag(cfg['Q_diag']), ndmin=2)       # state cost matrix
        self.R = np.array(np.diag(cfg['R_diag']), ndmin=2)       # input cost matrix

        # we can compute terminal cost based on solution to Discrete Algebraic Riccati Equation
        P   = cfg['P_diag']
        self.P_cfg = cfg['P_diag']                            # nominally 'dare', but takes hardcoded matrix   
        if isinstance(P, str) and P.lower() == 'dare':
            self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)  
        # or hardcoded
        else:
            self.P = np.array(P, ndmin=2)       # terminal

        self.x0 = np.array(x0).reshape(-1, 1)   # initial state
        self.u0 = np.array(np.array(cfg['u0'], dtype=float)).reshape(-1, 1)   # initial input
        self.h = cfg['h']                       # prediction horizon
        self.m = cfg['m']                       # control horizon (inputs freeze after m <= h)
        self.nx = self.A.shape[0]               # state dimensions
        self.nu = self.B.shape[1]               # input dimensions
        self.constraints = cfg['constraints']   # constraints
        self.disturbance = cfg['disturbance']
        self.Ts = cfg['Ts']  
        self.Tf = cfg['Tf'] 

        # for feasibility checks
        self.h_max = 100
        self.terminal_set_radius = 2
        self.h_min_feasible = None

        # for learning
        self.use_learned_model = cfg['use_learned_model']

        # terminal constraints
        self.enforce_terminal = cfg['enforce_terminal']
        self.horizon_feasibility_search = cfg['horizon_feasibility_search']

        # receding horizon config
        self.replan         = True
        self.replan_count   = -1                  
        self.plan_index     = 0
        self.replan_mode    = cfg['replan_mode']
        if self.replan_mode == 'receding_horizon':
            self.replan_trigger = 1
        elif self.replan_mode == 'control_horizon':
            self.replan_trigger = self.m 
        elif self.replan_mode == 'prediction_horizon':
            self.replan_trigger = self.h
        else:
            raise ValueError(f'invalid replan mode: {self.replan_mode}.')

        # RL may tune R, so we will create a separate parameter for this
        self.rl_parameter = cfg['rl_parameter'] # options: None, d (disturbance estimate), R (input weights in objective function)
        self.R0 = np.asarray(self.R).copy()

        # we can do disturbance rejection
        if self.disturbance:
            self.d_hat = np.zeros((self.nu, 1))  # estimated input disturbance
            self.x_prev = None                   # previous measured state
            self.u_prev = None                   # previous commanded input
        else:
            self.d_hat = None
        self.disturbance_enable_linear_rejection = cfg['disturbance_enable_linear_rejection']
        
        self.update_internal_parameters()  
        self.new_model_parameters = False

    # confirm feasible (expands h (but not m) until feasible): untested
    def confirm_feasibility(self, x0, u0):

        if not self.horizon_feasibility_search:
            return None

        x0 = np.array(x0).reshape(-1, 1)
        u0 = np.array(u0).reshape(-1, 1)

        print(f"Checking feasibility of h={self.h}")
        print(f"Terminal region: {self.terminal_set_radius}")

        # we'll be temporarily adjusting horizons, so store originals here
        h_0      = self.h
        m_0      = self.m
        if self.disturbance:
            x_prev_0 = self.x_prev
            u_prev_0 = self.u_prev

        try:
            # incrementally increase h until feasible 
            for h_trial in range(h_0, self.h_max + 1):

                # set new horizons 
                self.h = h_trial
                self.m = min(m_0, h_trial)
                self.new_model_parameters = True

                # see if it solves
                try:
                    self.solve(x0, u0, update_disturbance_estimate = False)
                except RuntimeError:
                    print(f"Horizon h={h_trial}: failed to solve, increasing horizon...")
                    continue

                # pull out the terminal state 
                x_terminal = self.result_state_sequence[-self.nx:].reshape(-1, 1)
                
                # compute distance 
                dist = float(np.linalg.norm(x_terminal))

                # check distance against terminal set radius
                if dist <= self.terminal_set_radius:
                    if h_trial == h_0:
                        print(f"Selected h={h_0} feasible.")
                    else:
                        print(f"Selected h={h_0} infeasible.")
                        print(f"Suggest h={h_trial} with terminal distance {dist:.4f}.")
                    self.h_min_feasible = h_trial
                    return h_trial
                else:
                    if h_trial == h_0:
                        print(f"Selected h={h_0} solves but doesn't reach terminal region. Increasing horizon...")
                    else:
                        print(f"...h={h_trial} doesn't reach terminal region, increasing farther...")

            print(f"Terminal region not reachable within h_max={self.h_max}.")
            return None

        # i want to guarantee these are restored even if exceptions
        finally:
            # restore original horizons 
            self.h = h_0
            self.m = m_0
            self.new_model_parameters = True
            self.update_internal_parameters()
            if self.disturbance:
                self.x_prev = x_prev_0
                self.u_prev = u_prev_0

    # allows for updated parameters
    def update_internal_parameters(self):

        # recompute P if it's set to 'dare' (depends on A, B, Q, R)
        if isinstance(self.P_cfg, str) and self.P_cfg.lower() == 'dare':
            self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self._define_constraints()
        self._augment_matrices()
        self._construct_optimization_problem()

    # constraints can be defined in different ways 
    def _define_constraints(self):

        if self.constraints['type'] == 'box':
            self.x_min = np.array(self.constraints['x_min']).reshape(-1, 1)
            self.x_max = np.array(self.constraints['x_max']).reshape(-1, 1)
            self.u_min = np.array(self.constraints['u_min']).reshape(-1, 1)
            self.u_max = np.array(self.constraints['u_max']).reshape(-1, 1)
        elif self.constraints['type'] == 'lmi':
            self.Mx = np.array(self.constraints['Mx'], ndmin=2)
            self.bx = np.array(self.constraints['bx'], ndmin=2)
            self.Mu = np.array(self.constraints['Mu'], ndmin=2)
            self.bu = np.array(self.constraints['bu'], ndmin=2)
        else:
            raise ValueError("Invalid constraint type: Must be 'box' or 'lmi'")

    # augment all the things 
    def _augment_matrices(self):
        self._build_augmented_system_matrices()
        self._build_augmented_cost_matrices()
        self._build_augmented_constraints()

    def _build_augmented_system_matrices(self):

        # initialize the augmented matrices over prediction horizon
        self.A_aug = np.zeros((self.nx * self.h, self.nx))
        self.B_aug = np.zeros((self.nx * self.h, self.nu * self.h))
        if self.disturbance:
            self.D_aug = np.zeros((self.nx * self.h, self.nu * self.h))
        else:
            self.D_aug = None

        # build the augmented matrices
        for i in range(self.h):
            self.A_aug[i*self.nx:(i+1)*self.nx, :] = np.linalg.matrix_power(self.A, i+1)
            for j in range(i+1):
                self.B_aug[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = np.linalg.matrix_power(self.A, i-j) @ self.B
        if self.disturbance:
            self.D_aug = self.B_aug @ np.tile(np.eye(self.nu), (self.h, 1))

    def _build_augmented_cost_matrices(self):

        # initialize the augmented cost matrices over prediction horizon
        self.Q_aug = np.zeros((self.nx * self.h, self.nx * self.h))
        self.R_aug = np.zeros((self.nu * self.h, self.nu * self.h))

        # build the augmented cost matrices
        for i in range(self.h):
            # up to but not including the terminal step, use Q 
            if i < self.h - 1:
                self.Q_aug[i*self.nx:(i+1)*self.nx, i*self.nx:(i+1)*self.nx] = self.Q
            # at terminal step (i.e., end of prediction horizon), use P
            else:
                self.Q_aug[i*self.nx:(i+1)*self.nx, i*self.nx:(i+1)*self.nx] = self.P

            # R is used for all steps 
            self.R_aug[i*self.nu:(i+1)*self.nu, i*self.nu:(i+1)*self.nu] = self.R
    
    def _build_augmented_constraints(self):
       
        if self.constraints["type"] == "box":
            self.x_min_aug = np.tile(self.x_min, (self.h, 1))
            self.x_max_aug = np.tile(self.x_max, (self.h, 1))
            self.u_min_aug = np.tile(self.u_min, (self.h, 1))
            self.u_max_aug = np.tile(self.u_max, (self.h, 1))

        elif self.constraints["type"] == "lmi":
            self.Mx_aug = np.kron(np.eye(self.h), self.Mx)
            self.bx_aug = np.tile(np.asarray(self.bx).reshape(-1, 1),(self.h, 1),)
            self.Mu_aug = np.kron(np.eye(self.h), self.Mu)
            self.bu_aug = np.tile(np.asarray(self.bu).reshape(-1, 1),(self.h, 1),)

        else:
            raise ValueError(f"Unknown constraint type: {self.constraints['type']}")
        
    def _construct_optimization_problem(self):

        # define the optimization variable (optimized up to m, then will be frozen up to h)
        self.opt_u = cp.Variable((self.nu * self.m, 1))

        # define full input sequence: opt_u up to m + frozen inputs 
        if self.m < self.h:
            u_last = self.opt_u[(self.m - 1) * self.nu : self.m * self.nu]
            self.u_full = cp.vstack([self.opt_u] + [u_last] * (self.h - self.m))
        else:
            self.u_full = self.opt_u

        # define initial state (updated at each solve)
        self.x0_param = cp.Parameter((self.nx, 1), value=self.x0)

        # define disturbance 
        if self.disturbance:
            # disturbance estimated using basic linear assumption
            self.d_hat_param = cp.Parameter((self.nu, 1), value=self.d_hat)
            # disturbance estimate adjusted by rl agent
            self.d_adjustment_param = cp.Parameter((self.nu, 1), value=np.zeros((self.nu, 1)))
            # just the linear offset (comment this out)
            #d_offset = self.D_aug @ self.d_hat_param
            # both the linear offset and the rl-learned adjustment 
            d_offset = self.D_aug @ (self.d_hat_param + self.d_adjustment_param)
        else:
            d_offset = np.zeros((self.nx * self.h, 1))

        # inputs matrix (this needs to be augmented for the full horizon)
        self.R_aug_param = cp.Parameter((self.nu * self.h, self.nu * self.h), PSD=True)
        if self.rl_parameter == 'R':
            _R_aug_val = np.zeros((self.nu * self.h, self.nu * self.h))
            for i in range(self.h):
                _R_aug_val[i*self.nu:(i+1)*self.nu, i*self.nu:(i+1)*self.nu] = self.R0
            self.R_aug_param.value = _R_aug_val
        else:
            self.R_aug_param.value = self.R_aug.copy()

        # define s, which is the predicted state sequence over h
        self.s = cp.Variable((self.nx * self.h, 1))
        x_pred_expr = self.A_aug @ self.x0_param + self.B_aug @ self.u_full + d_offset

        # define the cost function (quadratic over s)
        #self.opt_cost = cp.quad_form(self.s, self.Q_aug) + cp.quad_form(self.u_full, self.R_aug)
        self.opt_cost = cp.quad_form(self.s, self.Q_aug) + cp.quad_form(self.u_full, self.R_aug_param)
        
        # constrain s to the dynamics (this connects s to the predicted state sequence)
        self.opt_constraints = [self.s == x_pred_expr]

        # other constraints 
        if self.constraints["type"] == "box":
            self.opt_constraints += [self.x_min_aug <= self.s]
            self.opt_constraints += [self.s <= self.x_max_aug]
            self.opt_constraints += [self.u_min_aug <= self.u_full]
            self.opt_constraints += [self.u_full <= self.u_max_aug]
        elif self.constraints["type"] == "lmi":
            self.opt_constraints += [self.Mx_aug @ self.s <= self.bx_aug]
            self.opt_constraints += [self.Mu_aug @ self.u_full <= self.bu_aug]
        else:
            raise ValueError(f"Unknown constraint type: {self.constraints['type']}")

        # enforce terminal constraint
        if self.enforce_terminal:

            x_terminal = self.s[-self.nx:, :]
            #self.opt_constraints += [cp.norm(x_terminal, 2) <= self.terminal_set_radius]
            position_terminal = x_terminal[:2, :]
            self.opt_constraints += [cp.norm(position_terminal, 2) <= self.terminal_set_radius]

        # define the optimization problem
        self.prob = cp.Problem(cp.Minimize(self.opt_cost), self.opt_constraints)

    def solve(self, x0, u0, update_disturbance_estimate = True, rl_adjustment = None):

        # 1. Update the parameters
        if self.new_model_parameters:
            self.update_internal_parameters()
            self.new_model_parameters = False  
            self.replan_count = -1 

        # 2. Initialize variables 
        
        # inputs
        self.u0 = np.array(u0).reshape(-1, 1)
        
        # states
        self.x0 = np.array(x0).reshape(-1, 1)
        self.x0_param.value = self.x0
        
        # disturbances
        if self.disturbance and self.x_prev is not None and update_disturbance_estimate:
            prediction_error = self.x0 - self.A @ self.x_prev - self.B @ self.u_prev
            self.d_hat, _, _, _ = np.linalg.lstsq(self.B, prediction_error, rcond=None)

        if self.disturbance:
            if self.disturbance_enable_linear_rejection:
                self.d_hat_param.value = self.d_hat.copy()
            else:
                self.d_hat_param.value = np.zeros((self.nu, 1))




        # if self.disturbance:
        #     self.d_hat_param.value = self.d_hat
        #     # if provided, apply the rl-learned adjustment
        #     if rl_adjustment is not None and self.rl_parameter == 'd':
        #         self.d_adjustment_param.value = np.asarray(rl_adjustment).reshape(-1, 1)
        #     else:
        #         self.d_adjustment_param.value = np.zeros((self.nu, 1))

        # 2.b. Reinforce learning part

        # initialize
        if self.disturbance:
            self.d_adjustment_param.value = np.zeros((self.nu, 1)) # we keep a zero'd param either way
        self.R_aug_param.value = self.R_aug

        # use the rl_adjustment based on config
        if rl_adjustment is not None:

            if self.rl_parameter == 'd' and self.disturbance:

                self.d_adjustment_param.value = np.asarray(rl_adjustment).reshape(-1, 1)

            elif self.rl_parameter == 'R':

                R_scale = np.exp(np.asarray(rl_adjustment).reshape(-1))   # shape (nu,)
                R_diag  = np.diag(self.R0) * R_scale                      # shape (nu,), element-wise
                _R_aug_val = np.zeros((self.nu * self.h, self.nu * self.h))
                for i in range(self.h):
                    _R_aug_val[i*self.nu:(i+1)*self.nu, i*self.nu:(i+1)*self.nu] = np.diag(R_diag)
                self.R_aug_param.value = _R_aug_val

            elif self.rl_parameter not in ('d', 'R', None):

                raise ValueError(f"Unknown RL parameter {self.rl_parameter}. Choose None if no RL.")



        # 3. Solve the optimization problem (if triggered)
        self.replan = (self.replan_count < 0 or self.replan_count >= self.replan_trigger)

        if self.replan:
        
            self.prob.solve()
            if self.prob.status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError(f"MPC solver failed: {self.prob.status}")
        
            # results
            self.result_control_sequence    = self.u_full.value.copy()
            self.result_state_sequence      = self.s.value.copy()
            self.result_control_next        = self.result_control_sequence[:self.nu].copy()

            # reset the planner count
            self.replan_count = 1
            self.plan_index = 0

        else:

            _input_start = self.replan_count * self.nu
            _input_end = _input_start + self.nu
        
            self.result_control_next = self.result_control_sequence[_input_start:_input_end].copy()
            self.plan_index = self.replan_count
            self.replan_count += 1

        # 4. Store state and commanded input for next disturbance update
        if self.disturbance and update_disturbance_estimate:
            self.x_prev = self.x0.copy()
            self.u_prev = self.result_control_next.copy()

  


