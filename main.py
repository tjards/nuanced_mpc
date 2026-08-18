# standard inputs
import numpy as np
import json
from scipy.linalg import solve_discrete_are

# custom imports
import plant as le_plant 
import target as le_target
import disturbance_generator 
import mpc 
import visualization.plot as plot
from data_manager import Dataset
import cala

# ------------------------------------------------------------------
# Pipeline Setup
# ------------------------------------------------------------------ 
pipeline = {
    'model':    False,
    'control':  True,
    'rl':       True,
    'visuals':  True
}
# ------------------------------------------------------------------
# Initialize plant and data
# ------------------------------------------------------------------

# initialize the global timer
t = 0.0

# initialize plant
plant = le_plant.Plant()
x = plant.x0.copy()

# initialize disturbances
disturbor = disturbance_generator.Disturbance(field = None)
d = disturbor.evolve(field = None, x = x, t = t)

# initial dataset
with open('configs/config_data.json') as f:
    cfg_dat = json.load(f)
data = Dataset(filepath=cfg_dat["filepath"], overwrite=cfg_dat["overwrite"])

# ------------------------------------------------------------------
# Model dynamics by exciting plant modes (no disturbances)
# ------------------------------------------------------------------
if pipeline['model']:

    # initialize modeller 
    modeller = mpc.Modeller()

    print(f'Modelling started at time: {round(t)} seconds')
    print(f"Exciting the plant modes for modelling...")

    x, t, state_history, input_history, A_hat_history, B_hat_history, step_history = modeller.excite(plant, disturbor, x, t)

    print(f'Modelling completed at time: {round(t)} seconds')

    data.stage(phase = 'modelling', 
            step = step_history, 
            A_hat = A_hat_history, 
            B_hat = B_hat_history, 
            state = state_history, 
            input = input_history,)
    data.store()

    print(f"Modelling complete. Model viable: {modeller.viable}")

    # pull new data
    modelling_data          = data.read('modelling') 

else:

    # pull data from defaults
    data_defaults = Dataset(filepath=cfg_dat["defaults"], overwrite=False)
    modelling_data          = data_defaults.read('modelling')

# extract what we need later 
epsilon = 1e-5
A_hat = modelling_data['A_hat'][-1]
A_hat[A_hat < epsilon] = 0.0
B_hat = modelling_data['B_hat'][-1]
B_hat[B_hat < epsilon] = 0.0

# ------------------------------------------------------------------
# Create disturbance field 
# ------------------------------------------------------------------
import nonlinear_field
field = nonlinear_field.VortexField()
disturbor = disturbance_generator.Disturbance(field = field, x = x, t = t)

# ------------------------------------------------------------------
# Create target
# ------------------------------------------------------------------

# initial target
target  = le_target.Target()
xr      = target.evolve(t)

# ------------------------------------------------------------------
# Run the Controller 
# ------------------------------------------------------------------
if pipeline['control']:

   
    # initialize the MPC controller and load params f
    controller = mpc.MPC(x - xr)  # controller uses reference frame with xr at center 

    # ------------------------------------------------------------------
    # Initialize RL for disturbances   
    # ------------------------------------------------------------------
    if pipeline['rl']:
        cala_horizon_manager = cala.cala_suite(controller)
    rl_adjustment = np.zeros(controller.nu)

    if controller.use_learned_model:
        controller.A = A_hat    #modeller.A_hat
        controller.B = B_hat    #modeller.B_hat
        controller.new_model_parameters = True
    else:
        print('using first-principles model')

    # check feasibility of the current state and input
    controller.confirm_feasibility(x - xr, controller.u0) # controller uses reference frame with xr at center 

    # initialize the control input
    u = controller.u0

    print(f'Controller started at time: {round(t)} seconds')

    for k in range(int(controller.Tf / controller.Ts)):

        # evolve the disturbance
        d = disturbor.evolve(field = field, x = x, t = t)


        # begin cala trial
        if pipeline['rl']:
            rl_adjustment = cala.pre_controller(cala_horizon_manager, x, t)
            cala_horizon_manager.d_true = d.copy()
        else:
            rl_adjustment = None

        # run controller
        #controller.solve(x - xr, u)  # controller uses reference frame with xr at center 
        controller.solve(x - xr, u, rl_adjustment=rl_adjustment)

        # store predicted sequence 
        current_plan = controller.result_state_sequence.reshape(controller.h,controller.nx,).copy()
        # reference shift 
        current_plan += xr

        # apply first control input 
        u = controller.result_control_next.flatten() 

        # evolve the disturbance
        #d = disturbor.evolve(field = field, x = x, t = t)

        # evolve the plant
        x = plant.evolve(x, u, d, disturb=True)

        # evolve target
        #xr = target.evolve(t)
        xr = target.evolve(t+controller.Ts)

        # update cala trial
        if pipeline['rl']:
            predicted_reference = controller.result_state_sequence.reshape(controller.h, controller.nx).copy()
            cala_horizon_manager.d_hat = controller.d_hat.copy()
            cala.post_controller(cala_horizon_manager, predicted_reference, x, xr)

        data.stage(phase = 'controller', 
                step = t, 
                A_hat = controller.A, 
                B_hat = controller.B, 
                d_hat = controller.d_hat, 
                d = plant.d, 
                target = target.evolve(t), 
                state = x, 
                input = u, 
                plan = current_plan)
        data.store(flush_after=True)

        # increment the global timer
        t += controller.Ts

    print(f'Controller completed at time: {round(t)} seconds')
    print(f"Final state distance from goal: {np.linalg.norm(x - xr):.4f}")

    controller_data         = data.read('controller')

else:

    data_defaults = Dataset(filepath=cfg_dat["defaults"], overwrite=False)
    controller_data         = data_defaults.read('controller')

# ------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------
if pipeline['visuals']:

    # pull visualization configs 
    with open('configs/config_visualization.json') as f:
        cfg_viz = json.load(f)

    animate_path            = cfg_viz['animate_path']
    plot_inputs_path        = cfg_viz['plot_inputs_path']
    plot_velocities_path    = cfg_viz['plot_velocities_path']
    keep_modelling_history  = cfg_viz['keep_modelling_history']
    show_field              = cfg_viz['show_field']

    with open('configs/config_mpc.json') as f:
        cfg_mpc = json.load(f)
    constraints = cfg_mpc['constraints']


    if keep_modelling_history:
        full_state_history      = list(modelling_data["state"]) + list(controller_data["state"])[1:]  # avoid duplicate x0
        full_input_history      = list(modelling_data["input"]) + list(controller_data["input"])
        predicted_sequences     = [None] * len(modelling_data["state"]) + list(controller_data["plan"])
        time_history            = list(modelling_data["step"]) + list(controller_data["step"])
        target_history          = [None] * len(modelling_data["state"]) + list(controller_data["target"])
    else:
        full_state_history      = list(controller_data["state"])[1:]  
        full_input_history      = list(controller_data["input"])
        predicted_sequences     = list(controller_data["plan"])
        time_history            = list(controller_data["step"])
        target_history            = list(controller_data["target"]) 



    print('Producing plots...')
    plot.plot_inputs(time_history, full_input_history, constraints, filename=plot_inputs_path)
    plot.plot_velocities(time_history, full_state_history, constraints, filename=plot_velocities_path)
    plot.plot_trajectory(controller_data['step'],controller_data['state'],x_target=controller_data['target'],filename='visualization/plots/trajectory.png')


    #plot.animate_trajectory(full_state_history, predicted_sequences, solve_discrete_are(controller.A, controller.B, controller.Q, controller.R),filename=animate_path)
    
    # temp: this data will need to be stored before plotting (i.e., don't plot from memory)
    if pipeline['rl']:
        cala_horizon_manager.plot_learning()
    #    cala_horizon_manager.cala.plot_correction(t=0.0, resolution=200)
    #     cala_horizon_manager.cala.plot_correction(t=5.0, resolution=100)
    #     cala_horizon_manager.cala.plot_correction(t=10.0, resolution=100)
    #     cala_horizon_manager.cala.plot_correction(t=15.0, resolution=100)


    print('Producing animation...')

    if show_field and disturbor.dist_type == 'field':
        field_in = field
    else:
        field_in = None

    #plot.animate_trajectory(time_history, full_state_history, predicted_sequences, x_target = target_history, field = field_in, filename=animate_path)
