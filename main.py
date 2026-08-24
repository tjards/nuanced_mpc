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
import os


# ------------------------------------------------------------------
# Pipeline Setup
# ------------------------------------------------------------------ 
pipeline = {
    'model':        False,
    'control':      True,
    'rl_train':     True,
    'rl_evaluate':  True,
    'visuals':      True
}

# main function
def main(configs_base = 'configs/default'):

    # ------------------------------------------------------------------
    # Config location
    # ------------------------------------------------------------------

    configs_base = str(configs_base).rstrip('/')
    print(f'pulling configs from: {configs_base}')

    # ------------------------------------------------------------------
    # Initialize plant and data
    # ------------------------------------------------------------------

    # initialize the global timer
    t = 0.0

    # initialize plant
    plant   = le_plant.Plant(configs_base = configs_base)
    x       = plant.x0.copy()

    # initialize disturbances
    disturbor   = disturbance_generator.Disturbance(configs_base = configs_base, field = None)
    d           = disturbor.evolve(field = None, x = x, t = t)

    # initial dataset
    with open(f'{configs_base}/config_data.json') as f:
        cfg_dat = json.load(f)
    data = Dataset(filepath=cfg_dat["filepath"], overwrite=cfg_dat["overwrite"])

    # ------------------------------------------------------------------
    # Model dynamics by exciting plant modes (no disturbances)
    # ------------------------------------------------------------------
    if pipeline['model']:

        # initialize modeller 
        modeller = mpc.Modeller(configs_base = configs_base)

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
        data_defaults   = Dataset(filepath=cfg_dat["defaults"], overwrite=False)
        modelling_data  = data_defaults.read('modelling')

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
    field       = nonlinear_field.VortexField(configs_base = configs_base)
    disturbor   = disturbance_generator.Disturbance(configs_base = configs_base, field = field, x = x, t = t)

    # ------------------------------------------------------------------
    # Create target
    # ------------------------------------------------------------------

    # initial target
    target  = le_target.Target(configs_base = configs_base)
    xr      = target.evolve(t)

    # ------------------------------------------------------------------
    # Run the Controller 
    # ------------------------------------------------------------------
    if pipeline['control']:

        # initialize the MPC controller and load params f
        controller = mpc.MPC(configs_base, x - xr)  # controller uses reference frame with xr at center 

        # ------------------------------------------------------------------
        # Initialize RL for disturbances   
        # ------------------------------------------------------------------
        if pipeline['rl_train']:
            cala_horizon_manager = cala.cala_suite(configs_base = configs_base, controller = controller, data_path=cfg_dat["rl_training_path"], overwrite=True)

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
            if pipeline['rl_train']:
                rl_adjustment = cala.pre_controller(cala_horizon_manager, x, t, x_error = x - xr)
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
            if pipeline['rl_train']:
                cala_horizon_manager.u_trial.append(np.asarray(u).reshape(-1).copy())
                cala_horizon_manager.d_hat_trial.append(np.asarray(controller.d_hat).reshape(-1).copy())
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


    # ------------------------------------------------------
    # Compare rl-learned R with benchmark R
    # -------------------------------------------------------
    if pipeline['rl_evaluate']:

        # --------------------------------------------------
        # evaluation configuration
        # --------------------------------------------------

        cala_eval_Tf    = 200.0
        t_eval          = t

        #eval_data = Dataset(filepath="data/cala/evaluation.h5",overwrite=True)
        eval_data = Dataset(filepath=cfg_dat["rl_evaluation_path"],overwrite=True)

        # --------------------------------------------------
        # create identical starting conditions
        # --------------------------------------------------

        plant_learned   = le_plant.Plant(configs_base = configs_base)
        plant_benchmark = le_plant.Plant(configs_base = configs_base)

        x_learned       = plant_learned.x0.copy()
        x_benchmark     = plant_benchmark.x0.copy()

        target_eval     = le_target.Target(configs_base = configs_base)
        xr_eval         = target_eval.evolve(t_eval)

        dist_learned = disturbance_generator.Disturbance(configs_base = configs_base, field=field,x=x_learned,t=t_eval)
        dist_benchmark = disturbance_generator.Disturbance(configs_base = configs_base, field=field,x=x_benchmark,t=t_eval)

        # --------------------------------------------------
        # independent MPCs
        # --------------------------------------------------

        controller_learned = mpc.MPC(configs_base, x_learned - xr_eval)
        controller_benchmark = mpc.MPC(configs_base, x_benchmark - xr_eval)

        # apply same learned plant model used in main simulation
        for ctrl in [controller_learned, controller_benchmark]:

            if ctrl.use_learned_model:
                ctrl.A = A_hat.copy()
                ctrl.B = B_hat.copy()
                ctrl.new_model_parameters = True

        u_learned   = controller_learned.u0.copy().flatten()
        u_benchmark = controller_benchmark.u0.copy().flatten()

        # --------------------------------------------------
        # load frozen CALA R-policy from training
        # --------------------------------------------------

        cala_eval = cala.cala_suite(configs_base = configs_base, controller = controller_learned, data_path=cfg_dat["rl_training_path"],overwrite=False)
        cala_eval.load_policy()

        # --------------------------------------------------
        # evaluation rollout
        # --------------------------------------------------

        for k in range(int(cala_eval_Tf / controller_learned.Ts)):

            # target corresponding to current instant
            xr_eval = target_eval.evolve(t_eval)

            # ----------------------------------------------
            # actual disturbances
            # ----------------------------------------------

            d_learned = dist_learned.evolve(field=field,x=x_learned,t=t_eval)
            d_benchmark = dist_benchmark.evolve(field=field,x=x_benchmark,t=t_eval)

            # ----------------------------------------------
            # RL-LEARNED R
            # ----------------------------------------------

            # build feature activation at current learned-R state
            phi = cala_eval.feature_map.build_features(x_learned,t_eval)

            # exploit learned mean policy -- NO exploration
            rl_R_adjustment = cala_eval.cala.get_correction(phi)
            controller_learned.solve(x_learned - xr_eval,u_learned,rl_adjustment=rl_R_adjustment)
            u_learned = (controller_learned.result_control_next.flatten())
            x_learned = plant_learned.evolve(x_learned,u_learned,d_learned,disturb=True)

            # ----------------------------------------------
            # BENCHMARK R
            # ----------------------------------------------

            controller_benchmark.solve(x_benchmark - xr_eval,u_benchmark,rl_adjustment=None)
            u_benchmark = (controller_benchmark.result_control_next.flatten())
            x_benchmark = plant_benchmark.evolve(x_benchmark,u_benchmark,d_benchmark,disturb=True)

            # ----------------------------------------------
            # next target / time
            # ----------------------------------------------

            t_next = t_eval + controller_learned.Ts
            xr_next = target_eval.evolve(t_next)

            # ----------------------------------------------
            # store learned-R result
            # ----------------------------------------------

            eval_data.stage(
                phase="rl_learned_R",
                step=t_next,
                A_hat=controller_learned.A,
                B_hat=controller_learned.B,
                d_hat=controller_learned.d_hat,
                d=plant_learned.d,
                target=xr_next,
                state=x_learned,
                input=u_learned,
            )
            eval_data.store(flush_after=True)

            # ----------------------------------------------
            # store benchmark result
            # ----------------------------------------------

            eval_data.stage(
                phase="benchmark_R",
                step=t_next,
                A_hat=controller_benchmark.A,
                B_hat=controller_benchmark.B,
                d_hat=controller_benchmark.d_hat,
                d=plant_benchmark.d,
                target=xr_next,
                state=x_benchmark,
                input=u_benchmark,
            )
            eval_data.store(flush_after=True)

            t_eval = t_next
            t += controller.Ts

        # pull back from disk
        learned_R_data = eval_data.read("rl_learned_R")
        benchmark_R_data = eval_data.read("benchmark_R")



    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    if pipeline['visuals']:

        # pull visualization configs 
        with open(f'{configs_base}/config_visualization.json') as f:
            cfg_viz = json.load(f)

        # general viz configs
        do_plots                = cfg_viz['do_plots']
        do_animate              = cfg_viz['do_animate']
        show_field              = cfg_viz['show_field']
        keep_modelling_history  = cfg_viz['keep_modelling_history']

        # define paths
        animate_path            = cfg_viz['animate_path']
        plot_inputs_path        = cfg_viz['plot_inputs_path']
        plot_velocities_path    = cfg_viz['plot_velocities_path']
        plot_trajectory_path    = cfg_viz['plot_trajectory_path']

        # make folders
        training_folder         = cfg_viz['training_folder']
        os.makedirs(training_folder, exist_ok=True)
        evaluation_folder       = cfg_viz['evaluation_folder']
        os.makedirs(evaluation_folder, exist_ok=True)
        control_folder          = cfg_viz['control_folder']
        os.makedirs(control_folder, exist_ok=True)

        with open(f'{configs_base}/config_mpc.json') as f:
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

        if do_plots:
            print('Producing plots...')
            plot.plot_inputs(time_history, full_input_history, constraints, filename=plot_inputs_path)
            plot.plot_velocities(time_history, full_state_history, constraints, filename=plot_velocities_path)
            plot.plot_trajectory(controller_data['step'],controller_data['state'], x_target=controller_data['target'], filename=plot_trajectory_path)
            if pipeline['rl_train']:
                cala_horizon_manager.plot_learning(folder=training_folder)
                cala_horizon_manager.cala.plot_correction(t=0.0, resolution=200, folder = training_folder)
            if pipeline['rl_evaluate']:
                eval_data           = Dataset(filepath=cfg_dat["rl_evaluation_path"], overwrite=False)
                learned_R_data      = eval_data.read("rl_learned_R")
                benchmark_R_data    = eval_data.read("benchmark_R")
                plot.plot_R_evaluation(learned_R_data, benchmark_R_data, filename=f"{evaluation_folder}/R_evaluation.png")
                plot.plot_R_evaluation_cum(learned_R_data, benchmark_R_data, R0=None, filename = f"{evaluation_folder}/R_evaluation_cum.png")
                #plot.plot_R_evaluation_pareto(learned_R_data,benchmark_R_data,R0=None,filename=os.path.join(evaluation_folder, "R_evaluation_pareto.png"))
            
        if do_animate:
            print('Producing animation...')
            if show_field and disturbor.dist_type == 'field':
                field_in = field
            else:
                field_in = None
            plot.animate_trajectory(time_history, full_state_history, predicted_sequences, x_target = target_history, field = field_in, filename=animate_path)
            # old (keep for now)
            #plot.animate_trajectory(full_state_history, predicted_sequences, solve_discrete_are(controller.A, controller.B, controller.Q, controller.R),filename=animate_path)

if __name__ == "__main__":

    main(configs_base = 'configs/default')

