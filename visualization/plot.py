'''

This module mostly vibe-coded using Claude Sonnet 4.6

Minor refinements made by author: tjards

'''

# standard imports 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os


# helper function for contour plotting
#def _cost_grid(X, Y, V):
#    pts = np.stack([X, Y], axis=-1)   
#    V2  = V[:2, :2]                   
#    return np.einsum('...i,ij,...j->...', pts, V2, pts)


def animate_trajectory(time_history, state_history, predicted_sequences, field = None,
                       x_target=None, filename='trajectory.gif'):
    
    
    times   = np.array(time_history)
    states  = np.array(state_history)        
    T       = len(predicted_sequences)      
    #nx     = V.shape[0]
    nx      = states.shape[0]

    if x_target is None:
        x_target = np.zeros(nx)
    #x_target = np.asarray(x_target, dtype=float)
    x_target = np.array(x_target)

    # ------------------------------------------------------------------
    # Grid bounds: cover all actual + predicted states with margin
    # ------------------------------------------------------------------
    all_pts = np.vstack([states[:, :2]] + [p[:, :2] for p in predicted_sequences if p is not None])
    margin  = 0.5

    #x1_min = min(all_pts[:, 0].min(), x_target[0]) - margin
    #x1_max = max(all_pts[:, 0].max(), x_target[0]) + margin
    #x2_min = min(all_pts[:, 1].min(), x_target[1]) - margin
    #x2_max = max(all_pts[:, 1].max(), x_target[1]) + margin

    x1_min = min(all_pts[:, 0].min(), x_target[0, 0]) - margin
    x1_max = max(all_pts[:, 0].max(), x_target[0, 0]) + margin
    x2_min = min(all_pts[:, 1].min(), x_target[0, 1]) - margin
    x2_max = max(all_pts[:, 1].max(), x_target[0, 1]) + margin


    # square the grid for better visualization
    r1, r2 = x1_max - x1_min, x2_max - x2_min
    if r1 > r2:
        pad = (r1 - r2) / 2
        x2_min -= pad; x2_max += pad
    else:
        pad = (r2 - r1) / 2
        x1_min -= pad; x1_max += pad

    # ------------------------------------------------------------------
    # Cost contour grid
    # ------------------------------------------------------------------
    # N_grid = 300
    # xx = np.linspace(x1_min, x1_max, N_grid)
    # yy = np.linspace(x2_min, x2_max, N_grid)
    # X, Y = np.meshgrid(xx, yy)
    # Z    = _cost_grid(X, Y, V)

    # # log levels give dark centre, light exterior
    # z_lo   = max(float(Z.min()), 1e-8)
    # z_hi   = float(Z.max())
    # levels = np.logspace(np.log10(z_lo), np.log10(z_hi), 30)

    # ------------------------------------------------------------------
    # Field contour 
    # ------------------------------------------------------------------

    field_contour   = None
    field_level     = None

    if field is not None:

        speed_max = 0.0

        sample_frames = np.linspace(0, len(states) - 1, min(20, len(states)), dtype=int)

        for frame in sample_frames:

            _, _, _, _, speed = field.grid_for_plots(t=times[frame], xlim=(x1_min, x1_max), ylim=(x2_min, x2_max))

            speed_max = max(speed_max, float(speed.max()))
            field_levels = np.linspace(0.0,  max(speed_max, 1e-8), field.config.pp.levels)


    # ------------------------------------------------------------------
    # Figure elements
    # ------------------------------------------------------------------
    
    fig, ax = plt.subplots(figsize=(7, 7))
    field_contour = [None]

    if field is not None:

        X, Y, U, W, speed = field.grid_for_plots( t=times[0], xlim=(x1_min, x1_max), ylim=(x2_min, x2_max))
        field_contour[0] = ax.contourf(X, Y, speed, levels=field_levels, alpha=field.config.pp.alpha, zorder=0)
        fig.colorbar(field_contour[0], ax=ax, label='Disturbance intensity')


    # contours
    #ax.contourf(X, Y, Z, levels=levels, cmap='gray_r')
    #ax.contour(X, Y, Z, levels=levels, colors='dimgray', linewidths=0.4, linestyles='--', alpha=0.5)

    # targets
    #ax.plot(x_target[0], x_target[1], 'g+', markersize=14, markeredgewidth=2, label='target', zorder=6)
    #ax.plot(x_target[0, 0], x_target[0, 1], 'g+', markersize=14, markeredgewidth=2, label='target', zorder=6)
    cur_target,    = ax.plot([], [], 'g+', markersize=14, markeredgewidth=2, label='target', zorder=6)
    trace_target,  = ax.plot([], [], color='green', lw=2.0, zorder=5)

    # states
    ax.plot(states[0, 0], states[0, 1], 'ro', markersize=8, label='start', zorder=6)

    # trajectory
    trace_line, = ax.plot([], [], color='royalblue', lw=2.0, zorder=5)
    cur_dot,    = ax.plot([], [], 'o', color='royalblue', markersize=8, zorder=7)
    
    # predictions 
    pred_line,  = ax.plot([], [], color='royalblue', lw=1.0, alpha=0.75, ls=':', zorder=4)
    pred_dots,  = ax.plot([], [], 'o', color='royalblue', markersize=4, alpha=0.75, label='predicted', zorder=5)

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xlabel('$x_1$  (position)', fontsize=12)
    ax.set_ylabel('$x_2$  (position)', fontsize=12)
    ax.set_title('Convex MPC Trajectory', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()

    # ------------------------------------------------------------------
    # animation callbacks
    # ------------------------------------------------------------------
    def init():
        trace_line.set_data([], [])
        cur_dot.set_data([], [])
        pred_line.set_data([], [])
        pred_dots.set_data([], [])
        cur_target.set_data([], [])
        return trace_line, cur_dot, pred_line, pred_dots, cur_target, trace_target

    def update(frame):

        
        #disturbance update
        if field is not None:

            field_contour[0].remove()

            X, Y, U, W, speed = field.grid_for_plots( t=times[frame], xlim=(x1_min, x1_max), ylim=(x2_min, x2_max))
            field_contour[0] = ax.contourf(X, Y, speed, levels=field_levels, alpha=field.config.pp.alpha, zorder=0)
            #fig.colorbar(field_contour, ax=ax, label='Disturbance intensity')


        # trajectory update 
        trace_line.set_data(states[:frame + 1, 0], states[:frame + 1, 1])
        cur_dot.set_data([states[frame, 0]], [states[frame, 1]])

        # target update
        cur_target.set_data([x_target[frame, 0]], [x_target[frame, 1]])
        trace_target.set_data(x_target[:frame + 1, 0], x_target[:frame + 1, 1])

        # prediction update
        if frame < T and predicted_sequences[frame] is not None:
            pred = predicted_sequences[frame]            
            px   = np.concatenate([[states[frame, 0]], pred[:, 0]])
            py   = np.concatenate([[states[frame, 1]], pred[:, 1]])
            pred_line.set_data(px, py)
            pred_dots.set_data(pred[:, 0], pred[:, 1])
        else:
            pred_line.set_data([], [])
            pred_dots.set_data([], [])

        return trace_line, cur_dot, pred_line, pred_dots, cur_target, trace_target

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(states),
        init_func=init,
        blit=False,
        interval=150,
    )

    ani.save(filename, writer=animation.PillowWriter(fps=8))
    print(f"Animation saved to '{filename}'")
    plt.close(fig)


def plot_inputs(time_history, input_history, constraints, filename='inputs.png'):

    inputs = np.array(input_history)   
    T, nu  = inputs.shape
    steps  = time_history #np.arange(T)

    # bounds
    u_min = u_max = None
    if constraints.get('type') == 'box':
        u_min = np.array(constraints['u_min'], dtype=float)
        u_max = np.array(constraints['u_max'], dtype=float)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(nu, 1, figsize=(9, 3 * nu), sharex=True)
    if nu == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        color = colors[i % len(colors)]
        ax.plot(steps, inputs[:, i], color=color, lw=1.8, label=f'$u_{i+1}$')

        if u_min is not None and u_max is not None:
            ax.axhline(u_min[i], color=color, lw=1.2, ls='--', alpha=0.7,
                       label=f'$u_{{{i+1},min}}$ = {u_min[i]:.2g}')
            ax.axhline(u_max[i], color=color, lw=1.2, ls=':',  alpha=0.7,
                       label=f'$u_{{{i+1},max}}$ = {u_max[i]:.2g}')
            ax.axhspan(u_min[i], u_max[i], color=color, alpha=0.06)

        ax.set_ylabel(f'$u_{i+1}$', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)

    axes[-1].set_xlabel('Time, $t$ [secs]', fontsize=12)
    fig.suptitle('Control Inputs vs. Constraints', fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Input plot saved to '{filename}'")
    plt.close(fig)


def plot_velocities(time_history, state_history, constraints, vel_indices=None, filename='velocities.png'):

    states = np.array(state_history)   # (T+1, nx)
    T, nx  = states.shape
    steps  = time_history[1:] #np.arange(T)

    if vel_indices is None:
        vel_indices = list(range(nx // 2, nx))

    # bounds
    x_min = x_max = None
    if constraints.get('type') == 'box':
        x_min = np.array(constraints['x_min'], dtype=float)
        x_max = np.array(constraints['x_max'], dtype=float)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    nv = len(vel_indices)

    fig, axes = plt.subplots(nv, 1, figsize=(9, 3 * nv), sharex=True)
    if nv == 1:
        axes = [axes]

    for plot_i, (ax, state_i) in enumerate(zip(axes, vel_indices)):
        color = colors[plot_i % len(colors)]
        label = f'$x_{{{state_i + 1}}}$'
        ax.plot(steps, states[:, state_i], color=color, lw=1.8, label=label)

        if x_min is not None and x_max is not None:
            ax.axhline(x_min[state_i], color=color, lw=1.2, ls='--', alpha=0.7,
                       label=f'$x_{{{state_i + 1},min}}$ = {x_min[state_i]:.2g}')
            ax.axhline(x_max[state_i], color=color, lw=1.2, ls=':',  alpha=0.7,
                       label=f'$x_{{{state_i + 1},max}}$ = {x_max[state_i]:.2g}')
            ax.axhspan(x_min[state_i], x_max[state_i], color=color, alpha=0.06)

        ax.set_ylabel(label, fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)

    axes[-1].set_xlabel('Time, $t$ [secs]', fontsize=12)
    fig.suptitle('Velocity States vs. Constraints', fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Velocity plot saved to '{filename}'")
    plt.close(fig)




def plot_trajectory_compare(
    time_history_list,
    state_history_list,
    x_target_list=None,
    filename='trajectory.png',
    target_tolerance=1e-3, 
    custom_label = ['target', 'with d-rejection', 'without d-rejection']
):


    traj_labels = custom_label

    n_trials = len(state_history_list)

    if len(time_history_list) != n_trials:
        raise ValueError(
            "time_history_list and state_history_list must have the same length"
        )

    if x_target_list is not None and len(x_target_list) != n_trials:
        raise ValueError(
            "x_target_list must have the same length as state_history_list"
        )

    # --------------------------------------------------------------
    # Convert / validate histories
    # --------------------------------------------------------------
    states_list = []

    for i, state_history in enumerate(state_history_list):

        states = np.asarray(state_history)

        if states.ndim != 2 or states.shape[1] < 2:
            raise ValueError(
                f"state_history_list[{i}] must have shape (T, nx) with nx >= 2"
            )

        states_list.append(states)

    # --------------------------------------------------------------
    # Target
    # --------------------------------------------------------------
    target = None

    if x_target_list is not None:

        targets = []

        for i, x_target in enumerate(x_target_list):

            target_i = np.asarray(x_target)

            if target_i.ndim == 1:
                target_i = target_i.reshape(1, -1)

            if target_i.ndim != 2 or target_i.shape[1] < 2:
                raise ValueError(
                    f"x_target_list[{i}] must have shape (T, nx) with nx >= 2"
                )

            targets.append(target_i)

        # First target is the reference target
        target = targets[0]

        # Verify all targets are sufficiently close to the first
        for i, target_i in enumerate(targets[1:], start=1):

            if target_i.shape != target.shape:
                raise ValueError(
                    f"Target {i} has shape {target_i.shape}, "
                    f"but target 0 has shape {target.shape}"
                )

            max_difference = np.max(
                np.abs(target_i[:, :2] - target[:, :2])
            )

            # targets should be the same (or close), else the comparison may not be fair
            if max_difference > target_tolerance:
                raise ValueError(
                    f"Target {i} differs from target 0 by as much as "
                    f"{max_difference:.6g}, exceeding target_tolerance="
                    f"{target_tolerance:.6g}"
                )

    # --------------------------------------------------------------
    # Plot bounds using ALL trajectories + single target
    # --------------------------------------------------------------
    all_pts = [states[:, :2] for states in states_list]

    if target is not None:
        all_pts.append(target[:, :2])

    all_pts = np.vstack(all_pts)

    margin = 0.5

    x1_min = all_pts[:, 0].min() - margin
    x1_max = all_pts[:, 0].max() + margin
    x2_min = all_pts[:, 1].min() - margin
    x2_max = all_pts[:, 1].max() + margin

    # make square
    r1 = x1_max - x1_min
    r2 = x2_max - x2_min

    if r1 > r2:
        pad = 0.5 * (r1 - r2)
        x2_min -= pad
        x2_max += pad

    else:
        pad = 0.5 * (r2 - r1)
        x1_min -= pad
        x1_max += pad

    # --------------------------------------------------------------
    # Labels
    # --------------------------------------------------------------
    #traj_labels = [
    #    'with d-rejection',
    #    'without d-rejection'
    #]

    # ==============================================================
    # PLOT 1: TRAJECTORIES
    # ==============================================================

    fig, ax = plt.subplots(figsize=(7, 7))

    # Target — plot once
    if target is not None:

        ax.plot(
            target[:, 0],
            target[:, 1],
            color='green',
            linestyle='--',
            lw=2.0,
            label=traj_labels[0],
            zorder=2
        )

        # final target
        ax.plot(
            target[-1, 0],
            target[-1, 1],
            '+',
            color='green',
            markersize=14,
            markeredgewidth=2,
            zorder=6
        )

    # --------------------------------------------------------------
    # Trajectories
    # --------------------------------------------------------------
    trajectory_colors = []

    for i, states in enumerate(states_list):

        traj_label = traj_labels[i+1]

        line, = ax.plot(
            states[:, 0],
            states[:, 1],
            linestyle='-',
            lw=2.0,
            label=traj_label,
            zorder=3
        )

        color = line.get_color()
        trajectory_colors.append(color)

        ax.plot(
            states[-1, 0],
            states[-1, 1],
            'o',
            color=color,
            markersize=6,
            zorder=5
        )

    # --------------------------------------------------------------
    # Formatting
    # --------------------------------------------------------------
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

    ax.set_xlabel('$x$-position', fontsize=10)
    ax.set_ylabel('$y$-position', fontsize=10)
    ax.set_title('Trajectory Comparison through Disturbance Field', fontsize=10)

    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_aspect('equal')

    ax.legend(
        loc='upper right',
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Trajectory plot saved to '{filename}'")
    plt.close(fig)

    # ==============================================================
    # PLOT 2: CUMULATIVE TRACKING ERROR
    # ==============================================================

    if target is not None:

        fig, ax = plt.subplots(figsize=(8, 5))

        for i, (time_history, states) in enumerate(
            zip(time_history_list, states_list)
        ):

            time = np.asarray(time_history)

            # Ensure state, target, and time histories line up
            n = min(
                len(time),
                len(states),
                len(target)
            )

            time_i = time[:n]
            states_i = states[:n]
            target_i = target[:n]

            # Instantaneous Euclidean position error
            tracking_error = np.linalg.norm(
                states_i[:, :2] - target_i[:, :2],
                axis=1
            )

            # Cumulative tracking error
            cumulative_error = np.cumsum(tracking_error)

            ax.plot(
                time_i,
                cumulative_error,
                color=trajectory_colors[i],
                lw=2.0,
                label=traj_labels[i+1]
            )

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Cumulative Tracking Error', fontsize=12)
        ax.set_title('Tracking Performance through Disturbance Field', fontsize=13)

        ax.grid(True, linestyle=':', alpha=0.5)

        ax.legend(
            loc='upper left',
            fontsize=10
        )

        plt.tight_layout()

        # Create second filename from first
        base, ext = os.path.splitext(filename)
        error_filename = f"{base}_tracking_error{ext}"

        plt.savefig(error_filename, dpi=150)
        print(
            f"Cumulative tracking error plot saved to "
            f"'{error_filename}'"
        )

        plt.close(fig)


def plot_trajectory(time_history, state_history, x_target=None, filename='trajectory.png'):

    states = np.array(state_history)   # shape: (T, nx)

    if states.ndim != 2 or states.shape[1] < 2:
        raise ValueError("state_history must have shape (T, nx) with nx >= 2")

    # optional target history
    target = None
    if x_target is not None:
        target = np.array(x_target)
        if target.ndim == 1:
            target = target.reshape(1, -1)

    # --------------------------------------------------------------
    # Plot bounds
    # --------------------------------------------------------------
    all_pts = [states[:, :2]]

    if target is not None and target.shape[1] >= 2:
        all_pts.append(target[:, :2])

    all_pts = np.vstack(all_pts)
    margin = 0.5

    x1_min = all_pts[:, 0].min() - margin
    x1_max = all_pts[:, 0].max() + margin
    x2_min = all_pts[:, 1].min() - margin
    x2_max = all_pts[:, 1].max() + margin

    # make square
    r1 = x1_max - x1_min
    r2 = x2_max - x2_min
    if r1 > r2:
        pad = 0.5 * (r1 - r2)
        x2_min -= pad
        x2_max += pad
    else:
        pad = 0.5 * (r2 - r1)
        x1_min -= pad
        x1_max += pad

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    # actual trajectory
    ax.plot(states[:, 0], states[:, 1], color='royalblue', lw=2.0, label='trajectory', zorder=3)

    # start and end
    ax.plot(states[0, 0], states[0, 1], 'ro', markersize=8, label='start', zorder=5)
    ax.plot(states[-1, 0], states[-1, 1], 'o', color='royalblue', markersize=8, label='end', zorder=5)

    # target history or final target
    if target is not None and target.shape[1] >= 2:
        if len(target) == len(states):
            ax.plot(target[:, 0], target[:, 1], color='green', lw=2.0, alpha=0.9, label='target path', zorder=2)
            ax.plot(target[-1, 0], target[-1, 1], 'g+', markersize=14, markeredgewidth=2, label='final target', zorder=6)
        else:
            ax.plot(target[-1, 0], target[-1, 1], 'g+', markersize=14, markeredgewidth=2, label='target', zorder=6)

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xlabel('$x_1$  (position)', fontsize=12)
    ax.set_ylabel('$x_2$  (position)', fontsize=12)
    ax.set_title('Trajectory', fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Trajectory plot saved to '{filename}'")
    plt.close(fig)


def plot_R_evaluation(learned_data, benchmark_data, filename="visualization/cala/R_evaluation.png"):

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # --------------------------------------------------
    # unpack
    # --------------------------------------------------

    t_learned = np.asarray(learned_data["step"]).reshape(-1)
    t_benchmark = np.asarray(benchmark_data["step"]).reshape(-1)

    x_learned = np.asarray(learned_data["state"])
    x_benchmark = np.asarray(benchmark_data["state"])

    target_learned = np.asarray(learned_data["target"])
    target_benchmark = np.asarray(benchmark_data["target"])

    # --------------------------------------------------
    # tracking error
    # --------------------------------------------------

    error_learned = np.linalg.norm(
        x_learned[:, :2] - target_learned[:, :2],
        axis=1
    )

    error_benchmark = np.linalg.norm(
        x_benchmark[:, :2] - target_benchmark[:, :2],
        axis=1
    )

    rmse_learned = np.sqrt(
        np.mean(error_learned**2)
    )

    rmse_benchmark = np.sqrt(
        np.mean(error_benchmark**2)
    )

    improvement = (
        100.0
        * (rmse_benchmark - rmse_learned)
        / (rmse_benchmark + 1e-12)
    )

    # --------------------------------------------------
    # plot
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    # trajectory
    ax = axes[0]

    ax.plot(
        target_learned[:, 0],
        target_learned[:, 1],
        linestyle="--",
        linewidth=2,
        label="Target"
    )

    ax.plot(
        x_learned[:, 0],
        x_learned[:, 1],
        linewidth=2,
        label="Learned R"
        #label=f"Learned R (RMSE={rmse_learned:.3f})"
    )

    ax.plot(
        x_benchmark[:, 0],
        x_benchmark[:, 1],
        linewidth=2,
        label=f"Benchmark R"
        #label=f"Benchmark R (RMSE={rmse_benchmark:.3f})"
    )

    ax.set_title("Tracking trajectory")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    # tracking error over time
    ax = axes[1]

    ax.plot(
        t_learned,
        error_learned,
        linewidth=2,
        label="Learned R"
    )

    ax.plot(
        t_benchmark,
        error_benchmark,
        linewidth=2,
        label="Benchmark R"
    )

    ax.set_title(
        f"Tracking Error — "
        f"RL improvement = {improvement:.1f}%"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position error")
    ax.grid(True)
    ax.legend()

    fig.suptitle(
        "RL-Learned MPC R vs Static Benchmark R"
    )

    fig.tight_layout()

    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # useful console result
    print(f"Learned-R position RMSE:   {rmse_learned:.4f}")
    print(f"Benchmark-R position RMSE: {rmse_benchmark:.4f}")
    print(f"RMSE improvement:          {improvement:.2f}%")

def plot_R_evaluation_cum(
    learned_data,
    benchmark_data,
    R0=None,
    filename="visualization/cala/R_evaluation_cum.png"
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # --------------------------------------------------
    # unpack
    # --------------------------------------------------
    t_learned = np.asarray(learned_data["step"]).reshape(-1)
    t_benchmark = np.asarray(benchmark_data["step"]).reshape(-1)

    x_learned = np.asarray(learned_data["state"])
    x_benchmark = np.asarray(benchmark_data["state"])

    target_learned = np.asarray(learned_data["target"])
    target_benchmark = np.asarray(benchmark_data["target"])

    u_learned = np.asarray(learned_data["input"])
    u_benchmark = np.asarray(benchmark_data["input"])

    # use common length just in case
    n = min(
        len(t_learned),
        len(t_benchmark),
        len(x_learned),
        len(x_benchmark),
        len(target_learned),
        len(target_benchmark),
        len(u_learned),
        len(u_benchmark),
    )

    t_learned = t_learned[:n]
    t_benchmark = t_benchmark[:n]
    x_learned = x_learned[:n]
    x_benchmark = x_benchmark[:n]
    target_learned = target_learned[:n]
    target_benchmark = target_benchmark[:n]
    u_learned = u_learned[:n]
    u_benchmark = u_benchmark[:n]

    # sample time
    if n >= 2:
        Ts = float(np.mean(np.diff(t_learned)))
    else:
        Ts = 1.0

    # --------------------------------------------------
    # tracking error
    # --------------------------------------------------
    error_learned = np.linalg.norm(
        x_learned[:, :2] - target_learned[:, :2],
        axis=1
    )

    error_benchmark = np.linalg.norm(
        x_benchmark[:, :2] - target_benchmark[:, :2],
        axis=1
    )

    rmse_learned = float(np.sqrt(np.mean(error_learned**2)))
    rmse_benchmark = float(np.sqrt(np.mean(error_benchmark**2)))

    rmse_improvement = (
        100.0
        * (rmse_benchmark - rmse_learned)
        / (rmse_benchmark + 1e-12)
    )

    # running / cumulative RMSE
    idx = np.arange(1, n + 1)
    #running_rmse_learned = np.sqrt(np.cumsum(error_learned**2) / idx)
    #running_rmse_benchmark = np.sqrt(np.cumsum(error_benchmark**2) / idx)
    running_rmse_learned = np.cumsum(error_learned**2) * Ts
    running_rmse_benchmark = np.cumsum(error_benchmark**2) * Ts

    # --------------------------------------------------
    # control effort
    # --------------------------------------------------
    if R0 is None:
        effort_learned = np.sum(u_learned**2, axis=1)
        effort_benchmark = np.sum(u_benchmark**2, axis=1)
        effort_ylabel = r"$\sum ||u||^2 \Delta t$"
    else:
        R0 = np.asarray(R0)
        effort_learned = np.einsum("bi,ij,bj->b", u_learned, R0, u_learned)
        effort_benchmark = np.einsum("bi,ij,bj->b", u_benchmark, R0, u_benchmark)
        effort_ylabel = r"$\sum u^\top R_0 u \, \Delta t$"

    cumulative_effort_learned = np.cumsum(effort_learned) * Ts
    cumulative_effort_benchmark = np.cumsum(effort_benchmark) * Ts

    total_effort_learned = float(cumulative_effort_learned[-1])
    total_effort_benchmark = float(cumulative_effort_benchmark[-1])

    effort_saving = (
        100.0
        * (total_effort_benchmark - total_effort_learned)
        / (total_effort_benchmark + 1e-12)
    )

    # --------------------------------------------------
    # plot layout
    # --------------------------------------------------
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1, 1])

    ax_traj = fig.add_subplot(gs[:, 0])   # full left column
    ax_rmse = fig.add_subplot(gs[0, 1])   # top-right
    ax_eff  = fig.add_subplot(gs[1, 1])   # bottom-right

    # --------------------------------------------------
    # left: trajectory
    # --------------------------------------------------
    ax_traj.plot(
        target_learned[:, 0],
        target_learned[:, 1],
        linestyle="--",
        color = 'green',
        linewidth=2,
        label="Target"
    )

    ax_traj.plot(
        x_learned[:, 0],
        x_learned[:, 1],
        linewidth=2,
        label="Learned R"
        #label=f"Learned R (RMSE={rmse_learned:.3f})"
    )

    ax_traj.plot(
        x_benchmark[:, 0],
        x_benchmark[:, 1],
        linewidth=2,
        label="Benchmark R"
        #label=f"Benchmark R (RMSE={rmse_benchmark:.3f})"
    )

    ax_traj.set_title("Tracking trajectory")
    ax_traj.set_xlabel("$x_0$")
    ax_traj.set_ylabel("$x_1$")
    ax_traj.set_aspect("equal")
    ax_traj.grid(True)
    ax_traj.legend()

    # --------------------------------------------------
    # top-right: cumulative / running RMSE
    # --------------------------------------------------
    ax_rmse.plot(
        t_learned,
        running_rmse_learned,
        linewidth=2,
        label="Learned R"
    )

    ax_rmse.plot(
        t_benchmark,
        running_rmse_benchmark,
        linewidth=2,
        label="Benchmark R"
    )

    ax_rmse.set_title(
        f"Cumulative RMSE — RL improvement: {rmse_improvement:.1f}%"
    )
    ax_rmse.set_xlabel("Time [s]")
    ax_rmse.set_ylabel("Running RMSE")
    ax_rmse.grid(True)
    ax_rmse.legend()

    # --------------------------------------------------
    # bottom-right: cumulative effort
    # --------------------------------------------------
    ax_eff.plot(
        t_learned,
        cumulative_effort_learned,
        linewidth=2,
        label="Learned R"
        #label=f"Learned R (total={total_effort_learned:.3f})"
    )

    ax_eff.plot(
        t_benchmark,
        cumulative_effort_benchmark,
        linewidth=2,
        label="Benchmark R"
        #label=f"Benchmark R (total={total_effort_benchmark:.3f})"
    )

    ax_eff.set_title(
        f"Cumulative control effort - Change: {-effort_saving:.1f}%"
    )
    ax_eff.set_xlabel("Time [s]")
    ax_eff.set_ylabel(effort_ylabel)
    ax_eff.grid(True)
    ax_eff.legend()

    fig.suptitle("RL-Learned MPC R vs Static Benchmark R")
    fig.tight_layout()

    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------
    # console summary
    # --------------------------------------------------
    print(f"Learned-R final RMSE:        {rmse_learned:.4f}")
    print(f"Benchmark-R final RMSE:      {rmse_benchmark:.4f}")
    print(f"RMSE improvement:            {rmse_improvement:.2f}%")
    print(f"Learned-R total effort:      {total_effort_learned:.4f}")
    print(f"Benchmark-R total effort:    {total_effort_benchmark:.4f}")
    print(f"Effort saving:               {effort_saving:.2f}%")

def plot_R_evaluation_pareto(
    learned_data,
    benchmark_data,
    R0=None,
    filename="visualization/cala/R_evaluation_pareto.png"
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # --------------------------------------------------
    # unpack
    # --------------------------------------------------
    t_learned = np.asarray(learned_data["step"]).reshape(-1)
    t_benchmark = np.asarray(benchmark_data["step"]).reshape(-1)

    x_learned = np.asarray(learned_data["state"])
    x_benchmark = np.asarray(benchmark_data["state"])

    target_learned = np.asarray(learned_data["target"])
    target_benchmark = np.asarray(benchmark_data["target"])

    u_learned = np.asarray(learned_data["input"])
    u_benchmark = np.asarray(benchmark_data["input"])

    # guard against slight length mismatch
    n = min(
        len(t_learned),
        len(t_benchmark),
        len(x_learned),
        len(x_benchmark),
        len(target_learned),
        len(target_benchmark),
        len(u_learned),
        len(u_benchmark),
    )

    t_learned = t_learned[:n]
    t_benchmark = t_benchmark[:n]
    x_learned = x_learned[:n]
    x_benchmark = x_benchmark[:n]
    target_learned = target_learned[:n]
    target_benchmark = target_benchmark[:n]
    u_learned = u_learned[:n]
    u_benchmark = u_benchmark[:n]

    # sample time
    if n >= 2:
        Ts = float(np.mean(np.diff(t_learned)))
    else:
        Ts = 1.0

    # --------------------------------------------------
    # tracking RMSE
    # --------------------------------------------------
    error_learned = np.linalg.norm(
        x_learned[:, :2] - target_learned[:, :2],
        axis=1
    )

    error_benchmark = np.linalg.norm(
        x_benchmark[:, :2] - target_benchmark[:, :2],
        axis=1
    )

    rmse_learned = float(np.sqrt(np.mean(error_learned**2)))
    rmse_benchmark = float(np.sqrt(np.mean(error_benchmark**2)))

    rmse_improvement = (
        100.0
        * (rmse_benchmark - rmse_learned)
        / (rmse_benchmark + 1e-12)
    )

    # --------------------------------------------------
    # control effort
    # --------------------------------------------------
    if R0 is None:
        effort_learned = np.sum(u_learned**2, axis=1)
        effort_benchmark = np.sum(u_benchmark**2, axis=1)
        xlabel = r"Total control effort $\sum ||u||^2 \Delta t$"
    else:
        R0 = np.asarray(R0)
        effort_learned = np.einsum("bi,ij,bj->b", u_learned, R0, u_learned)
        effort_benchmark = np.einsum("bi,ij,bj->b", u_benchmark, R0, u_benchmark)
        xlabel = r"Total control effort $\sum u^\top R_0 u \, \Delta t$"

    total_effort_learned = float(np.sum(effort_learned) * Ts)
    total_effort_benchmark = float(np.sum(effort_benchmark) * Ts)

    effort_saving = (
        100.0
        * (total_effort_benchmark - total_effort_learned)
        / (total_effort_benchmark + 1e-12)
    )

    # --------------------------------------------------
    # determine qualitative result
    # --------------------------------------------------
    if (rmse_learned <= rmse_benchmark) and (total_effort_learned <= total_effort_benchmark):
        outcome = "Learned R dominates benchmark"
    elif (rmse_learned >= rmse_benchmark) and (total_effort_learned >= total_effort_benchmark):
        outcome = "Learned R is dominated by benchmark"
    else:
        outcome = "Tracking / effort tradeoff"

    # --------------------------------------------------
    # plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))

    # points
    ax.scatter(
        total_effort_benchmark,
        rmse_benchmark,
        s=140,
        label="Benchmark R",
        zorder=3
    )

    ax.scatter(
        total_effort_learned,
        rmse_learned,
        s=140,
        label="Learned R",
        zorder=3
    )

    # line connecting them
    ax.plot(
        [total_effort_benchmark, total_effort_learned],
        [rmse_benchmark, rmse_learned],
        linestyle="--",
        linewidth=1.5,
        zorder=2
    )

    # annotations
    ax.annotate(
        f"Benchmark R\nRMSE={rmse_benchmark:.3f}\nEffort={total_effort_benchmark:.3f}",
        (total_effort_benchmark, rmse_benchmark),
        xytext=(10, 10),
        textcoords="offset points"
    )

    ax.annotate(
        f"Learned R\nRMSE={rmse_learned:.3f}\nEffort={total_effort_learned:.3f}",
        (total_effort_learned, rmse_learned),
        xytext=(10, -35),
        textcoords="offset points"
    )

    # axis labels / title
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Position RMSE")

    ax.set_title(
        "RL-Learned MPC R vs Static Benchmark R\n"
        f"RMSE improvement = {rmse_improvement:.1f}%   |   "
        f"Effort saving = {effort_saving:.1f}%"
    )

    ax.grid(True)
    ax.legend()

    # pad limits slightly
    x_vals = np.array([total_effort_benchmark, total_effort_learned], dtype=float)
    y_vals = np.array([rmse_benchmark, rmse_learned], dtype=float)

    x_pad = max(0.05 * (x_vals.max() - x_vals.min() + 1e-12), 1e-3)
    y_pad = max(0.05 * (y_vals.max() - y_vals.min() + 1e-12), 1e-3)

    ax.set_xlim(x_vals.min() - x_pad, x_vals.max() + x_pad)
    ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)

    # small textbox
    ax.text(
        0.02,
        0.98,
        outcome,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    fig.tight_layout()

    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    # --------------------------------------------------
    # console summary
    # --------------------------------------------------
    print(f"Learned-R RMSE:              {rmse_learned:.4f}")
    print(f"Benchmark-R RMSE:            {rmse_benchmark:.4f}")
    print(f"RMSE improvement:            {rmse_improvement:.2f}%")
    print(f"Learned-R total effort:      {total_effort_learned:.4f}")
    print(f"Benchmark-R total effort:    {total_effort_benchmark:.4f}")
    print(f"Effort saving:               {effort_saving:.2f}%")
    print(f"Pareto assessment:           {outcome}")

