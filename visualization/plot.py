'''

This module mostly vibe-coded using Claude Sonnet 4.6

Minor refinements made by author: tjards

'''

# standard imports 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json


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