
'''
Generate a nonlinear input-channel disturbance field

        x[k+1] = A x[k] + B (u[k] + d(p[k], t))

        where A,B,x,u are the plant and
        d(p,t) is a nonlinear 2D field composed of:
        
        1. slowly rotating background bias (wind), and
        2. several vortex-like spatial disturbances (defined by centers)


        derivation in /docs/disturbance/nonlinear.md

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from types import SimpleNamespace
import os
import json


class VortexConfig():

    def __init__(self):

        # these are the centers of the vortices (hard code now, make random later)
        self.vortex_centers = np.array([
            [-2.5, 2.0],
            [2.0, 2.0],
            [2.5, -2.0],
            [-2.0, -1.5]
            
        ]) 
        
        # corresponding size(s), amplitudes(s), and rate(s)
        self.vortex_sigmas = np.array([1.5, 
                                       1.8, 
                                       1.4,
                                       2.0])
        self.vortex_amps = 5*np.array([0.65, 
                                     1.00,
                                     -0.75,
                                     0.90])
        self.background_amp = 5*0.35
        self.omega = 0.2 #0.35
        self.moving_centers = True
        self.center_motion_amp = 0.35
        self.center_motion_rate = 0.2

        # centralized plotting parameter
        self.pp = SimpleNamespace(
            levels=30,
            alpha=0.75,
            density=1.2, 
            linewidth=1.0, 
            arrowsize=1.1, 
            xlim=(-5.0, 5.0), 
            ylim=(-5.0, 5.0), 
            n=35,
            dpi=180,
            framealpha=0.9,
            interval=50,
            rate=0.15,
            frames=90
        )

class VortexField:

    def __init__(self, configs_base):

        # load parameters from config
        with open(f'{configs_base}/config_visualization.json') as f:
            cfg_viz = json.load(f)
        field_folder = cfg_viz['field_folder']

        self.config = VortexConfig()
        print('vortex initialized')
        os.makedirs(f'{field_folder}', exist_ok=True)
        self.plot_field_at_t(f'{field_folder}/field.png', 0)
        #self.animate_field(animate_field_path=f"{field_folder}/field_animation.gif")


    def evolve_centers(self, t):

        # pull out initial vortex centers
        centers = np.asarray(self.config.vortex_centers).copy()

        if not self.config.moving_centers:
            return centers
        
        # else, iterate through each center
        for i in range(len(centers)):

            phase = i * np.pi / 2.0
            centers[i, 0] += self.config.center_motion_amp * np.sin(self.config.center_motion_rate * t + phase)
            centers[i, 1] += self.config.center_motion_amp * np.cos(self.config.center_motion_rate * t + phase)

        return centers 
    
    def evolve_background(self, t):

        b = self.config.background_amp * np.array([
            np.cos(self.config.omega * t),
            np.sin(self.config.omega * t)
            ])
        
        return b

    def compute_gain(self, c, t):

        return 1.0 + 0.30 * np.sin(0.5 * self.config.omega * t + np.linalg.norm(c))

    
    def compute_disturbance(self, x, t):

        # strip out x/y position
        p = np.asarray(x)[:2]

        # background wind
        b = self.evolve_background(t)
        
        # get the updated centers
        centers = self.evolve_centers(t)

        # initialize total disturbance
        d_actual = b.copy()

        for c, sigma, amp in zip(centers, self.config.vortex_sigmas, self.config.vortex_amps):

            diff = p - c
            spacial_influence = np.exp(-np.dot(diff, diff) / (sigma**2))
            vortex_direction = np.array([-diff[1], diff[0]])
            gain = self.compute_gain(c, t)

            d_actual += amp * gain * vortex_direction * spacial_influence

        return d_actual

    #def grid_for_plots(self, t):
    def grid_for_plots(self, t, xlim = None, ylim = None):

        if xlim is None:
            xlim = self.config.pp.xlim

        if ylim is None:
            ylim = self.config.pp.ylim

        #xs = np.linspace(self.config.pp.xlim[0], self.config.pp.xlim[1], self.config.pp.n)
        #ys = np.linspace(self.config.pp.ylim[0], self.config.pp.ylim[1], self.config.pp.n)
        xs = np.linspace(xlim[0], xlim[1], self.config.pp.n)
        ys = np.linspace(ylim[0], ylim[1], self.config.pp.n)

        X, Y = np.meshgrid(xs, ys)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for row in range(self.config.pp.n):
            for col in range(self.config.pp.n):
                d = self.compute_disturbance(np.array([X[row, col], Y[row, col]]), t)
                U[row, col] = d[0]
                V[row, col] = d[1]

        speed = np.sqrt(U**2 + V**2)
        return X, Y, U, V, speed        

    def _draw_field(self, ax, t):

        X, Y, U, V, speed = self.grid_for_plots(t=t)
        contour = ax.contourf(X, Y, speed, levels=self.config.pp.levels, alpha=self.config.pp.alpha)
        stream = ax.streamplot(X, Y, U, V, density=self.config.pp.density, linewidth=self.config.pp.linewidth, arrowsize=self.config.pp.arrowsize)

        ax.set_title(f"Disturbance field at t = {t:.2f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(self.config.pp.xlim)
        ax.set_ylim(self.config.pp.ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        #ax.legend(loc="upper right", framealpha=0.9)
        plt.tight_layout()

        return contour, stream


    def plot_field_at_t(self, plot_field_path, t):

        fig, ax = plt.subplots(figsize=(8,7))

        contour, stream = self._draw_field(ax = ax, t = t)
        
        fig.colorbar(contour, ax=ax, label = 'Disturbance Intensity')
        fig.savefig(plot_field_path, dpi=self.config.pp.dpi)
        plt.close(fig)

    def animate_field(self, animate_field_path):

        fig, ax = plt.subplots(figsize = (8, 7))
        
        # initial
        initial_contour, stream = self._draw_field(ax = ax, t = 0.0)

        fig.colorbar(initial_contour, ax=ax, label = 'Disturbance Intensity')

        # legends 
        legend_index= [
            Patch(alpha=self.config.pp.alpha, label="Disturbance intensity"),
            Line2D([0],[0], linewidth=self.config.pp.linewidth,label="Streamlines",),
        ]

        #ax.legend( handles=legend_index, loc="upper right", framealpha=self.config.pp.framealpha,)
    
        def update(frame):

            t = frame*self.config.pp.rate
            ax.clear()

            contour, stream = self._draw_field(ax = ax, t = t)
            #ax.legend(handles=legend_index, loc="upper right", framealpha=self.config.pp.framealpha,)

        ani = FuncAnimation(fig, update, frames = self.config.pp.frames, interval = self.config.pp.interval, blit = False)
        fig.tight_layout()
        ani.save(animate_field_path, writer="pillow", fps=max(1, round(1000 / self.config.pp.interval)),)

        plt.close(fig)



# testing 
#field = VortexField()
#field.plot_field_at_t(plot_field_path = 'visualization/plots/field.png', t =0.0)
#field.animate_field(animate_field_path="visualization/animations/field.gif")
















