import json
import numpy as np

# online modeller
class FeatureMap():
    
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
            # normalize within each feature set
            for name, index in self.names_index.items():
                phi[index] = (phi[index]/ (np.sum(phi[index]) + 1e-12))

        self.phi = phi.copy()

        return phi

    # evaluate one feature over the workspace for plotting
    def activation_grid(self, feature_index, t, resolution = 70):

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
        ax.scatter(test.centers[:, 0], test.centers[:, 1], marker="x",label="Feature centres")
        ax.set_title(test.names[feature_index])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(test.x_lims)
        ax.set_ylim(test.y_lims)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.legend()

        plt.show()


    # Plot activation across x,y,t (fixing one)
    def plot_fixed_axis(self, feature_index = 0, fixed_axis = 1, fixed_at = 0.0):

        resolution = 200
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




#-----------
# testing     
# ---------

import matplotlib.pyplot as plt

test = FeatureMap()
test.plot_feature(feature_index = 0, t = 0.0)
test.plot_fixed_axis(feature_index = -1, fixed_axis = 0, fixed_at = 0.0)

