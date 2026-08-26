# note: developed with the assistance of ChatGPT 5.6

"""
UMAP visualization of the CALA feature space and nonlinear disturbance field.

The objective is to visualize how the nonlinear disturbance field is
structured in the high-dimensional contextual feature space used by CALA.

For every sampled physical/contextual point:

    (x, y, t)

we compute:

    1. the CALA feature vector

           phi_raw(x, y, t)

    2. one GLOBAL normalization over the complete feature vector

           phi = phi_raw / sum(phi_raw)

    3. the true nonlinear disturbance

           d(x, y, t) = [d_x, d_y]

The normalized feature vectors are reduced to two dimensions using UMAP:

           phi --> [UMAP-1, UMAP-2]

The actual disturbance is NOT included in the UMAP embedding.

Instead, after UMAP:

    d_x and d_y are interpolated over a regular UMAP grid

and used to construct:

    magnitude = sqrt(d_x^2 + d_y^2)
    direction = atan2(d_y, d_x)

Three separate plots are produced:

    1. Disturbance direction contour
    2. Disturbance magnitude contour
    3. 3-D magnitude surface, colored by disturbance direction
"""

# ------------------------------------------------------
# standard imports
# ------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm, colormaps
from matplotlib.colors import Normalize

from sklearn.neighbors import KNeighborsRegressor


# ------------------------------------------------------
# custom imports
# ------------------------------------------------------

import cala
import nonlinear_field


# ======================================================
# FEATURE / FIELD UMAP
# ======================================================

class FeatureFieldUMAP:

    def __init__(
        self,
        configs_base="configs/default"
    ):

        # --------------------------------------------------
        # configuration location
        # --------------------------------------------------

        self.configs_base = str(
            configs_base
        ).rstrip("/")

        # --------------------------------------------------
        # CALA feature map
        # --------------------------------------------------

        self.feature_map = cala.RTFeatureMap(
            configs_base=self.configs_base
        )

        self.feature_map.normalize = False

        # --------------------------------------------------
        # actual nonlinear disturbance field
        # --------------------------------------------------

        self.field = nonlinear_field.VortexField(
            configs_base=self.configs_base
        )

        # --------------------------------------------------
        # storage
        # --------------------------------------------------

        self.samples = None

        self.reducer = None
        self.embedding = None

        # UMAP-grid representation
        self.grid_X = None
        self.grid_Y = None

        self.grid_dx = None
        self.grid_dy = None

        self.grid_magnitude = None
        self.grid_angle = None


    # ==================================================
    # GLOBAL FEATURE NORMALIZATION
    # ==================================================

    def _normalize_phi(
        self,
        phi
    ):
        """
        Apply one global L1 normalization over the entire
        feature vector.

            phi_normalized = phi / sum(phi)

        This includes:

            bias
            every radial basis feature
            sin(t)
            cos(t)

        No feature groups are separately weighted here.
        """

        phi = np.asarray(
            phi,
            dtype=float
        ).reshape(-1)

        total = float(
            np.sum(phi)
        )

        if abs(total) < 1e-12:

            return phi.copy()

        return (
            phi / total
        )


    # ==================================================
    # SAMPLE PHYSICAL / CONTEXTUAL SPACE
    # ==================================================

    def sample(
        self,

        x_lims=None,
        y_lims=None,
        t_lims=(0.0, 30.0),

        x_n=30,
        y_n=30,
        t_n=12,
    ):
        """
        Sample the complete x-y-t domain.

        For each sampled point:

            phi(x,y,t)
            d(x,y,t)

        are computed.

        Parameters
        ----------
        x_lims : tuple
            (x_min, x_max)

        y_lims : tuple
            (y_min, y_max)

        t_lims : tuple
            (t_min, t_max)

        x_n, y_n, t_n : int
            Number of samples on each axis.
        """

        # --------------------------------------------------
        # use CALA workspace if unspecified
        # --------------------------------------------------

        if x_lims is None:

            x_lims = (
                self.feature_map.x_lims
            )

        if y_lims is None:

            y_lims = (
                self.feature_map.y_lims
            )

        # --------------------------------------------------
        # build axes
        # --------------------------------------------------

        xs = np.linspace(
            x_lims[0],
            x_lims[1],
            int(x_n)
        )

        ys = np.linspace(
            y_lims[0],
            y_lims[1],
            int(y_n)
        )

        # endpoint=False avoids duplicating the end of a
        # periodic temporal interval when appropriate
        ts = np.linspace(
            t_lims[0],
            t_lims[1],
            int(t_n),
            endpoint=False
        )

        # --------------------------------------------------
        # staging
        # --------------------------------------------------

        phi_list = []

        xyz_list = []

        d_list = []

        # --------------------------------------------------
        # sample
        # --------------------------------------------------

        for t in ts:

            for y in ys:

                for x in xs:

                    # --------------------------------------
                    # 1. raw CALA features
                    # --------------------------------------

                    phi_raw = (
                        self.feature_map
                        .build_features(
                            [x, y],
                            t
                        )
                    )

                    # --------------------------------------
                    # 2. global normalization
                    # --------------------------------------

                    phi = self._normalize_phi(
                        phi_raw
                    )

                    # --------------------------------------
                    # 3. true nonlinear disturbance
                    # --------------------------------------

                    d = (
                        self.field
                        .compute_disturbance(
                            np.array(
                                [x, y]
                            ),
                            t
                        )
                    )

                    # --------------------------------------
                    # store
                    # --------------------------------------

                    phi_list.append(
                        phi.copy()
                    )

                    xyz_list.append(
                        [
                            x,
                            y,
                            t
                        ]
                    )

                    d_list.append(
                        np.asarray(
                            d,
                            dtype=float
                        ).copy()
                    )

        # --------------------------------------------------
        # convert to arrays
        # --------------------------------------------------

        phi = np.asarray(
            phi_list,
            dtype=float
        )

        xyz = np.asarray(
            xyz_list,
            dtype=float
        )

        d = np.asarray(
            d_list,
            dtype=float
        )

        dx = d[:, 0]
        dy = d[:, 1]

        magnitude = np.sqrt(
            dx**2 + dy**2
        )

        angle = np.arctan2(
            dy,
            dx
        )

        # --------------------------------------------------
        # store
        # --------------------------------------------------

        self.samples = {

            "phi": phi,

            "xyz": xyz,

            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "t": xyz[:, 2],

            "d": d,

            "dx": dx,
            "dy": dy,

            "magnitude": magnitude,
            "angle": angle,

            "x_lims": x_lims,
            "y_lims": y_lims,
            "t_lims": t_lims,

            "x_n": int(x_n),
            "y_n": int(y_n),
            "t_n": int(t_n),
        }

        # --------------------------------------------------
        # summary
        # --------------------------------------------------

        print(
            f"UMAP sampling complete: "
            f"{phi.shape[0]} samples"
        )

        print(
            f"Feature dimension: "
            f"{phi.shape[1]}"
        )

        print(
            "Feature normalization: "
            "global L1 over complete feature vector"
        )

        print(
            "Feature-vector sum range: "
            f"{np.min(np.sum(phi, axis=1)):.6f} to "
            f"{np.max(np.sum(phi, axis=1)):.6f}"
        )

        return self.samples


    # ==================================================
    # FIT UMAP
    # ==================================================

    def fit_umap(
        self,

        n_neighbors=200,

        min_dist=0.50,

        spread=1.0,

        random_state=42,

        metric="euclidean",
    ):
        """
        Reduce globally normalized feature vectors to 2-D.

        Larger n_neighbors and min_dist are intentionally
        used to encourage a richer, more connected embedding
        rather than many tiny isolated islands.
        """

        if self.samples is None:

            raise RuntimeError(
                "Call sample(...) before fit_umap(...)."
            )

        try:

            import umap

        except ImportError:

            raise ImportError(
                "UMAP is not installed.\n"
                "Install using:\n\n"
                "pip install umap-learn"
            )

        n_samples = len(
            self.samples["phi"]
        )

        # cannot exceed sample count
        n_neighbors = min(
            int(n_neighbors),
            n_samples - 1
        )

        # --------------------------------------------------
        # construct reducer
        # --------------------------------------------------

        self.reducer = umap.UMAP(

            n_components=2,

            n_neighbors=n_neighbors,

            min_dist=float(
                min_dist
            ),

            spread=float(
                spread
            ),

            metric=metric,

            random_state=random_state,

            init="spectral",

            low_memory=True,
        )

        # --------------------------------------------------
        # fit
        # --------------------------------------------------

        self.embedding = (
            self.reducer
            .fit_transform(
                self.samples["phi"]
            )
        )

        print(
            "UMAP embedding complete: "
            f"{self.embedding.shape}"
        )

        print(
            f"UMAP n_neighbors: "
            f"{n_neighbors}"
        )

        print(
            f"UMAP min_dist: "
            f"{min_dist}"
        )

        return self.embedding


    # ==================================================
    # BUILD SMOOTH FIELD IN UMAP SPACE
    # ==================================================

    def build_umap_field(
        self,

        grid_n=250,

        interpolation_neighbors=75,

        padding=0.02,
    ):
        """
        Construct a smooth regular field over UMAP coordinates.

        d_x and d_y are interpolated independently.

        Don't interpolate direction directly, because angles
        wrap from +pi to -pi.

        Instead:

            dx(z1,z2)
            dy(z1,z2)

        are interpolated first.

        Then:

            magnitude = sqrt(dx^2 + dy^2)

            angle = atan2(dy, dx)
        """

        if self.embedding is None:

            raise RuntimeError(
                "Call fit_umap(...) before "
                "build_umap_field(...)."
            )

        # --------------------------------------------------
        # embedded points
        # --------------------------------------------------

        z = np.asarray(
            self.embedding,
            dtype=float
        )

        dx = np.asarray(
            self.samples["dx"],
            dtype=float
        )

        dy = np.asarray(
            self.samples["dy"],
            dtype=float
        )

        # --------------------------------------------------
        # determine plotting bounds
        # --------------------------------------------------

        z1_min = float(
            np.min(z[:, 0])
        )

        z1_max = float(
            np.max(z[:, 0])
        )

        z2_min = float(
            np.min(z[:, 1])
        )

        z2_max = float(
            np.max(z[:, 1])
        )

        z1_range = (
            z1_max - z1_min
        )

        z2_range = (
            z2_max - z2_min
        )

        z1_min -= (
            padding * z1_range
        )

        z1_max += (
            padding * z1_range
        )

        z2_min -= (
            padding * z2_range
        )

        z2_max += (
            padding * z2_range
        )

        # --------------------------------------------------
        # regular UMAP grid
        # --------------------------------------------------

        z1_grid = np.linspace(
            z1_min,
            z1_max,
            int(grid_n)
        )

        z2_grid = np.linspace(
            z2_min,
            z2_max,
            int(grid_n)
        )

        X, Y = np.meshgrid(
            z1_grid,
            z2_grid
        )

        query = np.column_stack(
            [
                X.reshape(-1),
                Y.reshape(-1)
            ]
        )

        # --------------------------------------------------
        # interpolation
        # --------------------------------------------------

        k = min(
            int(interpolation_neighbors),
            len(z)
        )

        reg_dx = KNeighborsRegressor(

            n_neighbors=k,

            weights="distance"
        )

        reg_dy = KNeighborsRegressor(

            n_neighbors=k,

            weights="distance"
        )

        reg_dx.fit(
            z,
            dx
        )

        reg_dy.fit(
            z,
            dy
        )

        # --------------------------------------------------
        # predict smooth vector field
        # --------------------------------------------------

        DX = (
            reg_dx
            .predict(query)
            .reshape(X.shape)
        )

        DY = (
            reg_dy
            .predict(query)
            .reshape(Y.shape)
        )

        from scipy.ndimage import gaussian_filter

        DX = gaussian_filter(
            DX,
            sigma=1.5
        )

        DY = gaussian_filter(
            DY,
            sigma=1.5
        )


        # --------------------------------------------------
        # derived field properties
        # --------------------------------------------------

        magnitude = np.sqrt(
            DX**2 + DY**2
        )

        angle = np.arctan2(
            DY,
            DX
        )

        # --------------------------------------------------
        # save
        # --------------------------------------------------

        self.grid_X = X
        self.grid_Y = Y

        self.grid_dx = DX
        self.grid_dy = DY

        self.grid_magnitude = (
            magnitude
        )

        self.grid_angle = (
            angle
        )

        print(
            "Smooth disturbance field "
            "constructed in UMAP space."
        )

        return (
            X,
            Y,
            DX,
            DY,
            magnitude,
            angle
        )


    # ==================================================
    # PLOT 1:
    # DISTURBANCE DIRECTION
    # ==================================================

    def plot_direction(
        self,
        filename="visualization/features/umap_disturbance_direction.png",
        show_samples=False,
    ):

        self._check_grid()

        fig, ax = plt.subplots(figsize=(10, 8))

        # --------------------------------------------------
        # cyclic direction field
        # --------------------------------------------------

        direction = ax.pcolormesh(
            self.grid_X,
            self.grid_Y,
            self.grid_angle,
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
            shading="gouraud"
        )

        # optional actual UMAP samples
        if show_samples:
            ax.scatter(
                self.embedding[:, 0],
                self.embedding[:, 1],
                s=1.5,
                alpha=0.08,
                color="black"
            )

        # --------------------------------------------------
        # colorbar
        # --------------------------------------------------

        cbar = fig.colorbar(
            direction,
            ax=ax
        )

        cbar.set_label(
            "Disturbance direction"
        )

        cbar.set_ticks([
            -np.pi,
            -np.pi / 2,
            0,
            np.pi / 2,
            np.pi
        ])

        cbar.set_ticklabels([
            r"$-\pi$",
            r"$-\pi/2$",
            "0",
            r"$\pi/2$",
            r"$\pi$"
        ])

        ax.set_title(
            "Disturbance Direction in CALA Feature Space"
        )

        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

        ax.grid(
            True,
            alpha=0.15
        )

        fig.tight_layout()

        self._save(
            fig,
            filename
        )

    # ==================================================
    # PLOT 2:
    # DISTURBANCE MAGNITUDE
    # ==================================================

    def plot_magnitude(
        self,

        filename=
        "visualization/features/"
        "umap_disturbance_magnitude.png",

        levels=60,

        contour_lines=True,

        show_samples=False,
    ):

        self._check_grid()

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        # --------------------------------------------------
        # smooth magnitude contour
        # --------------------------------------------------

        contour = ax.contourf(

            self.grid_X,
            self.grid_Y,
            self.grid_magnitude,

            levels=int(levels),

            cmap="viridis",

            extend="max"
        )

        # --------------------------------------------------
        # contour lines
        # --------------------------------------------------

        # if contour_lines:

        #     lines = ax.contour(

        #         self.grid_X,
        #         self.grid_Y,
        #         self.grid_magnitude,

        #         levels=14,

        #         linewidths=0.45,

        #         alpha=0.40,

        #         colors="black"
        #     )

        #     ax.clabel(

        #         lines,

        #         inline=True,

        #         fontsize=7,

        #         fmt="%.1f"
        #     )

        # --------------------------------------------------
        # optional samples
        # --------------------------------------------------

        if show_samples:

            ax.scatter(

                self.embedding[:, 0],
                self.embedding[:, 1],

                s=1.5,

                alpha=0.08,

                color="black"
            )

        # --------------------------------------------------
        # colorbar
        # --------------------------------------------------

        cbar = fig.colorbar(
            contour,
            ax=ax
        )

        cbar.set_label(
            r"Disturbance magnitude $||d||$"
        )

        # --------------------------------------------------
        # labels
        # --------------------------------------------------

        ax.set_title(
            "Disturbance Magnitude in "
            "CALA Feature Space"
        )

        ax.set_xlabel(
            "UMAP-1"
        )

        ax.set_ylabel(
            "UMAP-2"
        )

        ax.grid(
            True,
            alpha=0.15
        )

        fig.tight_layout()

        self._save(
            fig,
            filename
        )


    # ==================================================
    # PLOT 3:
    # 3-D DISTURBANCE SURFACE
    # ==================================================

    def plot_surface(
        self,

        filename=
        "visualization/features/"
        "umap_disturbance_surface.png",
    ):

        self._check_grid()

        # --------------------------------------------------
        # figure
        # --------------------------------------------------

        fig = plt.figure(
            figsize=(11, 9)
        )

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        # --------------------------------------------------
        # cyclic direction normalization
        # --------------------------------------------------

        norm = Normalize(

            vmin=-np.pi,

            vmax=np.pi
        )

        # FIX:
        # modern Matplotlib API
        cmap = colormaps[
            "twilight_shifted"
        ]

        # --------------------------------------------------
        # map direction -> face color
        # --------------------------------------------------

        facecolors = cmap(
            norm(
                self.grid_angle
            )
        )

        # --------------------------------------------------
        # magnitude surface
        # --------------------------------------------------

        ax.plot_surface(

            self.grid_X,
            self.grid_Y,

            self.grid_magnitude,

            facecolors=facecolors,

            linewidth=0,

            antialiased=True,

            rcount=180,

            ccount=180,

            shade=True
        )

        # --------------------------------------------------
        # direction colorbar
        # --------------------------------------------------

        scalar_map = cm.ScalarMappable(

            norm=norm,

            cmap=cmap
        )

        scalar_map.set_array([])

        cbar = fig.colorbar(

            scalar_map,

            ax=ax,

            shrink=0.68,

            pad=0.10
        )

        cbar.set_label(
            "Disturbance direction"
        )

        cbar.set_ticks(
            [
                -np.pi,
                -np.pi / 2,
                0,
                np.pi / 2,
                np.pi
            ]
        )

        cbar.set_ticklabels(
            [
                r"$-\pi$",
                r"$-\pi/2$",
                "0",
                r"$\pi/2$",
                r"$\pi$"
            ]
        )

        # --------------------------------------------------
        # labels
        # --------------------------------------------------

        ax.set_title(
            "Nonlinear Disturbance Surface "
            "in CALA Feature Space"
        )

        ax.set_xlabel(
            "UMAP-1"
        )

        ax.set_ylabel(
            "UMAP-2"
        )

        ax.set_zlabel(
            r"Disturbance magnitude $||d||$"
        )

        # useful default camera angle
        ax.view_init(

            elev=32,

            azim=-55
        )

        fig.tight_layout()

        self._save(
            fig,
            filename
        )


    # ==================================================
    # SAVE HELPER
    # ==================================================

    def _save(
        self,
        fig,
        filename
    ):

        folder = os.path.dirname(
            filename
        )

        if folder:

            os.makedirs(
                folder,
                exist_ok=True
            )

        fig.savefig(

            filename,

            dpi=220,

            bbox_inches="tight"
        )

        plt.close(
            fig
        )

        print(
            f"saved: {filename}"
        )


    # ==================================================
    # VALIDATION HELPER
    # ==================================================

    def _check_grid(
        self
    ):

        if self.grid_X is None:

            raise RuntimeError(

                "No UMAP field found. "
                "Call build_umap_field(...) "
                "before plotting."
            )


    # ==================================================
    # MASTER PIPELINE
    # ==================================================

    def run(
        self,

        # --------------------------------------------------
        # physical/context sampling
        # --------------------------------------------------

        x_lims=None,

        y_lims=None,

        t_lims=(0.0, 30.0),

        x_n=30,

        y_n=30,

        t_n=12,

        # --------------------------------------------------
        # UMAP
        # --------------------------------------------------

        n_neighbors=200,

        min_dist=0.50,

        spread=1.0,

        metric="euclidean",

        random_state=42,

        # --------------------------------------------------
        # field interpolation
        # --------------------------------------------------

        grid_n=250,

        interpolation_neighbors=75,

        # --------------------------------------------------
        # output
        # --------------------------------------------------

        folder="visualization/features",
    ):

        os.makedirs(
            folder,
            exist_ok=True
        )

        # ==================================================
        # 1. SAMPLE RAW FEATURE SPACE
        # ==================================================

        self.sample(

            x_lims=x_lims,

            y_lims=y_lims,

            t_lims=t_lims,

            x_n=x_n,

            y_n=y_n,

            t_n=t_n,
        )

        # ==================================================
        # 2. UMAP
        # ==================================================

        self.fit_umap(

            n_neighbors=n_neighbors,

            min_dist=min_dist,

            spread=spread,

            metric=metric,

            random_state=random_state,
        )

        # ==================================================
        # 3. SMOOTH VECTOR FIELD IN UMAP SPACE
        # ==================================================

        self.build_umap_field(

            grid_n=grid_n,

            interpolation_neighbors=
            interpolation_neighbors,
        )

        # ==================================================
        # 4. THREE SEPARATE PLOTS
        # ==================================================





        self.plot_direction(

            filename=os.path.join(

                folder,

                "umap_disturbance_direction.png"
            )
        )

        self.plot_magnitude(

            filename=os.path.join(

                folder,

                "umap_disturbance_magnitude.png"
            )
        )

        self.plot_surface(

            filename=os.path.join(

                folder,

                "umap_disturbance_surface.png"
            )
        )

        print(
            "\nUMAP feature-field visualization complete."
        )