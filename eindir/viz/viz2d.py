import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator

from eindir.core.components import FPair, ObjectiveFunction


class Plot2dObj:
    """
    Class for plotting 2D objective functions.

    #### Description
    This class is used to create 2D and 3D plots of an objective function. It
    provides methods to prepare the values for the plot, and to create a 3D
    surface plot or a 2D contour plot of the function.

    #### Parameters
    **obj** (`ObjectiveFunction`)
    : An instance of the `ObjectiveFunction` class representing the function to
    plot.

    **nelem** (`int`)
    : The number of elements in the plot.

    #### Attributes
    **func** (`ObjectiveFunction`)
    : The function to plot.

    **nelem** (`int`)
    : The number of elements in the plot.

    **X**, **Y**, **Z** (`npt.NDArray`)
    : Arrays representing the X, Y, and Z coordinates for the plot.

    **contourExtent** (`list` of `float`)
    : The extent of the contour plot.

    **X_glob_min**, **Y_glob_min**, **Z_glob_min** (`float`)
    : The X, Y, and Z coordinates of the global minimum of the function.

    **pdat** (`NoneType`)
    : Placeholder for future data.

    #### Notes
    The `prepVals` method prepares the Z values for the plot.

    The `create3d` method creates a 3D surface plot of the function.

    The `createContour` method creates a 2D contour plot of the function.
    """

    def __init__(self, obj: ObjectiveFunction, nelem: int):
        """
        Initializes an instance of the Plot2dObj class.

        #### Parameters
        **obj** (`ObjectiveFunction`)
        : An instance of the `ObjectiveFunction` class representing the function
        to plot.

        **nelem** (`int`)
        : The number of elements in the plot.
        """
        self.func = obj
        self.nelem = nelem
        fll = self.func.limits.low
        fhh = self.func.limits.high
        step_size_x = abs(fll[0] - fhh[0]) / self.nelem
        step_size_y = abs(fll[1] - fhh[1]) / self.nelem
        plX = np.arange(fll[0], fhh[0], step_size_x)
        plY = np.arange(fll[1], fhh[1], step_size_y)
        self.X, self.Y = np.meshgrid(plX, plY, indexing="xy")
        self.Z = self.prepVals()
        self.contourExtent = [
            np.min(self.X.ravel()),
            np.max(self.X.ravel()),
            np.min(self.Y.ravel()),
            np.max(self.Y.ravel()),
        ]
        self.X_glob_min = self.X.ravel()[self.Z.argmin()]
        self.Y_glob_min = self.Y.ravel()[self.Z.argmin()]
        self.Z_glob_min = np.min(self.Z.ravel())
        # Set to the current grid min
        # TODO: Handle degenerate minima
        if isinstance(obj.globmin, type(None)):
            obj.globmin = FPair(
                val=self.Z_glob_min,
                pos=np.array([self.X_glob_min, self.Y_glob_min]),
            )
        self.pdat = None

    def prepVals(self):
        """
        Prepares the Z values for the plot.

        #### Notes
        This method evaluates the function at a grid of points and returns an
        array of the results.
        """
        grid_vals = [
            self.func(np.column_stack([self.X[itera], self.Y[itera]]))
            for itera in range(self.nelem)
        ]
        return np.array(grid_vals)

    def create3d(self, showGlob=True, savePath=None):
        """
        Creates a 3D surface plot of the function.

        #### Parameters
        **showGlob** (`bool`, optional)
        : Whether to show the global minimum on the plot. Default is `True`.

        **savePath** (`str`, optional)
        : The path to save the plot. If `None`, the plot is shown instead.
        Default is `None`.

        #### Notes
        This method creates a 3D surface plot of the function using matplotlib.
        The plot includes a colorbar and optional markers and labels for the
        global minimum.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = plt.subplot(projection="3d")
        surf = ax.plot_surface(self.X, self.Y, self.Z, cmap=cm.batlow, alpha=0.7)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        [t.set_va("center") for t in ax.get_yticklabels()]
        [t.set_ha("left") for t in ax.get_yticklabels()]
        [t.set_va("center") for t in ax.get_xticklabels()]
        [t.set_ha("right") for t in ax.get_xticklabels()]
        [t.set_va("center") for t in ax.get_zticklabels()]
        [t.set_ha("left") for t in ax.get_zticklabels()]
        fig.colorbar(surf, shrink=0.35, aspect=8)
        ax.view_init(elev=15, azim=220)
        if showGlob:
            ax.scatter(
                self.X_glob_min,
                self.Y_glob_min,
                self.Z_glob_min,
                color="black",
                alpha=1,
            )
            ax.text(
                self.X_glob_min,
                self.Y_glob_min,
                self.Z_glob_min,
                "Global Minima",
                color="black",
                alpha=1,
            )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(str(self.func))
        if savePath is not None:
            plt.savefig(savePath, dpi=300)
        else:
            plt.show()

    def createContour(self, showGlob=True, savePath=None):
        """
        Creates a 2D contour plot of the function.

        #### Parameters
        **showGlob** (`bool`, optional)
        : Whether to show the global minimum on the plot. Default is `True`.

        **savePath** (`str`, optional)
        : The path to save the plot. If `None`, the plot is shown instead.
        Default is `None`.

        #### Notes
        This method creates a 2D contour plot of the function using matplotlib.
        The plot includes a colorbar, contour lines with labels, and optional
        markersand labels for the global minimum.
        """
        plt.figure(figsize=(12, 10))
        ax = plt.subplot()
        [t.set_va("center") for t in ax.get_yticklabels()]
        [t.set_ha("left") for t in ax.get_yticklabels()]
        [t.set_va("center") for t in ax.get_xticklabels()]
        [t.set_ha("right") for t in ax.get_xticklabels()]
        plt.imshow(
            self.Z,
            extent=self.contourExtent,
            origin="lower",
            cmap=cm.batlow,
            alpha=0.8,
        )
        plt.colorbar()
        contours = ax.contour(self.X, self.Y, self.Z, 10, colors="black", alpha=0.9)
        plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
        if showGlob:
            plt.plot(
                self.X_glob_min,
                self.Y_glob_min,
                color="white",
                marker="x",
                markersize=5,
            )
            ax.text(
                self.X_glob_min + 0.1,
                self.Y_glob_min,
                "Global Minima",
                color="white",
            )
        if savePath is not None:
            plt.savefig(savePath, dpi=300)
        else:
            plt.show()
