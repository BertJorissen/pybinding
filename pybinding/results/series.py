"""Processing and presentation of the results with x-y variables"""
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Literal, Optional, List

from ..utils import with_defaults, pltutils
from ..support.pickle import pickleable
from matplotlib.collections import LineCollection, QuadMesh

from .path import Path, Area, AbstractArea
__all__ = ['Series', 'SeriesPath', 'SeriesArea']


@pickleable
class Series:
    """A series of data points determined by a common relation, i.e. :math:`y = f(x)`

    Attributes
    ----------
    variable : array_like
        Independent variable for which the data was computed.
    data : array_like
        An array of values which were computed as a function of `variable`.
        It can be 1D or 2D. In the latter case each column represents the result
        of a different function applied to the same `variable` input.
    labels : dict
        Plot labels: 'variable', 'data', 'orbitals', 'title' and 'columns'.
    """
    def __init__(self, variable: ArrayLike, data: ArrayLike, labels: Optional[dict] = None):
        self.variable = np.atleast_1d(variable)
        self.data = np.atleast_1d(data)
        self.labels = with_defaults(
            labels, variable="x", data="y", columns="", title="",
            orbitals=[str(i) for i in range(self.data.shape[1])] if self.data.ndim == 2 else [])

    def with_data(self, data: np.ndarray) -> 'Series':
        """Return a copy of this result object with different data"""
        result = copy(self)
        result.data = data
        return result

    def __add__(self, other: 'Series') -> 'Series':
        """Add together the data of two Series object in a new object."""
        if self.data.ndim < other.data.ndim:
            # keep information about the orbitals, so take the other series as a reference
            return other.with_data(self.data[:, np.newaxis] + other.data)
        elif self.data.ndim > other.data.ndim:
            return self.with_data(self.data + other.data[:, np.newaxis])
        else:
            return self.with_data(self.data + other.data)

    def __sub__(self, other: 'Series') -> 'Series':
        """Subtract the data of two Series object in a new object."""
        if self.data.ndim < other.data.ndim:
            # keep information about the orbitals, so take the other series as a reference
            return other.with_data(self.data[:, np.newaxis] - other.data)
        elif self.data.ndim > other.data.ndim:
            return self.with_data(self.data - other.data[:, np.newaxis])
        else:
            return self.with_data(self.data - other.data)

    def reduced(self, columns: Optional[List[int]] = None, orbitals: Optional[List[str]] = None,
                fill_other: float = 0.) -> 'Series':
        """Return a copy where the data is summed over the columns

        Only applies to results which may have multiple columns of data, e.g.
        results for multiple orbitals for LDOS calculation.

        Parameters
        ----------
        columns : Optional[List[int]]
            The colummns to contract to the new array.
            The length of `columns` agrees with the dimensions of data.shape[1].
            The value at each position corresponds to the new column of the new Series object
        orbitals: Optional[List[str]]
            Optional new list of entries for the `orbitals` label in `labels`
        fill_other : float
            In case an array is made with a new column, fill it with this value. Default: 0.
        """
        if columns is None:
            columns = np.zeros(self.data.shape[1])
        col_idx = np.array(columns, dtype=int)
        if np.all(col_idx == 0):
            # case where all the axis are summed over, no 'orbital' label is needed
            return self.with_data(self.data.sum(axis=1))
        col_max = np.max(col_idx) + 1
        if orbitals is None:
            orb_list = [str(i) for i in range(col_max)]
            for c_i in np.unique(col_idx):
                orb_list[c_i] = self.labels["orbitals"][np.argmax(col_idx == c_i)]
        else:
            orb_list = orbitals
        data = np.full((self.data.shape[0], col_max), fill_other)
        for c_i in np.unique(col_idx):
            data[:, c_i] = np.sum(self.data[:, col_idx == c_i], axis=1)
        series_out = self.with_data(data)
        series_out.labels["orbitals"] = orb_list
        return series_out

    def plot(self, ax: Optional[plt.Axes] = None, axes: Literal['xy', 'yx'] = 'xy', legend: bool = True,
             **kwargs) -> Optional[List[plt.Line2D]]:
        """Labeled line plot

        Parameters
        ----------
        ax : Optional[plt.Axes]
            The Axis to plot the results on.
        axes : Literal['xy', 'yx']
            The order of the axes, default: 'xy'.
        legend : bool
            Plot the legend of the bands on the axes.
        **kwargs
            Forwarded to `plt.plot()`.
        """
        if ax is None:
            ax = plt.gca()
        lines = []
        if axes == "xy":
            lines.append(ax.plot(self.variable, self.data, **kwargs))
            ax.set_xlim(self.variable.min(), self.variable.max())
            ax.set_xlabel(self.labels["variable"])
            ax.set_ylabel(self.labels["data"])
        elif axes == "yx":
            lines.append(ax.plot(self.data, self.variable, **kwargs))
            ax.set_ylim(self.variable.min(), self.variable.max())
            ax.set_xlabel(self.labels["data"])
            ax.set_ylabel(self.labels["variable"])

        if "title" in self.labels:
            ax.set_title(self.labels["title"])
        pltutils.despine(ax=ax)

        if self.data.ndim > 1 and legend:
            labels = [str(i) for i in range(self.data.shape[-1])]
            if "orbitals" in self.labels:
                labels = self.labels["orbitals"]
            pltutils.legend(labels=labels, title=self.labels["columns"], ax=ax)
        return lines


@pickleable
class SeriesPath(Series):
    """A series of data points determined by a common relation, i.e. :math:`y = f(x)`, with x in the reciprocal space

    Attributes
    ----------
    k_path : Path
        Independent variable for which the data was computed.
    data : array_like
        An array of values which were computed as a function of `variable`.
        It can be 1D or 2D. In the latter case each column represents the result
        of a different function applied to the same `variable` input.
    labels : dict
        Plot labels: 'variable', 'data', 'orbitals', 'title' and 'columns'.
    """

    def __init__(self, k_path: Path, data: ArrayLike, labels: Optional[dict] = None):
        self.k_path: Path = k_path
        super().__init__(self.k_path.as_1d(), data, labels)

    def line_plot(self, point_labels: Optional[List[str]] = None, ax: Optional[plt.Axes] = None, idx: int = 0,
                  plot_colorbar: bool = True, **kwargs) -> Optional[LineCollection]:
        """Line plot of the band structure with the color of the lines the data of the FatBands.

        Parameters
        ----------
        point_labels : Optional[List[str]]
            Labels for the `k_points`.
        ax : Optional[plt.Axes]
            The Axis to plot the bands on.
        idx : int
            The i-th column to plot. Default: 0.
        plot_colorbar : bool
            Show also the colorbar.
        **kwargs
            Forwarded to `matplotlib.collection.LineCollection()`.
        """
        if ax is None:
            ax = plt.gca()
        ax.set_aspect('equal')

        ax.set_title(self.labels["title"])
        line = pltutils.plot_color(self.k_path[:, 0], self.k_path[:, 1], self.data[:-1, idx], ax,
                                   **with_defaults(kwargs, cmap='RdYlBu_r'))
        if plot_colorbar:
            self.colorbar(ax=ax, label=self.labels["orbitals"][idx])
        self.k_path.decorate_plot(point_labels, ax)
        return line

    def colorbar(self, **kwargs):
        """Draw a colorbar with the label of :attr:`Sweep.data`"""
        return pltutils.colorbar(**with_defaults(kwargs, label=self.labels["data"]))


class SeriesArea(AbstractArea, SeriesPath):
    """A series of data points determined by a common relation, i.e. :math:`y = f(x)`, with x in the reciprocal area

    Attributes
    ----------
    k_area : Area
        Independent variable for which the data was computed.
    data : array_like
        An array of values which were computed as a function of `variable`.
        It can be 1D or 2D. In the latter case each column represents the result
        of a different function applied to the same `variable` input.
    labels : dict
        Plot labels: 'variable', 'data', 'orbitals', 'title' and 'columns'.
    """

    def __init__(self, k_area: Area, data: ArrayLike, labels: Optional[dict] = None):
        super().__init__(k_area)
        super(AbstractArea, self).__init__(self.karea_to_kpath(k_area), self.area_to_list(np.atleast_3d(data)), labels)

    def area_plot(self, point_labels: Optional[List[str]] = None, ax: Optional[plt.Axes] = None, idx: int = 0,
                  colorbar: bool = True, **kwargs) -> QuadMesh:
        """Line plot of the band structure with the color of the lines the data of the FatBands.

        Parameters
        ----------
        point_labels : Optional[List[str]]
            Labels for the `k_points`.
        ax : Optional[plt.Axes]
            The Axis to plot the bands on.
        idx : int
            The i-th column to plot. Default: 0.
        plot_colorbar : bool
            Show also the colorbar.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.pcolormesh`.
        """
        if ax is None:
            ax = plt.gca()
        ax.set_aspect('equal')

        ax.set_title(self.labels["title"])
        mesh = ax.pcolormesh(self.list_to_area(self.k_path[:, 0]), self.list_to_area(self.k_path[:, 1]),
                             self.data_area[:-1, :-1, idx],
                             **with_defaults(kwargs, cmap='RdYlBu_r', rasterized=True))
        ax._sci(mesh)

        if colorbar:
            self.colorbar(ax=ax, label=self.labels["orbitals"][idx])
        self.k_path.decorate_plot(point_labels, ax)
        return mesh

    @property
    def data_area(self) -> np.ndarray:
        return self.list_to_area(self.data)

    @data_area.setter
    def data_area(self, data: np.ndarray):
        self.data = self.area_to_list(data)
