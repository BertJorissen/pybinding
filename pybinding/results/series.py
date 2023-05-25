"""Processing and presentation of the results with x-y variables"""
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Literal, Optional, List

from ..utils import with_defaults, pltutils
from ..support.pickle import pickleable

__all__ = ['Series']


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
