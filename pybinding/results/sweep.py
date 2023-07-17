"""Processing and presentation of computed data in a sweep of parameters"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Literal, Optional, Union, Tuple
import matplotlib

from ..utils.misc import with_defaults
from ..utils import pltutils
from ..support.pickle import pickleable
from matplotlib.collections import LineCollection

__all__ = ['Sweep', 'NDSweep']


@pickleable
class Sweep:
    """2D parameter sweep with `x` and `y` 1D array parameters and `data` 2D array result

    Attributes
    ----------
    x : array_like
        1D array with x-axis values -- usually the primary parameter being swept.
    y : array_like
        1D array with y-axis values -- usually the secondary parameter.
    data : array_like
        2D array with `shape == (x.size, y.size)` containing the main result data.
    labels : dict
        Plot labels: 'title', 'x', 'y' and 'data'.
    tags : dict
        Any additional user defined variables.
    """
    def __init__(self, x: ArrayLike, y: ArrayLike, data: ArrayLike, labels: Optional[dict] = None,
                 tags: Optional[dict] = None):
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.data = np.atleast_2d(data)

        self.labels = with_defaults(labels, title="", x="x", y="y", data="data")
        self.tags = tags

    def __getitem__(self, item: Union[Tuple[int, int], int]) -> 'Sweep':
        """Same rules as numpy indexing"""
        if isinstance(item, tuple):
            idx_x, idx_y = item
        else:
            idx_x = item
            idx_y = slice(None)
        return self._with_data(self.x[idx_x], self.y[idx_y], self.data[idx_x, idx_y])

    def _with_data(self, x: ArrayLike, y: ArrayLike, data: ArrayLike) -> 'Sweep':
        return self.__class__(x, y, data, self.labels, self.tags)

    @property
    def _plain_labels(self) -> dict:
        """Labels with latex symbols stripped out"""
        trans = str.maketrans('', '', '$\\')
        return {k: v.translate(trans) for k, v in self.labels.items()}

    def _xy_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        """Expand x and y into 2D arrays matching data."""
        xgrid = np.column_stack([self.x] * self.y.size)
        ygrid = np.row_stack([self.y] * self.x.size)
        return xgrid, ygrid

    def save_txt(self, filename: str) -> None:
        """Save text file with 3 columns: x, y, data.

        Parameters
        ----------
        filename : str
        """
        with open(filename, 'w') as file:
            file.write("#{x:>11} {y:>12} {data:>12}\n".format(**self._plain_labels))

            xgrid, ygrid = self._xy_grids()
            for row in zip(xgrid.flat, ygrid.flat, self.data.flat):
                values = ("{:12.5e}".format(v) for v in row)
                file.write(" ".join(values) + "\n")

    def cropped(self, x: Optional[Tuple[float, float]] = None, y: Optional[Tuple[float, float]] = None) -> 'Sweep':
        """Return a copy with data cropped to the limits in the x and/or y axes

        A call with x=[-1, 2] will leave data only where -1 <= x <= 2.

        Parameters
        ----------
        x, y : Tuple[float, float]
            Min and max data limit.

        Returns
        -------
        :class:`~pybinding.Sweep`
        """
        idx_x = np.logical_and(x[0] <= self.x, self.x <= x[1]) if x else np.arange(self.x.size)
        idx_y = np.logical_and(y[0] <= self.y, self.y <= y[1]) if y else np.arange(self.y.size)
        return self._with_data(self.x[idx_x], self.y[idx_y], self.data[np.ix_(idx_x, idx_y)])

    def mirrored(self, axis: Literal['x', 'y'] = 'x') -> 'Sweep':
        """Return a copy with data mirrored in around specified axis

         Only makes sense if the axis starts at 0.

        Parameters
        ----------
        axis : 'x' or 'y'

        Returns
        -------
        :class:`~pybinding.Sweep`
        """
        if axis == 'x':
            x = np.concatenate((-self.x[::-1], self.x[1:]))
            data = np.vstack((self.data[::-1], self.data[1:]))
            return self._with_data(x, self.y, data)
        elif axis == 'y':
            y = np.concatenate((-self.y[::-1], self.y[1:]))
            data = np.hstack((self.data[:, ::-1], self.data[:, 1:]))
            return self._with_data(self.x, y, data)
        else:
            RuntimeError("Invalid axis")

    def interpolated(self, mul: Optional[Union[int, Tuple[int, int]]] = None,
                     size: Optional[Union[int, Tuple[int, int]]] = None,
                     kind: Literal['linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
                     'previous', 'next', 'zero', 'slinear', 'quadratic', 'cubic'] = 'linear') -> 'Sweep':
        """Return a copy with interpolate data using :class:`scipy.interpolate.interp1d`

        Call with `mul=2` to double the size of the x-axis and interpolate data to match.
        To interpolate in both axes pass a tuple, e.g. `mul=(4, 2)`.

        Parameters
        ----------
        mul : Union[int, Tuple[int, int]]
            Number of times the size of the axes should be multiplied.
        size : Union[int, Tuple[int, int]]
            New size of the axes. Zero will leave size unchanged.
        kind
            Forwarded to :class:`scipy.interpolate.interp1d`.

        Returns
        -------
        :class:`~pybinding.Sweep`
        """
        if not mul and not size:
            return self

        from scipy.interpolate import interp1d
        x, y, data = self.x, self.y, self.data

        if mul:
            try:
                mul_x, mul_y = mul
            except TypeError:
                mul_x, mul_y = mul, 1
            size_x = x.size * mul_x
            size_y = y.size * mul_y
        else:
            try:
                size_x, size_y = size
            except TypeError:
                size_x, size_y = size, 0

        if size_x > 0 and size_x != x.size:
            interpolate = interp1d(x, data, axis=0, kind=kind)
            x = np.linspace(x.min(), x.max(), size_x, dtype=x.dtype)
            data = interpolate(x)

        if size_y > 0 and size_y != y.size:
            interpolate = interp1d(y, data, kind=kind)
            y = np.linspace(y.min(), y.max(), size_y, dtype=y.dtype)
            data = interpolate(y)

        return self._with_data(x, y, data)

    def _convolved(self, sigma: float, axis: Literal['x', 'y'] = 'x') -> 'Sweep':
        """Return a copy where the data is convolved with a Gaussian function

        Parameters
        ----------
        sigma : float
            Gaussian broadening.
        axis : 'x' or 'y'

        Returns
        -------
        :class:`~pybinding.Sweep`
        """
        def convolve(v, data0):
            v0 = v[v.size // 2]
            gaussian = np.exp(-0.5 * ((v - v0) / sigma)**2)
            gaussian /= gaussian.sum()

            extend = 10  # TODO: rethink this
            data1 = np.concatenate((data0[extend::-1], data0, data0[:-extend:-1]))
            data1 = np.convolve(data1, gaussian, 'same')
            return data1[extend:-extend]

        x, y, data = self.x, self.y, self.data.copy()

        if 'x' in axis:
            for i in range(y.size):
                data[:, i] = convolve(x, data[:, i])
        if 'y' in axis:
            for i in range(x.size):
                data[i, :] = convolve(y, data[i, :])

        return self._with_data(x, y, data)

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs) -> matplotlib.collections.QuadMesh:
        """Plot a 2D colormap of :attr:`Sweep.data`

        Parameters
        ----------
        ax : Optional[plt.Axes]
            The axis to plot on.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.pcolormesh`.
        """
        if ax is None:
            ax = plt.gca()
        mesh = ax.pcolormesh(self.x, self.y, self.data.T,
                             **with_defaults(kwargs, cmap='RdYlBu_r', rasterized=True))
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_ylim(self.y.min(), self.y.max())

        ax.set_title(self.labels['title'])
        ax.set_xlabel(self.labels['x'])
        ax.set_ylabel(self.labels['y'])

        return mesh

    def colorbar(self, **kwargs):
        """Draw a colorbar with the label of :attr:`Sweep.data`"""
        return pltutils.colorbar(**with_defaults(kwargs, label=self.labels['data']))

    def _plot_slice(self, axis: Literal['x', 'y'], x: np.ndarray, y: ArrayLike, value: float,
                    ax: Optional[plt.Axes] = None, **kwargs) -> None:
        if ax is None:
            ax = plt.gca()
        ax.plot(x, y, **kwargs)

        split = self.labels[axis].split(' ', 1)
        label = split[0]
        unit = '' if len(split) == 1 else split[1].strip('()')
        ax.set_title('{}, {} = {:.2g} {}'.format(self.labels['title'], label, value, unit))

        ax.set_xlim(x.min(), x.max())
        ax.set_xlabel(self.labels['x' if axis == 'y' else 'y'])
        ax.set_ylabel(self.labels['data'])
        pltutils.despine(ax=ax)

    def _slice_x(self, x: float) -> np.ndarray:
        """Return a slice of data nearest to x and the found values of x.

        Parameters
        ----------
        x : float
        """
        idx = np.abs(self.x - x).argmin()
        return self.data[idx, :], self.x[idx]

    def _slice_y(self, y: float) -> np.ndarray:
        """Return a slice of data nearest to y and the found values of y.

        Parameters
        ----------
        y : float
        """
        idx = np.abs(self.y - y).argmin()
        return self.data[:, idx], self.y[idx]

    def plot_slice_x(self, x: ArrayLike, **kwargs) -> None:
        z, value = self._slice_x(x)
        self._plot_slice('x', self.y, z, value, **kwargs)

    def plot_slice_y(self, y: ArrayLike, **kwargs) -> None:
        z, value = self._slice_y(y)
        self._plot_slice('y', self.x, z, value, **kwargs)


@pickleable
class NDSweep:
    """ND parameter sweep

    Attributes
    ----------
    variables : tuple of array_like
        The parameters being swept.
    data : np.ndarray
        Main result array with `shape == [len(v) for v in variables]`.
    labels : dict
        Plot labels: 'title', 'x', 'y' and 'data'.
    tags : dict
        Any additional user defined variables.
    """
    def __init__(self, variables: ArrayLike, data: np.ndarray, labels: Optional[dict] = None,
                 tags: Optional[dict] = None):
        self.variables = variables
        self.data = np.reshape(data, [len(v) for v in variables])

        self.labels = with_defaults(labels, title="", axes=[], data="data")
        # alias the first 3 axes to x, y, z for compatibility with Sweep labels
        for axis, label in zip('xyz', self.labels['axes']):
            self.labels[axis] = label

        self.tags = tags