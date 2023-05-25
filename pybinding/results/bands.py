"""Processing and presentation of energy data"""
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Optional, Tuple, List
import matplotlib

from ..utils import with_defaults, x_pi, pltutils
from ..support.pickle import pickleable
from .path import Path, Area
from .series import Series
from matplotlib.collections import LineCollection, PathCollection

__all__ = ['Bands', 'FatBands', 'Eigenvalues', 'BandsArea', 'FatBandsArea']


@pickleable
class Eigenvalues:
    """Hamiltonian eigenvalues with optional probability map

    Attributes
    ----------
    values : np.ndarray
    probability : np.ndarray
    """
    def __init__(self, eigenvalues: np.ndarray, probability: Optional[np.ndarray] = None):
        self.values = np.atleast_1d(eigenvalues)
        self.probability = np.atleast_1d(probability)

    @property
    def indices(self) -> np.ndarray:
        return np.arange(0, self.values.size)

    def _decorate_plot(self, mark_degenerate: bool, number_states: bool, margin: float = 0.1,
                       ax: Optional[plt.Axes] = None) -> None:
        """Common elements for the two eigenvalue plots"""
        if ax is None:
            ax = plt.gca()
        if mark_degenerate:
            # draw lines between degenerate states
            from ..solver import Solver
            from matplotlib.collections import LineCollection
            pairs = ((s[0], s[-1]) for s in Solver.find_degenerate_states(self.values))
            lines = [[(i, self.values[i]) for i in pair] for pair in pairs]
            ax.add_collection(LineCollection(lines, color='black', alpha=0.5))

        if number_states:
            # draw a number next to each state
            for index, energy in enumerate(self.values):
                pltutils.annotate_box(index, (index, energy), fontsize='x-small',
                                      xytext=(0, -10), textcoords='offset points', ax=ax)
            margin = 0.25

        ax.set_xlabel('state')
        ax.set_ylabel('E (eV)')
        ax.set_xlim(-1, len(self.values))
        pltutils.despine(trim=True, ax=ax)
        pltutils.add_margin(margin, axis="y", ax=ax)

    def plot(self, mark_degenerate: bool = True, show_indices: bool = False, ax: Optional[plt.Axes] = None,
             **kwargs) -> matplotlib.collections.PathCollection:
        """Standard eigenvalues scatter plot

        Parameters
        ----------
        mark_degenerate : bool
            Plot a line which connects degenerate states.
        show_indices : bool
            Plot index number next to all states.
        ax : Optional[plt.Axes]
            The axis to plot on.
        **kwargs
            Forwarded to plt.scatter().
        """
        if ax is None:
            ax = plt.gca()
        collection = ax.scatter(self.indices, self.values, **with_defaults(kwargs, c='#377ec8', s=15, lw=0.1))
        self._decorate_plot(mark_degenerate, show_indices, ax=ax)
        return collection

    def plot_heatmap(self, size: tuple[int, int] = (7, 77), mark_degenerate: bool = True, show_indices: bool = False,
                     ax: Optional[plt.Axes] = None, **kwargs) -> Optional[float]:
        """Eigenvalues scatter plot with a heatmap indicating probability density

        Parameters
        ----------
        size : Tuple[int, int]
            Min and max scatter dot size.
        mark_degenerate : bool
            Plot a line which connects degenerate states.
        show_indices : bool
            Plot index number next to all states.
        ax : Optional[plt.Axes]
            The axis to plot on.
        **kwargs
            Forwarded to plt.scatter().
        """
        if ax is None:
            ax = plt.gca()
        if not np.any(self.probability):
            self.plot(mark_degenerate, show_indices, **kwargs, ax=ax)
            return 0

        # higher probability states should be drawn above lower ones
        idx = np.argsort(self.probability)
        indices, energy, probability = (v[idx] for v in
                                        (self.indices, self.values, self.probability))

        scatter_point_sizes = size[0] + size[1] * probability / probability.max()
        ax.scatter(indices, energy, **with_defaults(kwargs, cmap='YlOrRd', lw=0.2, alpha=0.85,
                                                    c=probability, s=scatter_point_sizes,
                                                    edgecolor="k"))

        self._decorate_plot(mark_degenerate, show_indices)
        return self.probability.max()


@pickleable
class Bands:
    """Band structure along a path in k-space

    Attributes
    ----------
    k_path : :class:`Path`
        The path in reciprocal space along which the bands were calculated.
        E.g. constructed using :func:`make_path`.
    energy : array_like
        Energy values for the bands along the path in k-space.
    """
    def __init__(self, k_path: Path, energy: np.ndarray):
        self.k_path: Path = np.atleast_1d(k_path).view(Path)
        self.energy: np.ndarray = np.atleast_2d(energy).T if np.ndim(energy) == 1 else np.atleast_2d(energy)

    def _point_names(self, k_points: list[float]) -> list[str]:
        names = []
        if self.k_path.point_labels:
            return self.k_path.point_labels
        for k_point in k_points:
            k_point = np.atleast_1d(k_point)
            values = map(x_pi, k_point)
            fmt = "[{}]" if len(k_point) > 1 else "{}"
            names.append(fmt.format(', '.join(values)))
        return names

    @property
    def num_bands(self) -> int:
        return self.energy.shape[1]

    def plot(self, point_labels: Optional[List[str]] = None, ax: Optional[plt.Axes] = None,
             **kwargs) -> Optional[List[plt.Line2D]]:
        """Line plot of the band structure

        Parameters
        ----------
        point_labels : Optional[List[str]]
            Labels for the `k_points`.
        ax : Optional[plt.Axes]
            The Axis to plot the bands on.
        **kwargs
            Forwarded to `plt.plot()`.
        """
        if ax is None:
            ax = plt.gca()
        default_color = pltutils.get_palette('Set1')[1]
        default_linewidth = np.clip(5 / self.num_bands, 1.1, 1.6)
        kwargs = with_defaults(kwargs, color=default_color, lw=default_linewidth)

        k_space = self.k_path.as_1d()
        lines_out = ax.plot(k_space, self.energy, **kwargs)

        self._decorate_plot(point_labels, ax)
        return lines_out

    def _decorate_plot(self, point_labels: Optional[List[str]] = None, ax: Optional[plt.Axes] = None) -> None:
        """Decorate the band structure

        Parameters
        ----------
        point_labels : Optional[List[str]]
            Labels for the `k_points`.
        ax : Optional[plt.Axes]
            The Axis to plot the bands on.
        """
        if ax is None:
            ax = plt.gca()

        k_space = self.k_path.as_1d()

        ax.set_xlim(k_space.min(), k_space.max())
        ax.set_xlabel('k-space')
        ax.set_ylabel('E (eV)')
        pltutils.add_margin(ax=ax)
        pltutils.despine(trim=True, ax=ax)

        point_labels = point_labels or self._point_names(self.k_path.points)
        assert len(point_labels) == len(self.k_path.point_indices), \
            "The length of point_labels and point_indices aren't the same, len({0}) != len({1})".format(
                point_labels, self.k_path.point_indices
            )
        ax.set_xticks(k_space[self.k_path.point_indices], point_labels)

        # Draw vertical lines at significant points. Because of the `transLimits.transform`,
        # this must be the done last, after all others plot elements are positioned.
        for idx in self.k_path.point_indices:
            ymax = ax.transLimits.transform([0, np.nanmax(self.energy[idx])])[1]
            ax.axvline(k_space[idx], ymax=ymax, color="0.4", lw=0.8, ls=":", zorder=-1)

    def plot_kpath(self, point_labels: Optional[List[str]] = None, **kwargs) -> None:
        """Quiver plot of the k-path along which the bands were computed

        Combine with :meth:`.Lattice.plot_brillouin_zone` to see the path in context.

        Parameters
        ----------
        point_labels : Optional[List[str]]
            Labels for the k-points.
        **kwargs
            Forwarded to :func:`~matplotlib.pyplot.quiver`.
        """
        self.k_path.plot(point_labels, **kwargs)

    def dos(self, energies: Optional[ArrayLike] = None, broadening: Optional[float] = None) -> Series:
        r"""Calculate the density of states as a function of energy

        .. math::
            \text{DOS}(E) = \frac{1}{c \sqrt{2\pi}}
                            \sum_n{e^{-\frac{(E_n - E)^2}{2 c^2}}}

        for each :math:`E` in `energies`, where :math:`c` is `broadening` and
        :math:`E_n` is `eigenvalues[n]`.

        Parameters
        ----------
        energies : array_like
            Values for which the DOS is calculated. Default: min/max from Bands().energy, subdivided in 100 parts [ev].
        broadening : float
            Controls the width of the Gaussian broadening applied to the DOS. Default: 0.05 [ev].
        Returns
        -------
        :class:`~pybinding.Series`
        """
        if energies is None:
            energies = np.linspace(np.nanmin(self.energy), np.nanmax(self.energy), 100)
        if broadening is None:
            broadening = (np.nanmax(self.energy) - np.nanmin(self.energy)) / 100
        scale = 1 / (broadening * np.sqrt(2 * np.pi) * self.energy.shape[0])
        dos = np.zeros(len(energies))
        for eigenvalue in self.energy:
            delta = eigenvalue[:, np.newaxis] - energies
            dos += scale * np.sum(np.exp(-0.5 * delta**2 / broadening**2), axis=0)
        return Series(energies, dos, labels=dict(variable="E (eV)", data="DOS"))


@pickleable
class FatBands(Bands):
    """Band structure with data per k-point, like SOC or pDOS

    Parameters
    ----------
    bands : :class:`Bands`
        The bands on wich the data is written
    data : array_like
        An array of values wich were computed as a function of the bands.k_path.
        It can be 2D or 3D. In the latter case each column represents the result
        of a different function applied to the same `variable` input.
    labels : dict
        Plot labels: 'data', 'title' and 'columns'.
    """
    def __init__(self, bands: Bands, data: ArrayLike, labels: Optional[dict] = None):
        super().__init__(bands.k_path, bands.energy)
        self.data = np.atleast_2d(data)
        self.labels = with_defaults(
            labels, variable="E (eV)", data="pDOS", columns="Orbitals", title="",
            orbitals=[str(i) for i in range(self.data.shape[2])] if self.data.ndim == 3 else []
        )

    def with_data(self, data: np.ndarray) -> 'FatBands':
        """Return a copy of this result object with different data"""
        result = copy(self)
        result.data = data
        return result

    def __add__(self, other: 'FatBands') -> 'FatBands':
        """Add together the data of two FatBands object in a new object."""
        if self.data.ndim < other.data.ndim:
            # keep information about the orbitals, so take the other series as a reference
            return other.with_data(self.data[:, :, np.newaxis] + other.data)
        elif self.data.ndim > other.data.ndim:
            return self.with_data(self.data + other.data[:, :, np.newaxis])
        else:
            return self.with_data(self.data + other.data)

    def __sub__(self, other: 'FatBands') -> 'FatBands':
        """Subtract the data of two FatBands object in a new object."""
        if self.data.ndim < other.data.ndim:
            # keep information about the orbitals, so take the other series as a reference
            return other.with_data(self.data[:, :, np.newaxis] - other.data)
        elif self.data.ndim > other.data.ndim:
            return self.with_data(self.data - other.data[:, :, np.newaxis])
        else:
            return self.with_data(self.data - other.data)

    def reduced(self, columns: Optional[List[int]] = None, orbitals: Optional[List[str]] = None,
                fill_other: float = 0.) -> 'FatBands':
        """Return a copy where the data is summed over the columns

        Only applies to results which may have multiple columns of data, e.g.
        results for multiple orbitals for LDOS calculation.

        Parameters
        ----------
        columns : Optional[List[int]]
            The colummns to contract to the new array.
            The length of `columns` agrees with the dimensions of data.shape[2].
            The value at each position corresponds to the new column of the new Series object
        orbitals: Optional[List[str]]
            Optional new list of entries for the `orbitals` label in `labels`
        fill_other : float
            In case an array is made with a new column, fill it with this value. Default: 0.
        """
        if columns is None:
            columns = np.zeros(self.data.shape[2])
        data = self.data
        if data.ndim == 2:
            data = self.data[:, np.newaxis]
        col_idx = np.array(columns, dtype=int)
        if np.all(col_idx == 0):
            # case where all the axis are summed over, no 'orbital' label is needed
            return self.with_data(data.sum(axis=2))
        col_max = np.max(col_idx) + 1
        if orbitals is None:
            orb_list = [str(i) for i in range(col_max)]
            for c_i in np.unique(col_idx):
                orb_list[c_i] = self.labels["orbitals"][np.argmax(col_idx == c_i)]
        else:
            orb_list = orbitals
        data_out = np.full((data.shape[0], data.shape[1], col_max), fill_other)
        for c_i in np.unique(col_idx):
            data_out[:, :, c_i] = np.nansum(data[:, :, col_idx == c_i], axis=2)
        fatbands_out = self.with_data(data_out)
        fatbands_out.labels["orbitals"] = orb_list
        return fatbands_out

    def plot(self, point_labels: Optional[List[str]] = None, ax: Optional[plt.Axes] = None, legend: bool = True,
             **kwargs) -> Optional[List[PathCollection]]:
        """Line plot of the band structure with the given data

        Parameters
        ----------
        point_labels : Optional[List[str]]
            Labels for the `k_points`.
        ax : Optional[plt.Axes]
            The Axis to plot the bands on.
        legend : bool
            Plot the legend of the bands on the axes.
        **kwargs
            Forwarded to `plt.plot()`.
        """
        if ax is None:
            ax = plt.gca()
        k_space = np.ones(self.energy.shape) * self.k_path.as_1d()[:, np.newaxis]
        lines = []
        data_length = self.data.shape[2] if self.data.ndim == 3 else 1
        for d_i in range(data_length):
            lines.append(ax.scatter(
                k_space,
                self.energy,
                s=np.nan_to_num(np.abs(self.data[:, :, d_i]) if self.data.ndim == 3 else self.data) * 20,
                alpha=0.5,
                **kwargs
            ))
        if legend:
            ax.legend(lines, self.labels["orbitals"], title=self.labels["columns"])
        ax.set_title(self.labels["title"])
        self._decorate_plot(point_labels, ax)
        return lines

    def plot_bands(self, **kwargs) -> List[plt.Line2D]:
        """Line plot of the band structure like in Bands."""
        return super().plot(**kwargs)

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
        k_space = self.k_path.as_1d()
        data = self.data[:, :, idx] if self.data.ndim == 3 else self.data
        ax.set_xlim(np.nanmin(k_space), np.nanmax(k_space))
        ax.set_ylim(np.nanmin(self.energy), np.nanmax(self.energy))
        ax.set_title(self.labels["title"])
        line = pltutils.plot_color(k_space, self.energy, data[:-1, :], ax, **with_defaults(kwargs, cmap='RdYlBu_r'))
        self._decorate_plot(point_labels, ax)
        if plot_colorbar:
            self.colorbar(ax=ax, label=self.labels["orbitals"][idx])
        return line

    def colorbar(self, **kwargs):
        """Draw a colorbar with the label of :attr:`Sweep.data`"""
        return pltutils.colorbar(**with_defaults(kwargs, label=self.labels["data"]))

    def dos(self, energies: Optional[ArrayLike] = None, broadening: Optional[float] = None) -> Series:
        r"""Calculate the density of states as a function of energy

        .. math::
            \text{DOS}(E) = \frac{1}{c \sqrt{2\pi}}
                            \sum_n{e^{-\frac{(E_n - E)^2}{2 c^2}}}

        for each :math:`E` in `energies`, where :math:`c` is `broadening` and
        :math:`E_n` is `eigenvalues[n]`.

        Parameters
        ----------
        energies : array_like
            Values for which the DOS is calculated. Default: min/max from Bands().energy, subdivided in 100 parts [ev].
        broadening : float
            Controls the width of the Gaussian broadening applied to the DOS. Default: 0.05 [ev].

        Returns
        -------
        :class:`~pybinding.Series`
        """
        if energies is None:
            energies = np.linspace(np.nanmin(self.energy), np.nanmax(self.energy), 100)
        if broadening is None:
            broadening = (np.nanmax(self.energy) - np.nanmin(self.energy)) / 100
        scale = 1 / (broadening * np.sqrt(2 * np.pi) * self.energy.shape[0])
        data = self.data if self.data.ndim == 3 else self.data[:, :, np.newaxis]
        dos = np.zeros((data.shape[2], len(energies)))
        for i_k, eigenvalue in enumerate(self.energy):
            delta = np.nan_to_num(eigenvalue[:, np.newaxis]) - energies
            gauss = np.exp(-0.5 * delta**2 / broadening**2)
            datal = np.nan_to_num(data[i_k])
            dos += scale * np.sum(datal[:, :, np.newaxis] * gauss[:, np.newaxis, :], axis=0)
        return Series(energies, dos.T, labels=self.labels)


@pickleable
class BandsArea(Bands):
    """Band structure alond an area in k-space

    Parameters
    ----------
    k_area : :class:`Area`
        The Area in reciprocal space for which the bands were calculated.
        E.g. constructed using :func:`make_area`.
    energy : array_like
        Energy values for the bands along the path in k-space.
    """
    def __init__(self, k_area: Area, energy: ArrayLike):
        self.k_dims = np.shape(k_area)
        k_path: Path = Path(
            np.atleast_1d(k_area.reshape(np.prod(self.k_dims[:2]), -1)),
            k_area.point_indices,
            k_area.point_labels
        )
        super().__init__(k_path, self.area_to_list(energy))

    @property
    def energy_area(self) -> np.ndarray:
        return self.list_to_area(self.energy)

    @energy_area.setter
    def energy_area(self, energy: np.ndarray):
        self.energy = self.area_to_list(np.atleast_2d(energy))

    def area_to_list(self, data: ArrayLike) -> np.ndarray:
        data_size = [self.k_dims[0] * self.k_dims[1]]
        data = np.atleast_2d(data)
        for ds in data.shape[2:]:
            data_size.append(ds)
        return data.reshape(data_size)

    def list_to_area(self, data: ArrayLike) -> np.ndarray:
        data_size = [self.k_dims[0], self.k_dims[1]]
        data = np.atleast_1d(data)
        for ds in data.shape[1:]:
            data_size.append(ds)
        return data.reshape(data_size)

    @property
    def k_area(self) -> Area:
        return Area(
            self.k_path.reshape(self.k_dims[0], self.k_dims[1], -1),
            self.k_path.point_indices,
            self.k_path.point_labels
        )

    def plot_karea(self, point_labels: Optional[List[str]] = None, **kwargs) -> None:
        """Scatter plot of the k-area along which the bands were computed

        Combine with :meth:`.Lattice.plot_brillouin_zone` to see the path in context.

        Parameters
        ----------
        point_labels : Optional[List[str]]
            Labels for the k-points.
        **kwargs
            Forwarded to :func:`~matplotlib.pyplot.scatter`.
        """
        self.k_area.plot(point_labels, **kwargs)
        # TODO: add a function to make a area-plot for the results


@pickleable
class FatBandsArea(BandsArea, FatBands):
    """Band structure with data per k-point, like SOC or pDOS

    Parameters
    ----------
    bands : :class:`BandsArea`
        The bands on wich the data is written
    data : array_like
        An array of values wich were computed as a function of the bands.k_path.
        It can be 2D or 3D. In the latter case each column represents the result
        of a different function applied to the same `variable` input.
    labels : dict
        Plot labels: 'data', 'title' and 'columns'.
    """
    def __init__(self, bands: BandsArea, data: ArrayLike, labels: Optional[dict] = None):
        super().__init__(bands.k_area, bands.energy_area)
        self.data_area = np.atleast_3d(data)
        self.labels = with_defaults(
            labels, variable="E (eV)", data="pDOS", columns="Orbitals", title="",
            orbitals=[str(i) for i in range(self.data.shape[2])] if self.data.ndim == 3 else []
        )

    @property
    def data_area(self) -> np.ndarray:
        return self.list_to_area(self.data)

    @data_area.setter
    def data_area(self, data: np.ndarray):
        self.data = self.area_to_list(data)
