"""Processing and presentation of the spatial results"""
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Optional, Union, Tuple, Dict, List
import matplotlib

from ..utils.misc import with_defaults
from ..utils import pltutils
from ..support.pickle import pickleable
from ..support.structure import Positions, AbstractSites, Sites, Hoppings
from matplotlib.collections import LineCollection, TriMesh, PolyCollection
from matplotlib.tri import TriContourSet
from .series import Series

__all__ = ['SpatialMap', 'StructureMap', 'Positions', 'SpatialLDOS', 'Structure']


@pickleable
class AbstractStructure:
    """Abstract class to be used for SpatialMap and Structure"""
    def __init__(self, positions_sites: Union[ArrayLike, AbstractSites, Sites, '_CppSites'],
                 sublattices: Optional[ArrayLike] = None):
        if sublattices is None and isinstance(positions_sites, AbstractSites):
            self._sites = positions_sites
        else:
            self._sites = Sites(positions_sites, sublattices)

    @property
    def num_sites(self) -> int:
        """Total number of lattice sites"""
        return self._sites.size

    @property
    def positions(self) -> Positions:
        """Lattice site positions. Named tuple with x, y, z fields, each a 1D array."""
        return self._sites.positions

    @property
    def xyz(self) -> np.ndarray:
        """Return a new array with shape=(N, 3). Convenient, but slow for big systems."""
        return np.array(self.positions).T

    @property
    def x(self) -> np.ndarray:
        """1D array of coordinates, short for :attr:`.positions.x <.SpatialMap.positions.x>`"""
        return self._sites.x

    @property
    def y(self) -> np.ndarray:
        """1D array of coordinates, short for :attr:`.positions.y <.SpatialMap.positions.y>`"""
        return self._sites.y

    @property
    def z(self) -> np.ndarray:
        """1D array of coordinates, short for :attr:`.positions.z <.SpatialMap.positions.z>`"""
        return self._sites.z

    @property
    def sublattices(self) -> np.ndarray:
        """1D array of sublattices IDs"""
        return self._sites.ids

    @property
    def sub(self) -> np.ndarray:
        """1D array of sublattices IDs, short for :attr:`.sublattices <.SpatialMap.sublattices>`"""
        return self._sites.ids

    def __getitem__(self, idx: Union[int, List[int]]) -> 'AbstractStructure':
        """Same rules as numpy indexing"""
        if hasattr(idx, "contains"):
            idx = idx.contains(*self.positions)  # got a Shape object -> evaluate it
        return self.__class__(self._sites[idx])

    def cropped(self, **limits) -> 'AbstractStructure':
        """Return a copy which retains only the sites within the given limits

        Parameters
        ----------
        **limits
            Attribute names and corresponding limits. See example.

        Examples
        --------
        Leave only the data where -10 <= x < 10 and 2 <= y < 4::

            new = original.cropped(x=[-10, 10], y=[2, 4])
        """
        return self[self._make_crop_indices(limits)]

    def _make_crop_indices(self, limits: Dict[str, List[int]]) -> np.ndarray:
        """Return the indices into `obj` which retain only the data within the given limits"""
        idx = np.ones(self.num_sites, dtype=bool)
        for name, limit in limits.items():
            v = getattr(self, name)
            idx = np.logical_and(idx, v >= limit[0])
            idx = np.logical_and(idx, v < limit[1])
        return idx


@pickleable
class SpatialMap(AbstractStructure):
    """Represents some spatially dependent property: data mapped to site positions"""
    def __init__(self, data: np.ndarray, positions: Union[ArrayLike, AbstractSites],
                 sublattices: Optional[ArrayLike] = None):
        self.data = np.atleast_1d(data)
        if sublattices is None and isinstance(positions, AbstractSites):
            super().__init__(positions)
        else:
            super().__init__(Sites(positions, sublattices))

        if self.num_sites != data.size:
            raise RuntimeError("Data size doesn't match number of sites")

    @property
    def data(self) -> np.ndarray:
        """1D array of values for each site, i.e. maps directly to x, y, z site coordinates"""
        return self._data

    @data.setter
    def data(self, value: ArrayLike):
        self._data = value

    def with_data(self, data) -> 'SpatialMap':
        """Return a copy of this object with different data mapped to the sites"""
        result = copy(self)
        result._data = data
        return result

    def save_txt(self, filename: str):
        with open(filename + '.dat', 'w') as file:
            file.write('# {:12}{:13}{:13}\n'.format('x(nm)', 'y(nm)', 'data'))
            for x, y, d in zip(self.x, self.y, self.data):
                file.write(("{:13.5e}" * 3 + '\n').format(x, y, d))

    def __getitem__(self, idx: Union[int, ArrayLike]) -> 'SpatialMap':
        """Same rules as numpy indexing"""
        if hasattr(idx, "contains"):
            idx = idx.contains(*self.positions)  # got a Shape object -> evaluate it
        return self.__class__(self._data[idx], self._sites[idx])

    def cropped(self, **limits) -> 'SpatialMap':
        """Return a copy which retains only the sites within the given limits

        Parameters
        ----------
        **limits
            Attribute names and corresponding limits. See example.

        Examples
        --------
        Leave only the data where -10 <= x < 10 and 2 <= y < 4::

            new = original.cropped(x=[-10, 10], y=[2, 4])
        """
        return self[self._make_crop_indices(limits)]

    def clipped(self, v_min: int, v_max: int) -> 'SpatialMap':
        """Clip (limit) the values in the `data` array, see :func:`~numpy.clip`"""
        return self.with_data(np.clip(self.data, v_min, v_max))

    def convolve(self, sigma: float = 0.25) -> np.ndarray:
        # TODO: slow and only works in the xy-plane
        x, y, _ = self.positions
        r = np.sqrt(x**2 + y**2)

        data = np.empty_like(self.data)
        for i in range(len(data)):
            idx = np.abs(r - r[i]) < sigma
            data[i] = np.sum(self.data[idx] * np.exp(-0.5 * ((r[i] - r[idx]) / sigma)**2))
            data[i] /= np.sum(np.exp(-0.5 * ((r[i] - r[idx]) / sigma)**2))

        self._data = data

    @staticmethod
    def _decorate_plot(ax: Optional[plt.Axes] = None):
        if ax is None:
            ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        pltutils.despine(trim=True, ax=ax)

    def plot_pcolor(self, ax: Optional[plt.Axes] = None, **kwargs) -> Union[TriMesh, PolyCollection]:
        """Color plot of the xy plane

        Parameters
        ----------
        ax : Optional[plt.Axes]
            The axis to plot on.
        **kwargs
            Forwarded to :func:`~matplotlib.pyplot.tripcolor`.
        """
        if ax is None:
            ax = plt.gca()
        x, y, _ = self.positions
        kwargs = with_defaults(kwargs, shading='gouraud', rasterized=True)
        pcolor = ax.tripcolor(x, y, self.data, **kwargs)
        self._decorate_plot(ax=ax)
        return pcolor

    def plot_contourf(self, num_levels: int = 50, ax: Optional[plt.Axes] = None, **kwargs) -> TriContourSet:
        """Filled contour plot of the xy plane

        Parameters
        ----------
        num_levels : int
            Number of contour levels.
        ax : Optional[plt.Axes]
            The axis to plot on.
        **kwargs
            Forwarded to :func:`~matplotlib.pyplot.tricontourf`.
        """
        if ax is None:
            ax = plt.gca()
        levels = np.linspace(self.data.min(), self.data.max(), num=num_levels)
        x, y, _ = self.positions
        kwargs = with_defaults(kwargs, levels=levels)
        contourf = ax.tricontourf(x, y, self.data, **kwargs)
        # Each collection has to be rasterized, `tricontourf()` does not accept `rasterized=True`
        for collection in contourf.collections:
            collection.set_rasterized(True)
        self._decorate_plot(ax=ax)
        return contourf

    def plot_contour(self, ax: Optional[plt.Axes] = None, **kwargs) -> TriContourSet:
        """Contour plot of the xy plane

        Parameters
        ----------
        ax : Optional[plt.Axes]
            The axis to plot on.
        **kwargs
            Forwarded to :func:`~matplotlib.pyplot.tricontour`.
        """
        if ax is None:
            ax = plt.gca()
        x, y, _ = self.positions
        contour = ax.tricontour(x, y, self.data, **kwargs)
        self._decorate_plot(ax=ax)
        return contour


@pickleable
class StructureMap(SpatialMap):
    """A subclass of :class:`.SpatialMap` that also includes hoppings between sites"""

    def __init__(self, data: ArrayLike, sites: Sites, hoppings: Hoppings, boundaries=()):
        super().__init__(data, sites)
        self._hoppings = hoppings
        self._boundaries = boundaries

    @property
    def spatial_map(self) -> SpatialMap:
        """Just the :class:`SpatialMap` subset without hoppings"""
        return SpatialMap(self._data, self._sites)

    @property
    def hoppings(self) -> Hoppings:
        """Sparse matrix of hopping IDs"""
        return self._hoppings

    @property
    def boundaries(self) -> list:
        """Boundary hoppings between different translation units (only for infinite systems)"""
        return self._boundaries

    def __getitem__(self, idx: Union[int, List[int]]) -> 'StructureMap':
        """Same rules as numpy indexing"""
        if hasattr(idx, "contains"):
            idx = idx.contains(*self.positions)  # got a Shape object -> evaluate it
        return self.__class__(self.data[idx], self._sites[idx], self._hoppings[idx],
                              [b[idx] for b in self._boundaries])

    def with_data(self, data) -> "StructureMap":
        """Return a copy of this object with different data mapped to the sites"""
        result = copy(self)
        result._data = data
        return result

    def plot(self, cmap: str = 'YlGnBu', site_radius: Tuple[float, float] = (0.03, 0.05), num_periods: int = 1,
             ax: Optional[plt.Axes] = None, **kwargs) -> Optional[matplotlib.collections.CircleCollection]:
        """Plot the spatial structure with a colormap of :attr:`data` at the lattice sites

        Both the site size and color are used to display the data.

        Parameters
        ----------
        cmap : str
            Matplotlib colormap to be used for the data.
        site_radius : Tuple[float, float]
            Min and max radius of lattice sites. This range will be used to visually
            represent the magnitude of the data.
        num_periods : int
            Number of times to repeat periodic boundaries.
        ax : Optional[plt.Axes]
            The axis to plot on.
        **kwargs
            Additional plot arguments as specified in :func:`.structure_plot_properties`.
        """
        if ax is None:
            ax = plt.gca()
        from ..system import (plot_sites, plot_hoppings, plot_periodic_boundaries,
                              structure_plot_properties, decorate_structure_plot)

        def to_radii(data: np.ndarray) -> Union[float, tuple, list]:
            if not isinstance(site_radius, (tuple, list)):
                return site_radius

            positive_data = data - data.min()
            maximum = positive_data.max()
            if not np.allclose(maximum, 0):
                delta = site_radius[1] - site_radius[0]
                return site_radius[0] + delta * positive_data / maximum
            else:
                return site_radius[1]

        props = structure_plot_properties(**kwargs)
        props['site'] = with_defaults(props['site'], radius=to_radii(self.data), cmap=cmap)
        collection = plot_sites(self.positions, self.data, **props['site'], ax=ax)

        hop = self.hoppings.tocoo()
        props['hopping'] = with_defaults(props['hopping'], color='#bbbbbb')
        plot_hoppings(self.positions, hop, **props['hopping'], ax=ax)

        props['site']['alpha'] = props['hopping']['alpha'] = 0.5
        plot_periodic_boundaries(self.positions, hop, self.boundaries, self.data,
                                 num_periods, **props, ax=ax)

        decorate_structure_plot(**props, ax=ax)

        if collection:
            ax._sci(collection)
            # dirty, but it is the same as plt.sci()
        return collection


@pickleable
class Structure(AbstractStructure):
    """Holds and plots the structure of a tight-binding system

    Similar to :class:`StructureMap`, but only holds the structure without
    mapping to any actual data.
    """
    def __init__(self, sites: Union[Sites, '_CppSites'], hoppings: Hoppings, boundaries=()):
        super().__init__(sites)
        self._hoppings = hoppings
        self._boundaries = boundaries

    @property
    def hoppings(self) -> Hoppings:
        """Sparse matrix of hopping IDs"""
        return self._hoppings

    @property
    def boundaries(self) -> list:
        """Boundary hoppings between different translation units (only for infinite systems)"""
        return self._boundaries

    def __getitem__(self, idx: Union[int, List[int]]) -> 'Structure':
        """Same rules as numpy indexing"""
        if hasattr(idx, "contains"):
            idx = idx.contains(*self.positions)  # got a Shape object -> evaluate it

        sliced = Structure(self._sites[idx], self._hoppings[idx],
                           [b[idx] for b in self._boundaries])
        if hasattr(self, "lattice"):
            sliced.lattice = self.lattice
        return sliced

    def find_nearest(self, position: ArrayLike, sublattice: str = "") -> int:
        """Find the index of the atom closest to the given position

        Parameters
        ----------
        position : array_like
            Where to look.
        sublattice : Optional[str]
            Look for a specific sublattice site. By default any will do.

        Returns
        -------
        int
        """
        return self._sites.find_nearest(position, sublattice)

    def cropped(self, **limits) -> 'Structure':
        """Return a copy which retains only the sites within the given limits

        Parameters
        ----------
        **limits
            Attribute names and corresponding limits. See example.

        Examples
        --------
        Leave only the data where -10 <= x < 10 and 2 <= y < 4::

            new = original.cropped(x=[-10, 10], y=[2, 4])
        """
        return self[self._make_crop_indices(limits)]

    def with_data(self, data: ArrayLike) -> StructureMap:
        """Map some data to this structure"""
        return StructureMap(data, self._sites, self._hoppings, self._boundaries)

    def plot(self, num_periods: int = 1, ax: Optional[plt.Axes] = None,
             **kwargs) -> Optional[matplotlib.collections.CircleCollection]:
        """Plot the structure: sites, hoppings and periodic boundaries (if any)

        Parameters
        ----------
        num_periods : int
            Number of times to repeat the periodic boundaries.
        ax : Optional[plt.Axes]
            The axis to plot on.
        **kwargs
            Additional plot arguments as specified in :func:`.structure_plot_properties`.
        """
        if ax is None:
            ax = plt.gca()
        from ..system import (plot_sites, plot_hoppings, plot_periodic_boundaries,
                             structure_plot_properties, decorate_structure_plot)

        props = structure_plot_properties(**kwargs)
        if hasattr(self, "lattice"):
            props["site"].setdefault("radius", self.lattice.site_radius_for_plot())

        plot_hoppings(self.positions, self._hoppings, **props['hopping'], ax=ax)
        collection = plot_sites(self.positions, self.sublattices, **props['site'], ax=ax)
        plot_periodic_boundaries(self.positions, self._hoppings, self._boundaries,
                                 self.sublattices, num_periods, **props, ax=ax)

        decorate_structure_plot(**props, ax=ax)
        return collection

class SpatialLDOS:
    """Holds the results of :meth:`KPM.calc_spatial_ldos`

    It behaves like a product of a :class:`.Series` and a :class:`.StructureMap`.
    """

    def __init__(self, data: np.ndarray, energy: np.ndarray, structure: Structure):
        self.data = data
        self.energy = energy
        self.structure = structure

    def structure_map(self, energy: float) -> StructureMap:
        """Return a :class:`.StructureMap` of the spatial LDOS at the given energy

        Parameters
        ----------
        energy : float
            Produce a structure map for LDOS data closest to this energy value.

        Returns
        -------
        :class:`.StructureMap`
        """
        idx = np.argmin(abs(self.energy - energy))
        return self.structure.with_data(self.data[idx])

    def ldos(self, position: ArrayLike, sublattice: str = "") -> Series:
        """Return the LDOS as a function of energy at a specific position

        Parameters
        ----------
        position : array_like
        sublattice : Optional[str]

        Returns
        -------
        :class:`.Series`
        """
        idx = self.structure.find_nearest(position, sublattice)
        return Series(self.energy, self.data[:, idx],
                      labels=dict(variable="E (eV)", data="LDOS", columns="orbitals"))