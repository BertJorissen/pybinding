"""Main model definition interface"""
import numpy as np
from scipy.sparse import csr_matrix
from numpy.typing import ArrayLike
from typing import Literal

from . import _cpp
from . import results
from .system import System, decorate_structure_plot
from .lattice import Lattice
from .leads import Leads

__all__ = ['Model']


class Model(_cpp.Model):
    """Builds a Hamiltonian from lattice, shape, symmetry and modifier parameters

    The most important attributes are :attr:`.system` and :attr:`.hamiltonian` which are
    constructed based on the input parameters. The :class:`.System` contains structural
    data like site positions. The tight-binding Hamiltonian is a sparse matrix in the
    :class:`.scipy.sparse.csr_matrix` format.

    Parameters
    ----------
    lattice : :class:`~pybinding.Lattice`
        The lattice specification.
    *args
        Can be any of: shape, symmetry or various modifiers. Note that:

        * There can be at most one shape and at most one symmetry. Shape and symmetry
          can be composed as desired, but physically impossible scenarios will result
          in an empty system.
        * Any number of modifiers can be added. Adding the same modifier more than once
          is allowed: this will usually multiply the modifier's effect.
    """
    def __init__(self, lattice: Lattice, *args):
        super().__init__(lattice.impl)

        self._lattice = lattice
        self._shape = None
        self.add(*args)

    def add(self, *args) -> None:
        """Add parameter(s) to the model

        Parameters
        ----------
        *args
            Any of: shape, symmetry, modifiers. Tuples and lists of parameters are expanded
            automatically, so `M.add(p0, [p1, p2])` is equivalent to `M.add(p0, p1, p2)`.
        TODO: Add types
        """
        for arg in args:
            if arg is None:
                raise RuntimeError("`None` was passed to Model: check that all "
                                   "modifier functions have return values")
            try:
                self.add(*arg)
            except TypeError:
                super().add(arg)
                if isinstance(arg, _cpp.Shape):
                    self._shape = arg

    def attach_lead(self, direction: int, contact: _cpp.Shape) -> None:
        """Attach a lead to the main system

        Not valid for 1D lattices.

        Parameters
        ----------
        direction : int
            Lattice vector direction of the lead. Must be one of: 1, 2, 3, -1, -2, -3.
            For example, `direction=2` would create a lead which intersects the main system
            in the :math:`a_2` lattice vector direction. Setting `direction=-2` would create
            a lead on the opposite side of the system, but along the same lattice vector.
        contact : :class:`~_pybinding.Shape`
            The place where the lead should contact the main system. For a 2D lattice it's
            just a :func:`.line` describing the intersection of the lead and the system.
            For a 3D lattice it's the area described by a 2D :class:`.FreeformShape`.
        """
        super().attach_lead(direction, contact)

    def structure_map(self, data: ArrayLike) -> results.StructureMap:
        """Return a :class:`.StructureMap` of the model system mapped to the specified `data`

        Parameters
        ----------
        data : Optional[array_like]
            Data array to map to site positions.

        Returns
        -------
        :class:`~pybinding.results.StructureMap`
        """
        return self.system.with_data(data)

    def tokwant(self) -> 'KwantFiniteSystem':
        """Convert this model into `kwant <http://kwant-project.org/>`_ format (finalized)

        This is intended for compatibility with the kwant package: http://kwant-project.org/.

        Returns
        -------
        kwant.system.System
            Finalized system which can be used with kwant compute functions.
        """
        from .support.kwant import tokwant
        return tokwant(self)

    @property
    def system(self) -> System:
        """Structural data like site positions and hoppings, see :class:`.System` for details"""
        return System(super().system, self.lattice)

    @property
    def hamiltonian(self) -> csr_matrix:
        """Hamiltonian sparse matrix in the :class:`.scipy.sparse.csr_matrix` format"""
        return super().hamiltonian

    @property
    def lattice(self) -> Lattice:
        """:class:`.Lattice` specification"""
        return self._lattice
    
    @property
    def leads(self) -> Leads:
        """List of :class:`.Lead` objects"""
        return Leads(super().leads, self.lattice)

    @property
    def shape(self) -> _cpp.Shape:
        """:class:`.Polygon` or :class:`.FreeformShape` object"""
        return self._shape

    @property
    def modifiers(self) -> list:
        """List of all modifiers applied to this model"""
        return (self.state_modifiers + self.position_modifiers +
                self.onsite_modifiers + self.hopping_modifiers)

    @property
    def onsite_map(self) -> results.StructureMap:
        """:class:`.StructureMap` of the onsite energy"""
        return self.structure_map(np.real(self.hamiltonian.diagonal()))

    def plot(self, num_periods: int = 1, lead_length: int = 6,
             axes: Literal['xy', 'xz', 'yx', 'yz', 'zx', 'zy'] = 'xy', **kwargs) -> None:
        """Plot the structure of the model: sites, hoppings, boundaries and leads

        Parameters
        ----------
        num_periods : int
            Number of times to repeat the periodic boundaries.
        lead_length : int
            Number of times to repeat the lead structure.
        axes : str
            The spatial axes to plot. E.g. 'xy', 'yz', etc.
        **kwargs
            Additional plot arguments as specified in :func:`.structure_plot_properties`.
        """
        kwargs['add_margin'] = False
        self.system.plot(num_periods, axes=axes, **kwargs)
        for lead in self.leads:
            lead.plot(lead_length, axes=axes, **kwargs)
        decorate_structure_plot(axes=axes)

    def to_lattice(self) -> Lattice:
        """
        The site and position modifier is already done by the model, the energy modifiers still need to be added to the model.

        1. Get the vectors [DONE]
        2. Get the positions [DONE]
        3. Get the onsite energies [DONE]
        4. Get the hoppings within the 'unit-cell'->(0,0)
        5. Get the hoppings accross the boundaries
        6. Test if equivalent results are produced using model_to_lattice and the model itself
        """
        # get the boundaries from the system
        boundaries = self.system.boundaries

        # get the relative index from the unit cell
        boundaries_idx = [
            (bound_i,
             np.array(np.round(np.linalg.solve(np.array(self.lattice.vectors)[:, :2].T, bound.shift[:2])), dtype=int))
            for bound_i, bound in enumerate(boundaries)
        ]

        # get the indices fro the lattive vectors of the latger cell
        lat_vectors_i = [
            next(bound for bound in boundaries_idx if (bound[1][0] > 0 and bound[1][1] == 0))[0],
            next(bound for bound in boundaries_idx if (bound[1][0] == 0 and bound[1][1] > 0))[0]
        ]

        # get the vectors for the larger cell
        lat_vectors = [boundaries[lat_vectors_i[0]].shift, boundaries[lat_vectors_i[1]].shift]

        # get the relatice index shift for each of the cells
        hopping_shifts = [
            np.array(np.round(bound[1] / np.array([
                boundaries_idx[lat_vectors_i[0]][1][0], boundaries_idx[lat_vectors_i[1]][1][1]
            ])), dtype=int)
            for bound in boundaries_idx
        ]

        positions = self.system.positions

        # --- Make everything ready for the onsite-energies ---
        # given in the model.system
        sys_sublattices = self.system.sublattices

        # given in the self.lattice
        lat_sublattices = self.lattice.sublattices
        lat_sublattices = [
            (lat_sublattices[name].unique_id, lat_sublattices[name].energy, name)
            for name in lat_sublattices.keys()
        ]
        lat_sublattices.sort(key=lambda x: x[0])

        # new name for the onsite terms
        onsite_terms = [
            "{0}-{1}".format(next(subl for subl in lat_sublattices if subl[0] == subs)[2], subs_i)
            for subs_i, subs in enumerate(sys_sublattices)
        ]

        # order the energies and the positions
        onsite_x = [
            np.array(positions.x[sys_sublattices == subl[0]])[:, np.newaxis, np.newaxis]
            for subl in lat_sublattices
        ]
        onsite_y = [
            np.array(positions.y[sys_sublattices == subl[0]])[:, np.newaxis, np.newaxis]
            for subl in lat_sublattices
        ]
        onsite_z = [
            np.array(positions.z[sys_sublattices == subl[0]])[:, np.newaxis, np.newaxis]
            for subl in lat_sublattices
        ]
        onsite_energies = [
            np.array(subl[1])[np.newaxis, :, :] * np.ones(np.sum(sys_sublattices == subl[0]))[:, np.newaxis, np.newaxis]
            for subl in lat_sublattices
        ]

        onsite_id = [subl[2] for subl in lat_sublattices]
        onsite_names = [
            [onsite_terms[subl_i] for subl_i in np.arange(len(onsite_terms))[sys_sublattices == subl[0]]]
            for subl in lat_sublattices
        ]

        # apply the modifier
        # TODO: add the modifiers here from the model ebject itself
        onsite_modifier = []
        hopping_modifier = []
        if self.onsite_modifiers:
            onsite_energies = [
                onsite_modifier(energy, x, y, z, sub_id)
                for energy, x, y, z, sub_id in zip(onsite_energies, onsite_x, onsite_y, onsite_z, onsite_id)
            ]

        # --- Make everything ready for the hopping-energies
        # given in the model.system
        sys_hoppings = self.system.hoppings.tocoo()

        # given in the self.lattice
        lat_hoppings = self.lattice.hoppings
        lat_hoppings = [
            (lat_hoppings[name].family_id, lat_hoppings[name].energy, name)
            for name in lat_hoppings.keys()
        ]
        lat_hoppings.sort(key=lambda x: x[0])

        # new name for the hopping terms
        hopping_used = []
        for hopl in lat_hoppings:
            if hopl[0] in sys_hoppings.data:
                hopping_used.append(hopl)

        hopping_terms = [
            "{0}-f-{1}-t-{2}-in-{3}".format(
                next(hopl for hopl in hopping_used if hopl[0] == hops)[2],
                onsite_terms[hop_r],
                onsite_terms[hop_c],
                hop_i)
            for hops, hop_r, hop_c, hop_i
            in zip(sys_hoppings.data, sys_hoppings.row, sys_hoppings.col, range(len(sys_hoppings.data)))
        ]

        # order the hopping-energies and the positions
        hopping_x1 = [
            np.array([
                positions.x[hops] for hops in sys_hoppings.row[sys_hoppings.data == hopl[0]]
            ])[:, np.newaxis, np.newaxis]
            for hopl in hopping_used
        ]
        hopping_y1 = [
            np.array([
                positions.y[hops] for hops in sys_hoppings.row[sys_hoppings.data == hopl[0]]
            ])[:, np.newaxis, np.newaxis]
            for hopl in hopping_used
        ]
        hopping_z1 = [
            np.array([
                positions.z[hops] for hops in sys_hoppings.row[sys_hoppings.data == hopl[0]]
            ])[:, np.newaxis, np.newaxis]
            for hopl in hopping_used
        ]
        hopping_x2 = [
            np.array([
                positions.x[hops] for hops in sys_hoppings.col[sys_hoppings.data == hopl[0]]
            ])[:, np.newaxis, np.newaxis]
            for hopl in hopping_used
        ]
        hopping_y2 = [
            np.array([
                positions.y[hops] for hops in sys_hoppings.col[sys_hoppings.data == hopl[0]]
            ])[:, np.newaxis, np.newaxis]
            for hopl in hopping_used
        ]
        hopping_z2 = [
            np.array([
                positions.z[hops] for hops in sys_hoppings.col[sys_hoppings.data == hopl[0]]
            ])[:, np.newaxis, np.newaxis]
            for hopl in hopping_used
        ]

        hopping_energies = [
            np.array(hopl[1])[np.newaxis, :, :] * np.ones(np.sum(sys_hoppings.data == hopl[0]))[:, np.newaxis, np.newaxis]
            for hopl in hopping_used
        ]
        hopping_id = [hopl[2] for hopl in hopping_used]

        hopping_names = [
            [hopping_terms[hopl_i] for hopl_i in
             np.arange(len(hopping_terms))[sys_hoppings.data == hopl[0]]]
            for hopl in hopping_used
        ]
        hopping_names_from = [
            [onsite_terms[hopl_i] for hopl_i in sys_hoppings.row[sys_hoppings.data == hopl[0]]]
            for hopl in hopping_used
        ]
        hopping_names_to = [
            [onsite_terms[hopl_i] for hopl_i in sys_hoppings.col[sys_hoppings.data == hopl[0]]]
            for hopl in hopping_used
        ]
        hopping_bound_used = []
        for bound in boundaries:
            hopping_bound_tmp = []
            for hopl in lat_hoppings:
                if hopl[0] in bound.hoppings.tocoo().data:
                    hopping_bound_tmp.append(hopl)
            hopping_bound_used.append(hopping_bound_tmp)

        hopping_bound_terms = [
            [
                "{0}-f{1}-t{2}-b{3}-{4}".format(
                    next(hopl for hopl in hopping_bound_used[bound_i] if hopl[0] == hops)[2],
                    onsite_terms[hop_r],
                    onsite_terms[hop_c],
                    bound_i,
                    hop_i)
                for hops, hop_r, hop_c, hop_i
                in zip(bound.hoppings.tocoo().data,
                       bound.hoppings.tocoo().row,
                       bound.hoppings.tocoo().col,
                       range(len(bound.hoppings.tocoo().data)))
            ] for bound_i, bound in enumerate(boundaries)
        ]

        hopping_bound_x1 = [
            [
                np.array([
                    positions.x[hops] + 1 * bound.shift[0]
                    for hops in bound.hoppings.tocoo().row[bound.hoppings.tocoo().data == hopl[0]]
                ])[:, np.newaxis, np.newaxis]
                for hopl in hopping_bound_used[bound_i]
            ]
            for bound_i, bound in enumerate(boundaries)
        ]
        hopping_bound_y1 = [
            [
                np.array([
                    positions.y[hops] + 1 * bound.shift[1]
                    for hops in bound.hoppings.tocoo().row[bound.hoppings.tocoo().data == hopl[0]]
                ])[:, np.newaxis, np.newaxis]
                for hopl in hopping_bound_used[bound_i]
            ]
            for bound_i, bound in enumerate(boundaries)
        ]
        hopping_bound_z1 = [
            [
                np.array([
                    positions.z[hops] + 1 * bound.shift[2]
                    for hops in bound.hoppings.tocoo().row[bound.hoppings.tocoo().data == hopl[0]]
                ])[:, np.newaxis, np.newaxis]
                for hopl in hopping_bound_used[bound_i]
            ]
            for bound_i, bound in enumerate(boundaries)
        ]
        hopping_bound_x2 = [
            [
                np.array([
                    positions.x[hops]
                    for hops in bound.hoppings.tocoo().col[bound.hoppings.tocoo().data == hopl[0]]
                ])[:, np.newaxis, np.newaxis]
                for hopl in hopping_bound_used[bound_i]
            ]
            for bound_i, bound in enumerate(boundaries)
        ]
        hopping_bound_y2 = [
            [
                np.array([
                    positions.y[hops]
                    for hops in bound.hoppings.tocoo().col[bound.hoppings.tocoo().data == hopl[0]]
                ])[:, np.newaxis, np.newaxis]
                for hopl in hopping_bound_used[bound_i]
            ]
            for bound_i, bound in enumerate(boundaries)
        ]
        hopping_bound_z2 = [
            [
                np.array([
                    positions.z[hops]
                    for hops in bound.hoppings.tocoo().col[bound.hoppings.tocoo().data == hopl[0]]
                ])[:, np.newaxis, np.newaxis]
                for hopl in hopping_bound_used[bound_i]
            ]
            for bound_i, bound in enumerate(boundaries)
        ]

        hopping_bound_energies = [
            [
                np.array(hopl[1])[np.newaxis, :, :] * np.ones(np.sum(bound.hoppings.tocoo().data == hopl[0]))[:, np.newaxis, np.newaxis]
                for hopl in hopping_bound_used[bound_i]
            ] for bound_i, bound in enumerate(boundaries)
        ]
        hopping_bound_id = [
            [
                hopl[2] for hopl in hopping_bound_used[bound_i]
            ] for bound_i, bound in enumerate(boundaries)
        ]
        hopping_bound_names = [
            [
                [hopping_bound_terms[bound_i][hopl_i] for hopl_i in np.arange(len(hopping_bound_terms[bound_i]))[bound.hoppings.tocoo().data == hopl[0]]]
                for hopl in hopping_bound_used[bound_i]
            ] for bound_i, bound in enumerate(boundaries)
        ]
        hopping_bound_names_from = [
            [
                [onsite_terms[hopl_i] for hopl_i in bound.hoppings.tocoo().row[bound.hoppings.tocoo().data == hopl[0]]]
                for hopl in hopping_bound_used[bound_i]
            ] for bound_i, bound in enumerate(boundaries)
        ]
        hopping_bound_names_to = [
            [
                [onsite_terms[hopl_i] for hopl_i in bound.hoppings.tocoo().col[bound.hoppings.tocoo().data == hopl[0]]]
                for hopl in hopping_bound_used[bound_i]
            ] for bound_i, bound in enumerate(boundaries)
        ]

        # apply the modifier
        if hopping_modifier:
            hopping_energies = [
                hopping_modifier(energy, x2, y2, z2, x1, y1, z1, hop_id)
                for energy, x1, y1, z1, x2, y2, z2, hop_id in zip(
                    hopping_energies, hopping_x1, hopping_y1, hopping_z1, hopping_x2, hopping_y2, hopping_z2, hopping_id)
            ]
            hopping_bound_energies = [
                [
                    hopping_modifier(energy, x1, y1, z1, x2, y2, z2, hop_id)
                    for energy, x1, y1, z1, x2, y2, z2, hop_id in zip(b_e, b_x1, b_y1, b_z1, b_x2, b_y2, b_z2, b_h)
                ] for b_e, b_x1, b_y1, b_z1, b_x2, b_y2, b_z2, b_h
                in zip(hopping_bound_energies, hopping_bound_x1, hopping_bound_y1, hopping_bound_z1,
                       hopping_bound_x2, hopping_bound_y2, hopping_bound_z2, hopping_bound_id)
            ]

        # --- make the pb.Lattice object ---
        # add the unit-cell vectors
        lat = Lattice(*lat_vectors)
        # add the onsite-energies
        for name_s, x_s, y_s, z_s, energy_s in zip(onsite_names, onsite_x, onsite_y, onsite_z, onsite_energies):
            for name, x, y, z, energy in zip(name_s, x_s, y_s, z_s, energy_s):
                lat.add_one_sublattice(name, [x.flatten()[0], y.flatten()[0], z.flatten()[0]], (energy + energy.T) / 2)

        # add hopping names
        for energy, h_name in zip(hopping_energies, hopping_names):
            lat.register_hopping_energies(dict(zip(h_name, np.swapaxes(energy, 1, 2))))

        # add hopping names over the boundaries
        for b_e, h_n in zip(hopping_bound_energies, hopping_bound_names):
            for energy, h_name in zip(b_e, h_n):
                if h_name:
                    lat.register_hopping_energies(dict(zip(h_name, np.swapaxes(energy, 1, 2))))

        # add the hoppings within the unit-cell
        for name_f, name_t, name_h in zip(hopping_names_from, hopping_names_to, hopping_names):
            for n_f, n_t, name in zip(name_f, name_t, name_h):
                if name:
                    lat.add_one_hopping([0, 0], n_t, n_f, name)

        # add the hoppings over the boundary
        for b_h_f, b_h_y, b_h_n, b_i in zip(
                hopping_bound_names_from, hopping_bound_names_to, hopping_bound_names, hopping_shifts):
            for name_f, name_t, name_h in zip(b_h_f, b_h_y, b_h_n):
                if name_h:
                    for n_f, n_t, name in zip(name_f, name_t, name_h):
                        lat.add_one_hopping(b_i, n_t, n_f, name)

        return lat