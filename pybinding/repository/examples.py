"""Example components: lattices, shapes and modifiers

Components which aren't very useful for simulations,
but great for examples and testing.
"""
import pybinding as pb


def chain_lattice(a: float = 1, t: float = -1, v: float = 0) -> pb.Lattice:
    """1D lattice

    Parameters
    ----------
    a : float
        Unit cell length.
    t : float
        Hopping energy.
    v : float
        Onsite energy.
    """
    lat = pb.Lattice(a)
    lat.add_one_sublattice('A', [0, 0], onsite_energy=v)
    lat.add_one_hopping(1, 'A', 'A', t)
    return lat


def square_lattice(d: float = 0.2, t: float = -1) -> pb.Lattice:
    """A simple square lattice

    Parameters
    ----------
    d : float
        Length of the unit cell.
    t : float
        Hopping energy.

    Returns
    -------
    pb.Lattice
    """
    lat = pb.Lattice(a1=[d, 0], a2=[0, d])
    lat.add_one_sublattice('A', [0, 0])
    lat.add_hoppings(
        ([0,  1], 'A', 'A', t),
        ([1,  0], 'A', 'A', t),
        ([1,  1], 'A', 'A', t),
        ([1, -1], 'A', 'A', t),
    )
    return lat
