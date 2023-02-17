"""A few useful physical constants

Note that energy is expressed in units of eV.
"""
from math import pi
import numpy as np

c = 299792458  #: [m/s] speed of light
e = 1.602 * 10**-19  #: [C] electron charge
epsilon0 = 8.854 * 10**-12  #: [F/m] vacuum permittivity
hbar = 6.58211899 * 10**-16  #: [eV*s] reduced Plank constant
phi0 = 2 * pi * hbar  #: [V*s] magnetic quantum
bhor = 1.8897259886  #: [1 Angstrom = 1.8897259886 Bhor]


class Pauli(np.ndarray):
    """Pauli matrices, 'e', 'x', 'y', and 'z' to get the identity matrix and the three Pauli matrices respectively."""
    def __new__(cls):
        pauli_all = [
            [
                [1, 0],
                [0, 1]
            ],
            [
                [0, 1],
                [1, 0]
            ],
            [
                [0, -1j],
                [1j, 0]
            ],
            [
                [1, 0],
                [0, -1]
            ]
        ]
        return np.asarray(pauli_all).view(cls)

    @property
    def e(self) -> np.ndarray:
        """2D array of the identity matrix"""
        return np.array(self[0])

    @property
    def x(self) -> np.ndarray:
        """2D array of the x-Pauli matrix"""
        return np.array(self[1])

    @property
    def y(self) -> np.ndarray:
        """2D array of the y-Pauli matrix"""
        return np.array(self[2])

    @property
    def z(self) -> np.ndarray:
        """2D array of the z-Pauli matrix"""
        return np.array(self[3])

    def __repr__(self) -> str:
        return "0: [[1, 0], [0, 1]], x: [[0, 1], [1, 0]], y: [[0, -1j], [1j, 0]], z: [[1, 0], [0, -1]]"


pauli = Pauli()  #: Pauli matrices -- use the ``.x``, ``.y`` and ``.z`` attributes
