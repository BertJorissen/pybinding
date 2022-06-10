"""Tight-binding models for group 6 transition metal dichalcogenides (TMD), 3 band."""
import re
import math
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from .tmd_abstract_lattice import AbstractLattice

_default_3band_params_NN = {  # from https://doi.org/10.1103/PhysRevB.88.085433
    # ->           a,  eps1,  eps2,     t0,    t1,    t2,   t11,   t12,    t22, lambda
    "MoS2":  [0.3190, 1.046, 2.104, -0.184, 0.401, 0.507, 0.218, 0.338,  0.057, 0.073],
    "WS2":   [0.3191, 1.130, 2.275, -0.206, 0.567, 0.536, 0.286, 0.384, -0.061, 0.211],
    "MoSe2": [0.3326, 0.919, 2.065, -0.188, 0.317, 0.456, 0.211, 0.290,  0.130, 0.091],
    "WSe2":  [0.3325, 0.943, 2.179, -0.207, 0.457, 0.486, 0.263, 0.329,  0.034, 0.228],
    "MoTe2": [0.3557, 0.605, 1.972, -0.169, 0.228, 0.390, 0.207, 0.239,  0.252, 0.107],
    "WTe2":  [0.3560, 0.606, 2.102, -0.175, 0.342, 0.410, 0.233, 0.270,  0.190, 0.237],
}

_default_3band_params_TNN = {  # from https://doi.org/10.1103/PhysRevB.88.085433
    # ->           a,  eps1,  eps2,     t0,     t1,    t2,   t11,    t12,    t22,    r0,     r1,\
    #             r2,   r11,   r12,     u0,     u1,    u2,   u11,    u12,    u22, lambda
    "MoS2":  [0.3190, 0.683, 1.707, -0.146, -0.114, 0.506, 0.085,  0.162,  0.073, 0.060, -0.236,
              0.067 , 0.016, 0.087, -0.038,  0.046, 0.001, 0.266, -0.176, -0.150, 0.073],
    "WS2":   [0.3191, 0.717, 1.916, -0.152, -0.097, 0.590, 0.047,  0.178,  0.016, 0.069, -0.261,
              0.107 ,-0.003, 0.109, -0.054,  0.045, 0.002, 0.325, -0.206, -0.163, 0.211],
    "MoSe2": [0.3326, 0.684, 1.546, -0.146, -0.130, 0.432, 0.144,  0.117,  0.075, 0.039, -0.209,
              0.069 , 0.052, 0.060, -0.042,  0.036, 0.008, 0.272, -0.172, -0.150, 0.091],
    "WSe2":  [0.3325, 0.728, 1.655, -0.146, -0.124, 0.507, 0.117,  0.127,  0.015, 0.036, -0.234,
              0.107 , 0.044, 0.075, -0.061,  0.032, 0.007, 0.329, -0.202, -0.164, 0.228],
    "MoTe2": [0.3557, 0.588, 1.303, -0.226, -0.234, 0.036, 0.400,  0.098,  0.017, 0.003, -0.025,
              -0.169, 0.082, 0.051,  0.057,  0.103, 0.187,-0.045, -0.141,  0.087, 0.107],
    "WTe2":  [0.3560, 0.697, 1.380, -0.109, -0.164, 0.368, 0.204,  0.093,  0.038,-0.015, -0.209,
              0.107 , 0.115, 0.009, -0.066,  0.011,-0.013, 0.312, -0.177, -0.132, 0.237],
}

_bert_3band_params_TNN = {  # from https://doi.org/10.1103/PhysRevB.88.085433
    # ->           a,   eps1,   eps2,     t0,     t1,     t2,    t11,    t12,    t22,     r0,     r1,\
    #             r2,    r11,    r12,     u0,     u1,     u2,    u11,    u12,    u22, lambda
    "MoS2":  [ 0.319,  0.817,  1.891, -0.147, -0.101,  0.538,  0.081,  0.177,  0.073,  0.068, -0.255,
               0.061,  0.007,  0.096, -0.05 ,  0.041,  0.001,  0.276, -0.181, -0.153,  0.073],
}


class Group6Tmd3Band(AbstractLattice):
    r"""Monolayer of a group 6 TMD using the nearest-neighbor 3-band model

    Parameters
    ----------
    name : str
        Name of the TMD to model. The available options are: MoS2, WS2, MoSe2,
        WSe2, MoTe2, WTe2. The relevant tight-binding parameters for these 
        materials are given by https://doi.org/10.1103/PhysRevB.88.085433.
    override_params : Optional[dict]
        Replace or add new material parameters. The dictionary entries must 
        be in the format `"name": [a, eps1, eps2, t0, t1, t2, t11, t12, t22]`.
    tnn: Boolean
        Take the Third Nearest Neighbor into account
    soc: Boolean
        Also calculate the Spin Orbit Coupling
    transform: (2x2) numpy array
        transformation matrix for the basis vectors
    """

    def __init__(self, **kwargs):

        lattice_orbital_dict = {"l": {"Mo": [0, 2, -2]},
                                "orbs": {"Mo": ["dz2", "dx2y2", "dxy"]},
                                "group": {"Mo": [0, 1, 1]}}
        self.__name = "MoS2"
        self.__tnn = True
        super().__init__(orbital=lattice_orbital_dict, n_v=0, n_b=3)
        self.single_orbital = False
        self.use_theta = False
        self._berry_phase_factor = -1
        self.params = None
        [setattr(self, var, kwargs[var]) for var in [*kwargs]]

    @property
    def tnn(self):
        return self.__tnn

    @tnn.setter
    def tnn(self, tnn):
        self.__tnn = tnn
        self._generate_matrices()

    @property
    def params(self):
        return (_default_3band_params_TNN if self.tnn else _default_3band_params_NN) \
            if self.__params_default else self.__params

    @params.setter
    def params(self, params):
        self.__params_default = params is None
        if self.__params_default:
            params = _default_3band_params_TNN if self.tnn else _default_3band_params_NN
        self.__name_list = [*params]
        self.__params = params
        self.name = self.__name_list[0] if self.name not in self.__name_list else self.name

    def _generate_matrices(self):
        if self.tnn:
            a, eps1, eps2, t0, t1, t2, t11, t12, t22, r0, r1, r2, r11, r12, u0, u1, u2, u11, u12, u22, lamb_m\
                = self.params[self.name]
        else:
            a, eps1, eps2, t0, t1, t2, t11, t12, t22, lamb_m = self.params[self.name]
            r0, r1, r2, r11, r12, u0, u1, u2, u11, u12, u22 = [None] * 11

        def h_0(e_0, e_1):
            return np.diag([e_0, e_1, e_1])

        def h_2_mat(t_0, t_1, t_2, t_11, t_12, t_22):
            return np.array([[ t_0,  t_2, -t_1],
                             [ t_2, t_22, t_12],
                             [ t_1,-t_12, t_11]])

        def h_5_mat(r_0, r_1, r_2, r_11, r_12):
            return np.array([[r_0, 2 / np.sqrt(3) * r_1, 0],
                             [2 / np.sqrt(3) * r_2, r_11 - 1 / np.sqrt(3) * r_12, 0],
                             [0.0, 0.0, r_11 + np.sqrt(3) * r_12]])
        h_0_m = h_0(eps1, eps2)
        h_2_m = h_2_mat(t0, t1, t2, t11, t12, t22)
        h_5_m = h_5_mat(r0, r1, r2, r11, r12) if self.tnn else None
        h_6_m = h_2_mat(u0, u1, u2, u11, u12, u22) if self.tnn else None
        keys = ["h_0_m", "h_2_m", "h_5_m", "h_6_m", "a", "lamb_m"]
        values = [h_0_m, h_2_m, h_5_m, h_6_m, a, lamb_m]
        self.lattice_params(**dict([(key, value) for key, value in zip(keys, values)]))


if __name__ == "__main__":
    group6_class = Group6Tmd3Band()
    group6_class.test1()
    res = group6_class.test2()
