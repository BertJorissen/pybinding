"""Tight-binding models for group 1 transition metal dichalcogenides (TMD), 5 band."""
import re
import math
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from .tmd_abstract_lattice import AbstractLattice

_default_5band_params = {  # from https://doi.org/10.1103/PhysRevB.88.085433
    # ->           a,  eps1,  eps2,     t0,     t1,    t2,   t11,    t12,    t22,    r0,     r1,\
    #             r2,   r11,   r12,     u0,     u1,    u2,   u11,    u12,    u22, lambda
    # and https://link.aps.org/doi/10.1103/PhysRevB.91.075310
    #             o1,     t,    td,    txy,      s,    sd,     u,     ud,    uxy
    "MoS2":  [0.3190, 0.683, 1.707, -0.146, -0.114, 0.506, 0.085,  0.162,  0.073, 0.060, -0.236,
              0.067 , 0.016, 0.087, -0.038,  0.046, 0.001, 0.266, -0.176, -0.150, 0.073,
              3.558 ,-0.189,-0.117,  0.024, -0.041, 0.003, 0.165, -0.122, -0.140],
}

_bert_5band_params = {  # from https://doi.org/10.1103/PhysRevB.88.085433
    # ->           a,   eps1,   eps2,     t0,     t1,     t2,    t11,    t12,    t22,    r0,     r1,\
    #             r2,    r11,    r12,     u0,     u1,     u2,    u11,    u12,    u22, lambda
    # and https://link.aps.org/doi/10.1103/PhysRevB.91.075310
    #             o1,      t,     td,    txy,      s,     sd,      u,     ud,    uxy
    "MoS2":  [ 0.319,  0.817,  1.891, -0.147, -0.101,  0.538,  0.081,  0.177,  0.073,  0.068, -0.255,
               0.061,  0.007,  0.096, -0.05 ,  0.041,  0.001,  0.276, -0.181, -0.153,  0.073,
               3.78 , -0.146, -0.163,  0.028, -0.04 ,  0.003,  0.158, -0.112, -0.155],
}

__all__ = ["Group1Tmd5Band"]

class Group1Tmd5Band(AbstractLattice):
    r"""Monolayer of a group 1 TMD using the third nearest-neighbor 5-band model

    Parameters
    ----------
    name : str
        Name of the TMD to model. The available options are: MoS2, WS2, MoSe2,
        WSe2, MoTe2, WTe2. The relevant tight-binding parameters for these 
        materials are given by https://doi.org/10.1103/PhysRevB.88.085433 and
        https://link.aps.org/doi/10.1103/PhysRevB.91.075310.
    override_params : Optional[dict]
        Replace or add new material parameters. The dictionary entries must 
        be in the format `"name": [a, eps1, eps2, t0, t1, t2, t11, t12, t22]`.
    soc: Boolean
        Also calculate the Spin Orbit Coupling
    transform: (2x2) numpy array
        transformation matrix for the basis vectors
    """
    def __init__(self, **kwargs):

        lattice_orbital_dict = {"l": {"Mo": [0, 2, -2, 1, -1]},
                                "orbs": {"Mo": ["dz2", "dx2y2", "dxy", "dxz", "dyz"]}}
        super().__init__(orbital=lattice_orbital_dict, n_v=0, n_b=5)
        self.single_orbital = False
        self._berry_phase_factor = 1
        self.params = _default_5band_params
        [setattr(self, var, kwargs[var]) for var in [*kwargs]]

    def _generate_matrices(self):
        a, eps1, eps2, t0, t1, t2, t11, t12, t22, r0, r1, r2, r11, r12, u0, u1, u2, u11, u12, u22, lamb_m, \
        o1, t, td, txy, s, sd, u, ud, uxy = self.params[self.name]

        def h_0(e_0, e_1, o_1):
            h_o = np.diag([o_1, o_1])
            h_e = np.diag([e_0, e_1, e_1])
            return self.block_diag(h_e, h_o)

        def h_2_mat(t_0, t_1, t_2, t_11, t_12, t_22, t_3, t_4, t_5):
            t_o = np.array([[ t_3, -t_4],
                            [ t_4,  t_5]])
            t_e = np.array([[ t_0,  t_2, -t_1],
                            [ t_2, t_22, t_12],
                            [ t_1,-t_12, t_11]])
            return self.block_diag(t_e, t_o)

        def h_5_mat(r_0, r_1, r_2, r_11, r_12, r_3, r_4):
            r_o = np.array([[3 / 2 * r_4 - r_3 / 2,                     0],
                            [                    0, 3 / 2 * r_3 - r_4 / 2]])
            r_e = np.array([[r_0, 2 / np.sqrt(3) * r_1, 0],
                            [2 / np.sqrt(3) * r_2, r_11 - 1 / np.sqrt(3) * r_12, 0],
                            [0.0, 0.0, r_11 + np.sqrt(3) * r_12]])
            return self.block_diag(r_e, r_o)
        h_0_m = h_0(eps1, eps2, o1)
        h_2_m = h_2_mat(t0, t1, t2, t11, t12, t22, t, txy, td)
        h_5_m = h_5_mat(r0, r1, r2, r11, r12, s, sd)
        h_6_m = h_2_mat(u0, u1, u2, u11, u12, u22, u, uxy, ud)
        keys = ["h_0_m", "h_2_m", "h_5_m", "h_6_m", "a", "lamb_m"]
        values = [h_0_m, h_2_m, h_5_m, h_6_m, a, lamb_m]
        self.lattice_params(**dict([(key, value) for key, value in zip(keys, values)]))


if __name__ == "__main__":
    group1_class = Group1Tmd5Band()
    group1_class.test1()
    res = group1_class.test2()
