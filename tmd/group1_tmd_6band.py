"""Tight-binding models for group 1 transition metal dichalcogenides (TMD), 6 band."""
import re
import math
import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from .tmd_abstract_lattice import AbstractLattice
_default_6band_params = {  # from https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.195402 ; lamM, lamX, a and
                           # theta are from https://iopscience.iop.org/article/10.1088/2053-1583/1/3/034003/meta
    # "name"        a,   theta, delta_0, delta_1, delta_2, delta_p, delta_z,
    #           v_pds,   v_pdp,   v_dds,   v_ddp,   v_ddd, v_pps_0, v_ppp_0, v_pps_2, v_ppp_2,  lamb_m,  lamb_c
    "MoS2":  [  0.316,   0.716,  -1.094,   0.000,  -1.512,  -3.560,  -6.886,
                3.689,  -1.241,  -0.895,   0.252,   0.228,    None,    None,   1.225,  -0.467,   0.075,   0.052],
}

_bert_6band_params = {  # from https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.195402 ; lamM, lamX, a and
                           # theta are from https://iopscience.iop.org/article/10.1088/2053-1583/1/3/034003/meta
    # "name"        a,   theta, delta_0,delta_1,delta_2,delta_p,delta_z,
    #           v_pds,   v_pdp,   v_dds,  v_ddp,  v_ddd,v_pps_0,v_ppp_0,v_pps_2,v_ppp_2, lamb_m,lamb_c
    "MoS2":  [  0.316,   0.716,  -0.665,  0.164,  1.583, -3.499, -1.439,
                2.229,  -0.707,  -0.727,  0.475,  0.317,  1.054, -0.064,  0.799,  -0.31,  0.075, 0.052],
}


class Group1Tmd6Band(AbstractLattice):
    r"""Monolayer of a group 1 TMD using the second nearest-neighbor 6-band model

    Parameters
    ----------
    name : str
        Name of the TMD to model. The available options are: MoS2
        WThe relevant tight-binding parameters for these
        materials are given by https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.195402
    override_params : Optional[dict]
        Replace or add new material parameters.m The dictionary entries must be
        in the format `"name": [ eps0,   eps2,    epsp,     epsz,   V_pdd,   V_pdp,   V_ddd,   V_ddp,  V_ddde,  V_ppd,
                                V_ppp,      a,   lamXX,    lamXM,   lamMM,    lamM,    lamX]
    sz : float
        Z-component of the spin degree of freedom
    """
    def __init__(self, **kwargs):

        lattice_orbital_dict = {"l": {"Mo": [0, 2, -2], "S": [1, -1, 0]},
                                "orbs": {"Mo": ["dz2", "dx2y2", "dxy"], "S": ["pxe", "pye", "pze"]},
                                "group": {"Mo": [0, 1, 1], "S": [0, 0, 1]}}
        super().__init__(orbital=lattice_orbital_dict, n_v=3, n_b=6)
        self.soc_eo_flip = False
        self.single_orbital = False
        self.use_theta = False
        self.params = _default_6band_params
        self._berry_phase_factor = -1
        [setattr(self, var, kwargs[var]) for var in [*kwargs]]

    def _generate_matrices(self):
        a, theta, delta_0, delta_1, delta_2, delta_p, delta_z, v_pds, v_pdp, v_dds, v_ddp, v_ddd, v_pps_0, v_ppp_0,\
        v_pps_2, v_ppp_2, lamb_m, lamb_c = self.params[self.name]

        if v_pps_0 is None:
            v_pps_0 = v_pps_2
        if v_ppp_0 is None:
            v_ppp_0 = v_ppp_2

        def h_0_mat(t1, t2, t3):
            return np.diag([t1, t2, t3])

        def h_1_mat(t1, t2):
            return np.sqrt(2) * np.array(
                [[0.0, 0.0, -np.cos(theta) * t1],
                 [np.sqrt(3) * np.cos(theta) * np.sin(theta) ** 2 * t1
                  - np.cos(theta) * (np.sin(theta) ** 2 - 1 / 2 * np.cos(theta) ** 2) * t2,
                  np.cos(theta) * np.sin(theta) ** 2 * t1 + np.sqrt(3) / 2 * np.cos(theta) ** 3 * t2, 0.0],
                 [np.sqrt(3) * np.sin(theta) * np.cos(theta) ** 2 * t1
                  + np.sin(theta) * (np.sin(theta) ** 2 - 1 / 2 * np.cos(theta) ** 2) * t2,
                  -np.sin(theta) * np.cos(theta) ** 2 * (-t1 + np.sqrt(3) / 2 * t2), 0.0]]
            ) if theta is not None and self.use_theta else np.sqrt(2) * np.array(
                [[0.0, 0.0, -np.sqrt(4/7) * t1],
                 [np.sqrt(3) * np.sqrt(4/7) * np.sqrt(3/7) ** 2 * t1
                  - np.sqrt(4/7) * (np.sqrt(3/7) ** 2 - 1 / 2 * np.sqrt(4/7) ** 2) * t2,
                  np.sqrt(4/7) * np.sqrt(3/7) ** 2 * t1 + np.sqrt(3) / 2 * np.sqrt(4/7) ** 3 * t2, 0.0],
                 [np.sqrt(3) * np.sqrt(3/7) * np.sqrt(4/7) ** 2 * t1
                  + np.sqrt(3/7) * (np.sqrt(3/7) ** 2 - 1 / 2 * np.sqrt(4/7) ** 2) * t2,
                  -np.sqrt(3/7) * np.sqrt(4/7) ** 2 * (-t1 + np.sqrt(3) / 2 * t2), 0.0]])

        h_0_m = h_0_mat(delta_0, delta_2, delta_2)
        h_0_c = h_0_mat(delta_p + v_ppp_0, delta_p + v_ppp_0, delta_z - v_pps_0)
        h_1_m = h_1_mat(v_pdp, v_pds)
        h_2_c = h_0_mat(v_pps_2, v_ppp_2, v_ppp_2)
        ur_mat = block_diag(self.orbital.rot_mat(2 * np.pi / 3), np.array([[1]]))
        h_2_m = np.dot(ur_mat, np.dot(np.diag((v_dds, v_ddd, v_ddp)), ur_mat.T))
        keys = ["h_0_m", "h_0_c", "h_1_m", "h_2_m", "h_2_c", "a", "lamb_m", "lamb_c"]
        values = [h_0_m, h_0_c, h_1_m, h_2_m, h_2_c, a, lamb_m, lamb_c]
        self.lattice_params(**dict([(key, value) for key, value in zip(keys, values)]))


if __name__ == "__main__":
    import tybinding as ty
    import matplotlib
    #matplotlib.use('Qt5Agg')
    group1_class = ty.Group1Tmd6Band()
    group1_class.params = ty.repository.tmd.group1_tmd_6band._default_6band_params
    group1_class.test1(soc=True)
    #res = group1_class.test2()
