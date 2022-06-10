"""Tight-binding models for group 5 transition metal dichalcogenides (TMD), 11 band."""
import re
import math
import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from .tmd_abstract_lattice import AbstractLattice

_default_11band_params = {  # from https://link.aps.org/doi/10.1103/PhysRevB.98.075202
    # "name"          a,     theta,    lamb_m,    lamb_c,
    #         delta_e_0, delta_o_1, delta_e_2, delta_e_p, delta_e_z, delta_o_p, delta_o_z,
    #         v_pps_e_0, v_ppp_e_0,
    #         v_pds_e_1, v_pdp_e_1,
    #         v_dds_e_2, v_ddp_e_2, v_ddd_e_2, v_pps_e_2, v_ppp_e_2,
    #         v_dds_e_5, v_ddp_e_5, v_ddd_e_5, v_pps_e_5, v_ppp_e_5,
    #         v_pps_o_0, v_ppp_o_0,
    #         v_pds_o_1, v_pdp_o_1,
    #  UNUSED[v_dds_o_2],v_ddp_o_2, v_ddd_o_2, v_pps_o_2, v_ppp_o_2,
    #  UNUSED[v_dds_o_5],v_ddp_o_5, v_ddd_o_5, v_pps_o_5, v_ppp_o_5
    "MoS2":  [   0.3166,     0.710,    0.0806,    0.0536,
                -0.4939,    0.5624,   -0.2473,   -4.5716,   -8.3498,   -1.5251,   -0.6737,
                   None,      None,
                 4.2398,   -1.2413,
                -0.6717,    0.5706,    0.2729,   -0.0914,   -0.4619,
                 0.0314,    0.0961,   -0.0305,    0.3723,    0.0014,
                   None,      None,
                 2.2251,   -0.7614,
                -0.8950,    0.0150,    0.0497,    0.8131,   -0.2763,
                 0.0100,    0.0051,    0.0184,   -0.0395,    0.0092],
    "MoSe2": [   0.3288,     0.710,    0.0806,    0.0820,
                -0.1276,    0.3046,   -0.2724,   -6.1588,   -7.3399,   -1.3298,   -0.9459,
                   None,      None,
                 3.4524,   -1.4295,
                -0.6674,    0.5573,    0.0970,    1.2630,   -0.4857,
                 0.0776,    0.0573,  -0.04778,    0.2372,    0.0249,
                   None,      None,
                 2.0197,   -0.6811,
                -0.8950,   0.01637,    0.0965,    0.9449,   -0.3039,
                 0.0100,    0.0140,    0.0354,   -0.0293,   -0.0094],
    "MoTe2": [   0.3519,     0.710,    0.0806,    0.1020,
                -0.6630,    0.0491,   -0.2852,   -0.5923,    -3.7035,   -1.3905,   -0.0094,
                   None,      None,
                 2.2362,   -0.6279,
                -0.4795,   -0.0934,    0.1656,    0.8198,    -0.2483,
                -0.1493,   -0.0627,    0.0360,    0.1169,     0.2683,
                   None,      None,
                 1.8294,   -0.5048,
                -0.8950,    0.3267,    0.3033,    0.8459,    -0.4143,
                 0.0100,   -0.0617,    0.1002,    0.0114,    -0.0092],
    "WS2":   [  0.31532,     0.710,    0.2754,    0.0536,
                -0.3609,    0.8877,   -0.7364,   -5.0982,    -9.4019,   -1.8175,   -1.0191,
                   None,      None,
                 5.2769,   -1.2119,
                -0.8942,    0.7347,    0.3417,   -0.3943,    -0.4069,
                 0.0508,    0.1278,   -0.0091,    0.1415,     0.0261,
                   None,      None,
                 2.4044,   -0.8115,
                -0.8950,   -0.0142,    0.0036,    0.8415,    -0.2661,
                 0.0100,   -0.0135,   -0.0191,   -0.0169,     0.0262],
    "WSe2":  [   0.3282,     0.710,    0.2754,    0.0820,
                -0.5558,    0.6233,   -1.9340,   -2.9498,    -6.5922,   -1.5016,   -1.4824,
                   None,      None,
                 5.1750,   -0.9139,
                -0.8697,    0.6206,    0.3743,    0.1311,    -0.2475,
                 0.0443,    0.0912,   -0.0447,    0.1197,     0.1075,
                   None,      None,
                 2.1733,   -0.7688,
                -0.8950,   -0.0469,    0.0923,    0.9703,    -0.2920,
                 0.0100,    0.0096,    0.0140,   -0.0451,     0.0113],
}

_bert_11band_params_dias = {  # from https://link.aps.org/doi/10.1103/PhysRevB.98.075202
    # "name"          a,     theta,    lamb_m,    lamb_c,
    #         delta_e_0, delta_o_1, delta_e_2, delta_e_p, delta_e_z, delta_o_p, delta_o_z,
    #         v_pps_e_0, v_ppp_e_0,
    #         v_pds_e_1, v_pdp_e_1,
    #         v_dds_e_2, v_ddp_e_2, v_ddd_e_2, v_pps_e_2, v_ppp_e_2,
    #         v_dds_e_5, v_ddp_e_5, v_ddd_e_5, v_pps_e_5, v_ppp_e_5,
    #         v_pps_o_0, v_ppp_o_0,
    #         v_pds_o_1, v_pdp_o_1,
    #  UNUSED[v_dds_o_2],v_ddp_o_2, v_ddd_o_2, v_pps_o_2, v_ppp_o_2,
    #  UNUSED[v_dds_o_5],v_ddp_o_5, v_ddd_o_5, v_pps_o_5, v_ppp_o_5
    "MoS2":  [   0.317,  0.   ,  0.081,  0.054,
                -2.185,  1.612,  1.031, -0.713, -1.869,  0.604,  0.005,
                 1.461, -0.257,
                -3.381, -0.472,
                 0.586, -0.081, -0.117, -0.41 ,  0.114,
                -0.129,  0.033, -0.215,  0.11 ,  0.032,
                -1.339,  2.042,
                -2.007,  1.322,
                   252,  0.288, -0.071,  0.322, -0.1  ,
                   261, -0.115,  0.186, -0.121,  0.013],
}

class Group5Tmd11Band(AbstractLattice):
    r"""Monolayer of a group 1 TMD using the second nearest-neighbor 11-band model

    Parameters
    ----------
    name : str
        Name of the TMD to model. The available options are: MoS2
        WThe relevant tight-binding parameters for these
        materials are given by https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.195402
    override_params : Optional[dict]
        Replace or add new material parameters. The dictionary entries must be
        in the format `"name": [ eps0,   eps2,    epsp,     epsz,   V_pdd,   V_pdp,   V_ddd,   V_ddp,  V_ddde,  V_ppd,
                                V_ppp,      a,   lamXX,    lamXM,   lamMM,    lamM,    lamX]
    sz : float
        Z-component of the spin degree of freedom
    """
    def __init__(self, **kwargs):

        lattice_orbital_dict = {"l": {"Mo": [0, 2, -2, 1, -1], "S": [1, -1, 0, 1, -1, 0]},
                                "orbs": {"Mo": ["dz2", "dx2y2", "dxy", "dxz", "dyz"],
                                         "S": ["pxe", "pye", "pze", "pxo", "pyo", "pzo"]},
                                "group": {"Mo": [0, 1, 1, 2, 2], "S": [0, 0, 1, 2, 2, 3]}}
        super().__init__(orbital=lattice_orbital_dict, n_v=6, n_b=11)
        self.single_orbital = False
        self.use_theta = False
        self.params = _default_11band_params
        self._berry_phase_factor = -1
        [setattr(self, var, kwargs[var]) for var in [*kwargs]]

    def _generate_matrices(self):
        a,             theta,    lamb_m,    lamb_c,\
        delta_e_0, delta_o_1, delta_e_2, delta_e_p, delta_e_z, delta_o_p, delta_o_z,\
        v_pps_e_0, v_ppp_e_0,\
        v_pds_e_1, v_pdp_e_1,\
        v_dds_e_2, v_ddp_e_2, v_ddd_e_2, v_pps_e_2, v_ppp_e_2,\
        v_dds_e_5, v_ddp_e_5, v_ddd_e_5, v_pps_e_5, v_ppp_e_5,\
        v_pps_o_0, v_ppp_o_0,\
        v_pds_o_1, v_pdp_o_1,\
        v_dds_o_2, v_ddp_o_2, v_ddd_o_2, v_pps_o_2, v_ppp_o_2,\
        v_dds_o_5, v_ddp_o_5, v_ddd_o_5, v_pps_o_5, v_ppp_o_5\
        = self.params[self.name]

        if v_pps_e_0 is None:
            v_pps_e_0 = v_pps_e_2
        if v_ppp_e_0 is None:
            v_ppp_e_0 = v_ppp_e_2
        if v_pps_o_0 is None:
            v_pps_o_0 = v_pps_o_2
        if v_ppp_o_0 is None:
            v_ppp_o_0 = v_ppp_o_2

        def h_1_mat(t1e, t2e, t1o, t2o):
            return np.array(
                [[0.0, 0.0, -np.cos(theta) * t1e, np.sin(theta) * t1e, 0.0],
                 [np.sqrt(3) * np.cos(theta) * np.sin(theta) ** 2 * t1e
                  - np.cos(theta) * (np.sin(theta) ** 2 - 1 / 2 * np.cos(theta) ** 2) * t2e,
                  np.cos(theta) * np.sin(theta) ** 2 * t1e + np.sqrt(3) / 2 * np.cos(theta) ** 3 * t2e, 0.0, 0.0,
                  np.sin(theta) * (1 - 2 * np.cos(theta) ** 2) * t1o
                  + np.sqrt(3) * np.cos(theta) ** 2 * np.sin(theta) * t2o],
                 [np.sqrt(3) * np.sin(theta) * np.cos(theta) ** 2 * t1e
                  + np.sin(theta) * (np.sin(theta) ** 2 - 1 / 2 * np.cos(theta) ** 2) * t2e,
                  -np.sin(theta) * np.cos(theta) ** 2 * (-t1e + np.sqrt(3) / 2 * t2e), 0.0, 0.0,
                  - np.cos(theta) * (1 - 2 * np.sin(theta) ** 2) * t1o
                  - np.sqrt(3) * np.sin(theta) ** 2 * np.cos(theta) * t2o]]
            ) if theta is not None and self.use_theta else 1 / (7 * np.sqrt(7)) * np.array(
                [[0.0, 0.0, -14 * t1e, 7 * np.sqrt(3) * t1o, 0.0],
                 [6 * np.sqrt(3) * t1e - 2 * t2e, 6 * t1e + 4 * np.sqrt(3) * t2e, 0.0, 0.0, -np.sqrt(3) * t1o + 12 * t2o],
                 [12 * t1e + np.sqrt(3) * t2e, 4 * np.sqrt(3) * t1e - 6 * t2e, 0.0, 0.0, -2 * t1o - 6 * np.sqrt(3) * t2o]])

        h_0_m = np.diag((delta_e_0, delta_e_2, delta_e_2, delta_o_1, delta_o_1))
        h_1_m = h_1_mat(v_pdp_e_1, v_pds_e_1, v_pdp_o_1, v_pds_o_1)
        h_0_c_e = np.diag((delta_e_p + v_ppp_e_0, delta_e_p + v_ppp_e_0, delta_e_z - v_pps_e_0))
        h_0_c_o = np.diag((delta_o_p - v_ppp_o_0, delta_o_p - v_ppp_o_0, delta_o_z + v_pps_o_0))
        h_0_c = block_diag(h_0_c_e, h_0_c_o)
        h_1_m_e = np.sqrt(2) * h_1_m[:, :3]
        h_1_m_o = np.sqrt(2) * h_1_m[:, 3:]
        h_1_m = block_diag(h_1_m_e, h_1_m_o)
        h_2_c = np.diag((v_pps_e_2, v_ppp_e_2, v_ppp_e_2, v_pps_o_2, v_ppp_o_2, v_ppp_o_2))
        h_5_c = np.diag((v_ppp_e_5, v_pps_e_5, v_ppp_e_5, v_ppp_o_5, v_pps_o_5, v_ppp_o_5))
        ur_mat = block_diag(self.orbital.rot_mat(2 * np.pi / 3), np.array([[1]]))
        h_2_m_3 = np.dot(ur_mat, np.dot(np.diag((v_dds_e_2, v_ddd_e_2, v_ddp_e_2)), ur_mat.T))
        h_2_m_2 = np.diag((v_ddp_o_2, v_ddd_o_2))
        h_2_m = block_diag(h_2_m_3, h_2_m_2)
        h_5_m_3 = np.dot(ur_mat.T, np.dot(np.diag((v_dds_e_5, v_ddd_e_5, v_ddp_e_5)), ur_mat))
        h_5_m_2 = np.diag((v_ddp_o_5, v_ddd_o_5))
        h_5_m = block_diag(h_5_m_3, h_5_m_2)
        keys = ["h_0_m", "h_0_c", "h_1_m", "h_2_m", "h_2_c", "h_5_m", "h_5_c", "a", "lamb_m", "lamb_c"]
        values = [h_0_m, h_0_c, h_1_m, h_2_m, h_2_c, h_5_m, h_5_c, a, lamb_m, lamb_c]
        self.lattice_params(**dict([(key, value) for key, value in zip(keys, values)]))

if __name__ == "__main__":
    group1_class = Group5Tmd11Band()
    group1_class.test1(theta=False, soc=True)