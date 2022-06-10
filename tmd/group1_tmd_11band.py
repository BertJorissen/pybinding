"""Tight-binding models for group 1 transition metal dichalcogenides (TMD), 6 band."""
import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from .tmd_abstract_lattice import AbstractLattice

_default_11band_params = {  # from https://journals.aps.org/prb/abstract/10.1103/PhysRevB.88.075409
    # "name"        a,   theta, delta_0, delta_1, delta_2, delta_p, delta_z,
    #           v_pds,   v_pdp,   v_dds,   v_ddp,   v_ddd, v_pps_0, v_ppp_0, v_pps_2, v_ppp_2,  lamb_m,  lamb_c
    "MoS2":  [  0.316,   0.712,  -1.016,   0.000,  -2.529,  -0.780,  -7.740,
               -2.619,  -1.396,  -0.933,  -0.478,  -0.442,    None,    None,   0.696,   0.278,   0.075,   0.052]
}

_group2_11bands_params = {  # from https://doi.org/10.1088/2053-1583/1/3/034003
    # "name"        a,   theta, delta_0, delta_1, delta_2, delta_p, delta_z,
    #           v_pds,   v_pdp,   v_dds,   v_ddp,   v_ddd, v_pps_0, v_ppp_0, v_pps_2, v_ppp_2,  lamb_m,  lamb_c
    "MoS2":  [  0.316,   0.716,  -1.512,   0.419,  -3.025,  -1.276,  -8.236,
               -2.619,  -1.396,  -0.933,  -0.478,  -0.442,    None,    None,   0.696,   0.278,   0.075,   0.052],
    "WS2":   [ 0.3153,   0.716,  -1.550,   0.851,  -3.090,  -1.176,  -7.836,
               -2.619,  -1.396,  -0.983,  -0.478,  -0.442,    None,    None,   0.696,   0.278,   0.215,   0.057]
}

_group1_11bands_CB_VB_params = {  # from https://doi.org/10.1088/0953-8984/27/36/365501
    # "name"        a,   theta, delta_0, delta_1, delta_2, delta_p, delta_z,
    #           v_pds,   v_pdp,   v_dds,   v_ddp,   v_ddd, v_pps_0, v_ppp_0, v_pps_2, v_ppp_2,  lamb_m,  lamb_c
    "MoS2":  [  0.316,   0.710,   0.201,  -1.563,  -0.352, -54.839, -39.275,
               -9.880,   4.196,  -1.153,   0.612,   0.086,    None,    None,  12.734,  -2.175,   0.086,   0.052]
}

_group1_11bands_VB_params = {  # from https://doi.org/10.1088/0953-8984/27/36/365501
    # "name"        a,   theta, delta_0, delta_1, delta_2, delta_p, delta_z,
    #           v_pds,   v_pdp,   v_dds,   v_ddp,   v_ddd, v_pps_0, v_ppp_0, v_pps_2, v_ppp_2,  lamb_m,  lamb_c
    "MoS2":  [  0.316,   0.710,   0.191,  -1.599,   0.081, -48.934, -37.981,
               -8.963,   4.115,  -1.154,   0.964,   0.117,    None,    None,  10.707,  -4.084,   0.086,   0.052]
}

_group1_11bands_minimal = {  # from https://doi.org/10.1088/0953-8984/27/36/365501
    # "name"        a,   theta, delta_0, delta_1, delta_2, delta_p, delta_z,
    #           v_pds,   v_pdp,   v_dds,   v_ddp,   v_ddd, v_pps_0, v_ppp_0, v_pps_2, v_ppp_2,  lamb_m,  lamb_c
    "MoS2":  [  0.316,   0.710, -11.683,-208.435, -75.952, -23.761, -35.968,
              -56.738,   1.318,  -2.652,   1.750,   1.482,    None,    None,   0.000,   0.000,   0.086,   0.052]
}

_group1_11bands_params_venkateswarlu = { # from 10.1103/PhysRevB.102.081103
    # "name"        a,   theta, delta_0, delta_1, delta_2, delta_p, delta_z,
    #           v_pds,   v_pdp,   v_dds,   v_ddp,   v_ddd, v_pps_0, v_ppp_0, v_pps_2, v_ppp_2,  lamb_m,  lamb_c
    "MoS2":  [  0.318,   0.704,   .1356,  -.4204,   .0149,  -38.71,  -29.45,
               -7.193,   3.267,  -.9035,   .7027,   .0897,   8.079,  -2.678,   7.336,  -2.432,   0.086,   0.052]
}

_group4_11bands_params = {  # from https://www.mdpi.com/2076-3417/6/10/284
    # "name"        a,   theta, delta_0, delta_1, delta_2, delta_p, delta_z,
    #           v_pds,   v_pdp,   v_dds,   v_ddp,   v_ddd, v_pps_0, v_ppp_0, v_pps_2, v_ppp_2,  lamb_m,  lamb_c
    "MoS2":  [  0.316,   0.716,  -1.094,  -0.050,  -1.511,  -3.559,  -6.886,
                3.689,  -1.241,  -0.895,   0.252,   0.228,    None,    None,   1.225,  -0.467,   0.086,   0.052],
    "MoSe2": [  3.288,   0.720,  -1.144,  -0.250,  -1.488,  -4.931,  -7.503,
                3.728,  -1.222,  -0.823,   0.215,   0.192,    None,    None,   1.256,  -0.205,   0.089,   0.256],
    "WS2":   [  3.153,   0.712,  -1.155,  -0.650,  -2.279,  -3.864,  -7.327,
                5.911,  -1.220,  -1.328,   0.121,   0.422,    None,    None,   1.178,  -0.273,   0.271,   0.057],
    "WSe2":  [  3.260,   0.722,  -0.935,  -1.250,  -2.321,  -5.629,  -6.759,
                5.803,  -1.081,  -1.129,   0.094,   0.317,    None,    None,   1.530,  -0.123,   0.251,   0.439]
}

_group1_11bands_pearce = { # from https://doi.org/10.1103/PhysRevB.94.155416
    # "name"        a,   theta, delta_0, delta_1, delta_2, delta_p, delta_z,
    #           v_pds,   v_pdp,   v_dds,   v_ddp,   v_ddd, v_pps_0, v_ppp_0, v_pps_2, v_ppp_2,  lamb_m,  lamb_c
    "MoS2":  [  0.319,   0.690,    2.12,   -0.46,   -1.41,   -5.38,   -3.69,
                -2.83,    0.67,   -0.24,   -0.62,    0.45,    None,    None,   -0.42,   -1.32,   0.075,   0.052],
}

_group1_11bands_bieniek =  { # from https://arxiv.org/pdf/1705.02917.pdf
    # "name"        a,   theta, delta_0, delta_1, delta_2, delta_p, delta_z,
    #           v_pds,   v_pdp,   v_dds,   v_ddp,   v_ddd, v_pps_0, v_ppp_0, v_pps_2, v_ppp_2,  lamb_m,  lamb_c
    "MoS2":  [  0.3193,   0.710,   -0.03,  -0.03,   -0.03,   -3.36,   -4.78,
                -3.39,    1.10,   -1.10,    0.76,    0.27,    None,    None,    1.19,   -0.83,   0.075,   0.052],
}

_group1_11_bands_Abdi = { # from https://iopscience.iop.org/article/10.1149/2162-8777/abb28b/pdf
    # from 10.1016/j.rinp.2022.105253
    # "name"        a,   theta, delta_0, delta_1, delta_2, delta_p, delta_z,
    #           v_pds,   v_pdp,   v_dds,   v_ddp,   v_ddd, v_pps_0, v_ppp_0, v_pps_2, v_ppp_2,  lamb_m,  lamb_c
    #"MoS2":  [  0.316,   0.710,   -1.512,   0.00,  -3.025,  -1.276,  -8.236,
    #           -2.619,  -1.396,   -0.933,   0.478,  -0.442,   None,    None,   0.696,    0.278,   0.075,   0.052],
    "MoS2":   [ 0.3166,   0.710,   -1.512,   0.00,  -3.025,  -1.276,  -8.236,
                -2.619,  -1.396,   -0.933,   0.478,  -0.442,   None,    None,   0.696,    0.278,   0.085,   0.052],
}


_default_6band_params = {  # from https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.195402 ; lamM, lamX, a and
                           # theta are from https://iopscience.iop.org/article/10.1088/2053-1583/1/3/034003/meta
    # "name"        a,   theta, delta_0, delta_1, delta_2, delta_p, delta_z,
    #           v_pds,   v_pdp,   v_dds,   v_ddp,   v_ddd, v_pps_0, v_ppp_0, v_pps_2, v_ppp_2,  lamb_m,  lamb_c
    "MoS2":  [  0.316,   0.716,  -1.094,   0.000,  -1.512,  -3.560,  -6.886,
                3.689,  -1.241,  -0.895,   0.252,   0.228,    None,    None,   1.225,  -0.467,   0.075,   0.052],
}


_bert_11band_params = {  # from https://journals.aps.org/prb/abstract/10.1103/PhysRevB.88.075409
    # "name"        a,   theta, delta_0,delta_1,delta_2,delta_p,delta_z,
    #           v_pds,   v_pdp,   v_dds,  v_ddp,  v_ddd,v_pps_0,v_ppp_0,v_pps_2,v_ppp_2, lamb_m,  lamb_c
    "MoS2":  [  0.316,   0.712,  -0.883,  0.567,  1.396, -1.802, -1.247,
                2.298,  -0.68 , -0.642,   0.229,  0.348,  1.525, -0.714,  0.707,   -0.4,   0.075,   0.052]
}

class Group1Tmd11Band(AbstractLattice):
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
        a, theta, delta_0, delta_1, delta_2, delta_p, delta_z, v_pds, v_pdp, v_dds, v_ddp, v_ddd, v_pps_0, v_ppp_0,\
        v_pps_2, v_ppp_2, lamb_m, lamb_c = self.params[self.name]

        if v_pps_0 is None:
            v_pps_0 = v_pps_2
        if v_ppp_0 is None:
            v_ppp_0 = v_ppp_2

        def h_1_mat(t1, t2):
            return np.array(
                [[0.0, 0.0, -np.cos(theta) * t1, np.sin(theta) * t1, 0.0],
                 [np.sqrt(3) * np.cos(theta) * np.sin(theta)**2 * t1
                  - np.cos(theta) * (np.sin(theta)**2 - 1 / 2 * np.cos(theta)**2) * t2,
                  np.cos(theta) * np.sin(theta)**2 * t1 + np.sqrt(3) / 2 * np.cos(theta)**3 * t2, 0.0, 0.0,
                  np.sin(theta) * (1 - 2 * np.cos(theta) ** 2) * t1
                  + np.sqrt(3) * np.cos(theta) ** 2 * np.sin(theta) * t2],
                 [np.sqrt(3) * np.sin(theta) * np.cos(theta) ** 2 * t1
                  + np.sin(theta) * (np.sin(theta) ** 2 - 1 / 2 * np.cos(theta) ** 2) * t2,
                  -np.sin(theta) * np.cos(theta)**2 * (-t1 + np.sqrt(3)/2 * t2), 0.0, 0.0,
                  - np.cos(theta) * (1 - 2 * np.sin(theta) ** 2) * t1
                  - np.sqrt(3) * np.sin(theta) ** 2 * np.cos(theta) * t2]]
            ) if theta is not None and self.use_theta else 1 / (7 * np.sqrt(7)) * np.array(
                [[0.0, 0.0, -14 * t1, 7 * np.sqrt(3) * t1, 0.0],
                 [6 * np.sqrt(3) * t1 - 2 * t2, 6 * t1 + 4 * np.sqrt(3) * t2, 0.0, 0.0, -np.sqrt(3) * t1 + 12 * t2],
                 [12 * t1 + np.sqrt(3) * t2, 4 * np.sqrt(3) * t1 - 6 * t2, 0.0, 0.0, -2 * t1 - 6 * np.sqrt(3) * t2]])


        h_0_m = np.diag((delta_0, delta_2, delta_2, delta_1, delta_1))
        h_1_m = h_1_mat(v_pdp, v_pds)
        #if self.even_odd:
        if True:
            h_0_c_o = np.diag((delta_p - v_ppp_0, delta_p - v_ppp_0, delta_z + v_pps_0))
            h_0_c_e = np.diag((delta_p + v_ppp_0, delta_p + v_ppp_0, delta_z - v_pps_0))
            h_0_c = self.block_diag(h_0_c_e, h_0_c_o)
            h_1_m_e = np.sqrt(2) * h_1_m[:, :3]
            h_1_m_o = np.sqrt(2) * h_1_m[:, 3:]
            h_1_m = self.block_diag(h_1_m_e, h_1_m_o)
        else:
            h_0_c_t = np.diag((delta_p, delta_p, delta_z))
            h_0_c_b = np.diag((delta_p, delta_p, delta_z))
            h_0_c_tb = np.diag((v_ppp_0, v_ppp_0, v_pps_0))
            h_0_c = np.concatenate((np.concatenate((h_0_c_t, h_0_c_tb), axis=1),
                                    np.concatenate((h_0_c_tb.T.conj(), h_0_c_b), axis=1)), axis=0)
            h_1_m_3 = h_1_m[:, :3]
            h_1_m_2 = h_1_m[:, 3:]
            h_1_m = np.concatenate((np.concatenate((h_1_m_3, h_1_m_2), axis=1),
                                    np.concatenate(
                                        (np.diag([1, 1, -1]).dot(h_1_m_3), np.diag([-1, -1, 1]).dot(h_1_m_2)), axis=1),
                                    ), axis=0)
        h_2_c = np.diag((v_pps_2, v_ppp_2, v_ppp_2, v_pps_2, v_ppp_2, v_ppp_2))
        ur_mat = self.block_diag(self.orbital.rot_mat(2 * np.pi / 3), np.array([[1]]))
        h_2_m_3 = np.dot(ur_mat, np.dot(np.diag((v_dds, v_ddd, v_ddp)), ur_mat.T))
        h_2_m_2 = np.diag((v_ddp, v_ddd))
        h_2_m = self.block_diag(h_2_m_3, h_2_m_2)
        keys = ["h_0_m", "h_0_c", "h_1_m", "h_2_m", "h_2_c", "a", "lamb_m", "lamb_c"]
        values = [h_0_m, h_0_c, h_1_m, h_2_m, h_2_c, a, lamb_m, lamb_c]
        self.lattice_params(**dict([(key, value) for key, value in zip(keys, values)]))

    def test1(self, soc=False, lat4=False, even_odd=False, theta=None, path=1):
        # path: 1 - G-K-M-G -- 2 - G-M-K-G -- 3 - M-G-K-M
        print("Doing every calculation with high precision,\n this can take a minute.")
        grid = plt.GridSpec(3, 4, hspace=0.4)
        plt.figure(figsize=(10, 8))
        self.even_odd = even_odd
        self.use_theta = theta
        self.soc = soc
        self.lat4 = lat4
        for square, params, name, title in zip(grid, [_default_11band_params,
                                               _group1_11bands_params_venkateswarlu,
                                               _group1_11bands_VB_params,
                                               _group1_11bands_CB_VB_params,
                                               _group1_11bands_minimal,
                                               _group2_11bands_params,
                                               _group2_11bands_params,
                                               _group4_11bands_params,
                                               _group4_11bands_params,
                                               _group4_11bands_params,
                                               _group4_11bands_params,
                                               _default_6band_params],
                                        ["MoS2", "MoS2", "MoS2", "MoS2", "MoS2", "MoS2", "WS2", "MoS2", "MoSe2", "WS2",
                                         "WSe2", "MoS2"],
                                        ["Cap. MoS2", "Ven. MoS2", "RiVB MoS2", "RiCV MoS2", "RiMN MoS2", "Rol. MoS2",
                                         "Rol. WS2", "Sil. MoS2", "Sil. MoSe2", "Sil. WS2", "Sil. WSe2", "Ros. MoS2"]):
            plt.subplot(square, title=title)
            for i in [False, True]:
                self.even_odd = i
                self.params = params
                self.name = name
                model = pb.Model(self.lattice(), pb.translational_symmetry())
                k_points = model.lattice.brillouin_zone()
                gamma = [0, 0]
                k = k_points[1]
                m = (k_points[2] + k_points[1]) / 2
                if path == 1:
                    bands = pb.solver.lapack(model).calc_bands(gamma, k, m, gamma, step=0.05)
                    bands.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"],
                               lw=1.5, color="C1" if i else "C0", ls=":" if i else "-")
                elif path == 2:
                    bands = pb.solver.lapack(model).calc_bands(gamma, m, k, gamma, step=0.05)
                    bands.plot(point_labels=[r"$\Gamma$", "M", "K", r"$\Gamma$"],
                               lw=1.5, color="C1" if i else "C0", ls=":" if i else "-")
                elif path == 3:
                    bands = pb.solver.lapack(model).calc_bands(m, gamma, k, m, step=0.05)
                    bands.plot(point_labels=["M", r"$\Gamma$", "K", "M"],
                               lw=1.5, color="C1" if i else "C0", ls=":" if i else "-")
            plt.gca().set_ylim([-5, 5])
        plt.show()

    def test2(self, soc=False):
        bands = []
        plt.figure()
        for lat4 in [False, True]:
            self.lat4 = lat4
            self.soc = soc
            model = pb.Model(self.lattice(), pb.translational_symmetry())
            solver = pb.solver.lapack(model)
            if not lat4:
                k_points = model.lattice.brillouin_zone()
                gamma = [0, 0]
                k = k_points[0]
                m = (k_points[0] + k_points[1]) / 2
            bands.append(solver.calc_bands(gamma, k, m, gamma, step=0.05))
        bands[0].plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"], lw=1.5, color="C0")
        bands[1].plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"], lw=1.5, color="C1", ls=":")
        error = np.array(
            [[np.abs(bands[1].energy[i] - bands[0].energy[i, j]).min()
              for j in range(bands[0].energy.shape[1])] for i in
             range(bands[0].energy.shape[0])])
            # class.test2()[0] for maximal error
        return error.max(), error, bands

if __name__ == "__main__":
    group1_class = Group1Tmd11Band()
    group1_class.test1(path=2, soc=True, even_odd=True)