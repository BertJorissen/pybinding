"""Tight-binding models for group 4 transition metal dichalcogenides (TMD), 11 band."""
import re
import math
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

_default_11band_params = {  # from https://link.aps.org/doi/10.1103/PhysRevB.92.205108
    #"name"         a,    eps1,    eps3,    eps4,    eps6,    eps7,    eps9,   eps10,  t1_1_1,  t1_2_2,
    #          t1_3_3,  t1_4_4,  t1_5_5,  t1_6_6,  t1_7_7,  t1_8_8,  t1_9_9,t1_10_10,t1_11_11,  t1_3_5,
    #          t1_6_8,  1_9_11,  t1_1_2,  t1_3_4,  t1_4_5,  t1_6_7,  t1_7_8, t1_9_10,t1_10_11,  t5_4_1,
    #          t5_3_2,  t5_5_2,  t5_9_6, t5_11_6, t5_10_7,  t5_9_8, t5_11_8,  t6_9_6, t6_11_6,  t6_9_8,
    #         t6_11_8,       c,     dXX,     dXM,   lambM,   lambX
    "MoS2":  [ 0.318 ,  1.0688, -0.7755, -1.2902, -0.1380,  0.0874, -2.8949, -1.9065, -0.2069,  0.0323,
              -0.1739,  0.8651, -0.1872, -0.2979,  0.2747, -0.5581, -0.1916,  0.9122,  0.0059, -0.0679,
               0.4096,  0.0075, -0.2562, -0.0995, -0.0705, -0.1145, -0.2487,  0.1063, -0.0385, -0.7883,
              -1.3790,  2.1584, -0.8836, -0.9402,  1.4114, -0.9535,  0.6517, -0.0686, -0.1498, -0.2205,
              -0.2451, 12.29  ,  3.13  ,  2.41  ,  0.0836,  0.0556],
    "MoSe2": [ 0.332 ,  0.7819, -0.6567, -1.1726, -0.2297,  0.0149, -2.9015, -1.7806, -0.1460,  0.0177,
              -0.2112,  0.9638, -0.1724, -0.2636,  0.2505, -0.4734, -0.2166,  0.9911, -0.0036, -0.0735,
               0.3520,  0.0047, -0.1912, -0.0755, -0.0680, -0.0960, -0.2012,  0.1216, -0.0394, -0.6946,
              -1.3258,  1.9415, -0.7720, -0.8738,  1.2677, -0.8578,  0.5545, -0.0691, -0.1553, -0.2227,
              -0.2154, 12.90  ,  3.34  ,  2.54  ,  0.0836,  0.2470],
    "WS2":   [ 0.318 ,  1.3754, -1.1278, -1.5534, -0.0393,  0.1984, -3.3706, -2.3461, -0.2011,  0.0263,
              -0.1749,  0.8726, -0.2187, -0.3716,  0.3537, -0.6892, -0.2112,  0.9673,  0.0143, -0.0818,
               0.4896, -0.0315, -0.3106, -0.1105, -0.0989, -0.1467, -0.3030,  0.1645, -0.1018, -0.8855,
              -1.4376,  2.3121, -1.0130, -0.9878,  1.5629, -0.9491,  0.6718, -0.0659, -0.1533, -0.2618,
              -0.2736, 12.32  ,  3.14  ,  2.42  ,  0.2874,  0.0556],
    "WSe2":  [ 0.332 ,  1.0349, -0.9573, -1.3937, -0.1667,  0.0984, -3.3642, -2.1820, -0.1395,  0.0129,
              -0.2171,  0.9763, -0.1985, -0.3330,  0.3190, -0.5837, -0.2399,  1.0470,  0.0029, -0.0912,
               0.4233, -0.0377, -0.2321, -0.0797, -0.0920, -0.1250, -0.2456,  0.1857, -0.1027, -0.7744,
              -1.4014,  2.0858, -0.8998, -0.9044,  1.4030, -0.8548,  0.5711, -0.0676, -0.1608, -0.2618,
              -0.2424, 12.96  ,  3.35  ,  2.55  ,  0.2874,  0.2470]
}

_default_11band_params_matlab = {  # from S. Fang code
    #"name"         a,    eps1,    eps3,    eps4,    eps6,    eps7,    eps9,   eps10,  t1_1_1,  t1_2_2,
    #          t1_3_3,  t1_4_4,  t1_5_5,  t1_6_6,  t1_7_7,  t1_8_8,  t1_9_9,t1_10_10,t1_11_11,  t1_3_5,
    #          t1_6_8,  1_9_11,  t1_1_2,  t1_3_4,  t1_4_5,  t1_6_7,  t1_7_8, t1_9_10,t1_10_11,  t5_4_1,
    #          t5_3_2,  t5_5_2,  t5_9_6, t5_11_6, t5_10_7,  t5_9_8, t5_11_8,  t6_9_6, t6_11_6,  t6_9_8,
    #         t6_11_8,       c,     dXX,     dXM,   lambM,   lambX
    "MoS2":  [ 0.318 ,  1.0688,-0.77546, -1.2902,  -0.138, 0.08738, -2.8949, -1.9065, -0.2069,0.032345,
             -0.17385, 0.86513,-0.18725,-0.29791,  0.2747,-0.55809,-0.19158, 0.91215,0.0058631,-0.067891,
              0.40961,0.007516,-0.25616,-0.099527,-0.070508,-0.1145,-0.24872,0.10626,-0.038544,-0.7883,
               -1.379,  2.1584,-0.88356,-0.94019,  1.4114,-0.95348, 0.65168,-0.068597,-0.14979,-0.22048,
               -0.24507, 12.29,  3.13  ,  2.41  ,  0.0836,  0.0556],
    "MoSe2": [ 0.332 , 0.78192,-0.65672, -1.1726,-0.22974,0.014947, -2.9015, -1.7806,-0.14602, 0.017738,
             -0.21122,  0.9638,-0.17242,-0.26363, 0.25053,-0.47341, -0.2166, 0.99107,-0.0036331,-0.07347,
              0.35202,0.0047333,-0.19118,-0.075456,-0.068039,-0.096031,-0.20118,0.12163,-0.039403,-0.69463,
              -1.3258,  1.9415,-0.77195,-0.87375,  1.2677,-0.85776, 0.55453,-0.069127,-0.15531, -0.2227,
             -0.21537, 12.90  ,  3.34  ,  2.54  ,  0.0836,  0.2470],
    "WS2":   [ 0.318  ,  1.3754,-1.1278, -1.5534,-0.039259,0.19844, -3.3706, -2.3461,  0.20109, .026255,
              -0.17491,0.87261,-0.21869,-0.37161, 0.35371,-0.68916, -0.2112, 0.96733, 0.014269,-0.081777,
               0.48965,-0.031484,-0.31058,-0.11054,-0.098891,-0.14665,-0.30304,0.16454,-0.10178,-0.88549,
               -1.4376, 2.3121,  -1.013,-0.98778,  1.5629,-0.94906, 0.67181,-0.065943,-0.15333,-0.26184,
              -0.27364, 12.32  ,  3.14  ,  2.42  ,  0.2874,  0.0556],
    "WSe2":  [ 0.332  ,  1.0349,-0.95728, -1.3937,-0.16674,0.098437,-3.3642,  -2.182,-0.13951, 0.012861,
              -0.21706, 0.97629,-0.19847,-0.33299, 0.31903,-0.58366,-0.23985,  1.047,0.0029342,-0.091183,
               0.42334,-0.037681,-0.2321,-0.079716,-0.09203,-0.12498,-0.24558,0.18574,-0.10266, -0.77438,
               -1.4014,  2.0858, -0.8998,-0.90437,   1.403, -0.8548, 0.57113,-0.067618,-0.16076,-0.26179,
               -0.24244, 12.96  ,  3.35  ,  2.55  ,  0.2874,  0.2470]
}


class Group4Tmd11Band:
    r"""Monolayer of a group 4 TMD using the second nearest-neighbor 11-band model

    Parameters
    ----------
    name : str
        Name of the TMD to model. The available options are: MoS2, WS2, MoSe2,
        WSe2. The relevant tight-binding parameters for these
        materials are given by https://link.aps.org/doi/10.1103/PhysRevB.92.205108
    override_params : Optional[dict]
        Replace or add new material parameters. The dictionary entries must
        be in the format `"name": [eps1,    eps3,    eps4,    eps6,    eps7,    eps9,   eps10,  t1_1_1,  t1_2_2,  t1_3_3,
                                   t1_4_4,  t1_5_5,  t1_6_6,  t1_7_7,  t1_8_8,  t1_9_9,t1_10_10,t1_11_11,  t1_3_5,  t1_6_8,
                                   t1_9_11,  t1_1_2,  t1_3_4,  t1_4_5,  t1_6_7,  t1_7_8, t1_9_10,t1_10_11,  t5_4_1,  t5_3_2,
                                   t5_5_2,  t5_9_6, t5_11_6, t5_10_7,  t5_9_8, t5_11_8,  t6_9_6, t6_11_6,  t6_9_8, t6_11_8,
                                   a,       c,     dXX,     dXM,    lambM, lambX]`.

    Examples
    --------
    .. plot::
        :context: reset
        :alt: Molybdenum disulfide: unit cell for the nearest-neighbor 3-band model

        import group4_tmd_11band

        group4_tmd_11band.monolayer_11band("MoS2").plot()

    .. plot::
        :context: close-figs
        :alt: Molybdenum disulfide: 11-band model band structure

        model = pb.Model(group4_tmd_11band.monolayer_11band("MoS2"), pb.translational_symmetry())
        solver = pb.solver.lapack(model)

        k_points = model.lattice.brillouin_zone()
        gamma = [0, 0]
        k = k_points[0]
        m = (k_points[0] + k_points[1]) / 2

        plt.figure(figsize=(6.7, 2.3))

        plt.subplot(121, title="MoS2 11-band model band structure")
        bands = solver.calc_bands(gamma, k, m, gamma)
        bands.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])

        plt.subplot(122, title="Band structure path in reciprocal space")
        model.lattice.plot_brillouin_zone(decorate=False)
        bands.plot_kpath(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])

    .. plot::
        :context: close-figs
        :alt: Band structure of various group 11 TMDs: MoS2, WS2, MoSe2, WSe2

        grid = plt.GridSpec(2, 2, hspace=0.4)
        plt.figure(figsize=(6.7, 8))

        for square, name in zip(grid, ["MoS2", "WS2", "MoSe2", "WSe2"]):
            model = pb.Model(group4_tmd_11band.monolayer_11band(name), pb.translational_symmetry())
            solver = pb.solver.lapack(model)

            k_points = model.lattice.brillouin_zone()
            gamma = [0, 0]
            k = k_points[0]
            m = (k_points[0] + k_points[1]) / 2

            plt.subplot(square, title=name)
            bands = solver.calc_bands(gamma, k, m, gamma)
            bands.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"], lw=1.5)
    """
    def __init__(self, **kwargs):
        self.single_orbital = False
        self.single_name_m = ["-dxz", "-dyz", "-dxy", "-dx2-y2", "-dz2"]
        self.single_name_x = ["-pxo", "-pyo", "-pzo", "-pxe", "-pye", "-pze"]
        # make a list of all the parameters we import during the creation of the various matrices for the hoppings from
        # the table from Fang's papers
        self._name = "MoS2"
        self._lat4 = False
        # defaults; also needed to import the matrices from Fang
        self.params = _default_11band_params
        self._rt3 = np.sqrt(3)
        self.soc = False
        self.soc_eo_flip = True
        self.soc_polarized = False
        self.sz = 1.
        self._ur = np.array([[-.5, self._rt3 / 2, 0], [-self._rt3 / 2, -.5, 0], [0, 0, 1]])
        self._berry_phase_factor = -1
        # see if there are parameters to overwrite
        # make the lattice vectors
        # add other default parameters
        # see if there are more parameters to overwrite; also the matrices from Fang could be overwritten
        [setattr(self, var, kwargs[var]) for var in [*kwargs]]

    @property
    def params(self):
        return self.__params

    @params.setter
    def params(self, params):
        self.__name_list = [*params]
        self.__params = params
        self.name = [*params][0] if self.name not in [*params] else self.name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name not in self.__name_list:
            print("The name %s is not in the params-list" % name)
        else:
            self._name = name
            self._generate_matrices()

    @property
    def lat4(self):
        return self._lat4

    @lat4.setter
    def lat4(self, lat4):
        self._lat4 = lat4

    @property
    def a1(self):
        return np.array([1, 0]) * self.a

    @property
    def a2(self):
        return (np.array([0, self._rt3]) if self.lat4 else np.array([-1 / 2, self._rt3 / 2])) * self.a

    @property
    def soc_doubled_ham(self):
        return not self.soc_polarized if self.soc else False

    @property
    def soc_eo_flip_used(self):
        return self.soc and self.soc_eo_flip and not self.soc_polarized

    @property
    def n_valence_band(self):
        return 13 if self.soc_doubled_ham else 6

    @property
    def n_bands(self):
        return 22 if self.soc_doubled_ham else 11

    def _generate_matrices(self):
        # generate the matrices from Fang
        # import data
        a, eps1, eps3, eps4, eps6, eps7, eps9, eps10, t1_1_1, t1_2_2, \
        t1_3_3, t1_4_4, t1_5_5, t1_6_6, t1_7_7, t1_8_8, t1_9_9, t1_10_10, t1_11_11, t1_3_5, \
        t1_6_8, t1_9_11, t1_1_2, t1_3_4, t1_4_5, t1_6_7, t1_7_8, t1_9_10, t1_10_11, t5_4_1, \
        t5_3_2, t5_5_2, t5_9_6, t5_11_6, t5_10_7, t5_9_8, t5_11_8, t6_9_6, t6_11_6, t6_9_8, \
        t6_11_8, c, dCC, dCM, lamb_m, lamb_c = self.params[self.name]

        # make functions for various matrices
        def matrix_0(e_0, e_1):
            return np.diag([e_1, e_1, e_0])

        def matrix_n(t_0_n, t_1_n, t_2_n, t_3_n, t_4_n):
            return np.array([[t_0_n, 0, 0], [0, t_1_n, t_2_n], [0, t_3_n, t_4_n]])

        def matrix_2(t_0_2, t_1_2, t_2_2, t_3_2, t_4_2, t_5_2):
            return np.array([[t_0_2, t_3_2, t_4_2], [-t_3_2, t_1_2, t_5_2], [-t_4_2, t_5_2, t_2_2]])

        # make the matrices
        h_0_aa = matrix_0(0.000000, eps1)[:2, :2]
        h_0_bb = matrix_0(eps3, eps4)
        h_0_cc = matrix_0(eps6, eps7)
        h_0_dd = matrix_0(eps9, eps10)
        h_1_ba = matrix_n(t5_4_1, t5_5_2, 0.000000, t5_3_2, 0.000000)[:, :2]
        h_1_dc = matrix_n(t5_10_7, t5_11_8, t5_11_6, t5_9_8, t5_9_6)
        h_2_aa = matrix_2(t1_1_1, t1_2_2, 0.000000, t1_1_2, 0.000000, 0.000000)[:2, :2]
        h_2_bb = matrix_2(t1_4_4, t1_5_5, t1_3_3, t1_4_5, -t1_3_4, t1_3_5)
        h_2_cc = matrix_2(t1_7_7, t1_8_8, t1_6_6, t1_7_8, -t1_6_7, t1_6_8)
        h_2_dd = matrix_2(t1_10_10, t1_11_11, t1_9_9, t1_10_11, -t1_9_10, t1_9_11)
        h_3_dc = matrix_n(0.000000, t6_11_8, t6_11_6, t6_9_8, t6_9_6)
        keys = ["h_0_aa", "h_0_bb", "h_0_cc", "h_0_dd", "h_1_ba", "h_1_dc", "h_2_aa", "h_2_bb", "h_2_cc", "h_2_dd",
                "h_3_dc", "a", "c", "d_cc", "d_cm", "lamb_m", "lamb_c"]
        values = [h_0_aa, h_0_bb, h_0_cc, h_0_dd, h_1_ba, h_1_dc, h_2_aa, h_2_bb, h_2_cc, h_2_dd, h_3_dc,
                  a, c, dCC, dCM, lamb_m, lamb_c]
        [setattr(self, key, value) for key, value in zip(keys, values)]

    def ham(self, h, ur_l, ur_r):
        return h, ur_l.T.dot(h.dot(ur_r)), ur_l.dot(h.dot(ur_r.T))

    def lattice(self):
        lat = pb.Lattice(a1=self.a1, a2=self.a2)

        metal_name, chalcogenide_name = re.findall("[A-Z][a-z]*", self.name)

        def h_0_m_sz(sz):
            s_part_aa = sz * self.lamb_m * 1j * np.array([[0, -1/2], [1/2, 0]])
            s_part_cc = sz * self.lamb_m * 1j * np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
            return block_diag(self.h_0_aa + s_part_aa, self.h_0_cc + s_part_cc)

        def h_0_c_sz(sz):
            s_part_bb = sz * self.lamb_c * 1j * np.array([[0, -1/2, 0], [1/2, 0, 0], [0, 0, 0]])
            return block_diag(self.h_0_bb + s_part_bb, self.h_0_dd + s_part_bb)

        if self.soc:
            if self.soc_polarized:
                h_0_m = h_0_m_sz(self.sz)
                h_0_c = h_0_c_sz(self.sz)
            else:
                soc_part_m = np.zeros((5, 5)) * 1j
                soc_part_c = np.zeros((6, 6)) * 1j
                if self.soc_eo_flip:
                    soc_part_m[:2, 2:] = self.sz * self.lamb_m * np.array(
                        [[1j / 2, -1 / 2, self._rt3 / 2],
                         [-1 / 2, -1j / 2, -1j / 2 * self._rt3]])
                    soc_part_m[2:, :2] = -soc_part_m[:2, 2:].T

                    soc_part_c[:3, 3:] = self.sz * self.lamb_c * np.array(
                        [[0, 0, 1/2],
                         [0, 0, -1j/2],
                         [-1/2, 1j / 2, 0]])
                    soc_part_c[3:, :3] = -soc_part_c[:3, 3:].T
                h_0_m = np.concatenate((np.concatenate((h_0_m_sz(self.sz), soc_part_m), axis=1),
                                        np.concatenate((soc_part_m.conj().T, h_0_m_sz(-self.sz)), axis=1)), axis=0)
                h_0_c = np.concatenate((np.concatenate((h_0_c_sz(self.sz), soc_part_c), axis=1),
                                        np.concatenate((soc_part_c.conj().T, h_0_c_sz(-self.sz)), axis=1)), axis=0)
        else:
            h_0_m = h_0_m_sz(0.)
            h_0_c = h_0_c_sz(0.)

        if self.single_orbital:
            m = [metal_name + mi for mi in self.single_name_m]
            if self.soc_doubled_ham:
                m = [mi + "u" for mi in m] + [mi + "d" for mi in m]
            n_m = len(m)
            for i in range(n_m):
                lat.add_one_sublattice(m[i], [0, 0], np.real(h_0_m[i, i]))
            for i in range(n_m):
                for j in np.arange(i + 1, n_m):
                    lat.add_one_hopping([0, 0], m[i], m[j], h_0_m[i, j])  # could be the transpose
        else:
            lat.add_one_sublattice(metal_name, [0, 0], h_0_m.T)

        if self.lat4:
            if self.single_orbital:
                m2 = [metal_name + "2" + mi for mi in self.single_name_m]
                if self.soc_doubled_ham:
                    m2 = [mi + "u" for mi in m2] + [mi + "d" for mi in m2]
                for i in range(n_m):
                    lat.add_one_sublattice(m2[i], [0, 0], np.real(h_0_m[i, i]))
                for i in range(n_m):
                    for j in np.arange(i + 1, n_m):
                        lat.add_one_hopping([0, 0], m2[i], m2[j], h_0_m[i, j])  # could be the transpose
            else:
                lat.add_one_sublattice(metal_name + "2", [self.a/2, self.a*self._rt3/2], h_0_m.T)

        if self.single_orbital:
            c = [chalcogenide_name + ci for ci in self.single_name_x]
            if self.soc_doubled_ham:
                c = [ci + "u" for ci in c] + [ci + "d" for ci in c]
            n_c = len(c)
            for i in range(n_c):
                lat.add_one_sublattice(c[i], [self.a / 2, self.a * self._rt3 / 6], np.real(h_0_c[i, i]))
            for i in range(n_c):
                for j in np.arange(i + 1, n_c):
                    lat.add_one_hopping([0, 0], c[i], c[j], h_0_c[i, j])  # could be the transpose
        else:
            lat.add_one_sublattice(chalcogenide_name, [self.a/2, self.a*self._rt3/6], h_0_c.T)

        if self.lat4:
            if self.single_orbital:
                c2 = [chalcogenide_name + "2" + ci for ci in self.single_name_x]
                if self.soc_doubled_ham:
                    c2 = [ci + "u" for ci in c2] + [ci + "d" for ci in c2]
                for i in range(n_c):
                    lat.add_one_sublattice(c2[i], [self.a / 2, self.a * self._rt3 / 6], np.real(h_0_c[i, i]))
                for i in range(n_c):
                    for j in np.arange(i + 1, n_c):
                        lat.add_one_hopping([0, 0], c2[i], c2[j], h_0_c[i, j])  # could be the transpose
            else:
                lat.add_one_sublattice(chalcogenide_name + "2", [0, self.a*2*self._rt3/3], h_0_c.T)

        ur_m = block_diag(self._ur[:2, :2], self._ur)
        ur_c = block_diag(self._ur, self._ur)

        (h_1_m_1, h_1_m_2, h_1_m_3) = self.ham(block_diag(self.h_1_ba, self.h_1_dc), ur_c, ur_m)
        (h_2_m_1, h_2_m_2, h_2_m_3) = self.ham(block_diag(self.h_2_aa, self.h_2_cc), ur_m, ur_m)
        (h_2_c_1, h_2_c_2, h_2_c_3) = self.ham(block_diag(self.h_2_bb, self.h_2_dd), ur_c, ur_c)
        (h_3_m_1, h_3_m_2, h_3_m_3) = self.ham(block_diag(np.zeros([3, 2]), self.h_3_dc), ur_c, ur_m)

        if self.soc_doubled_ham:
            h_1_m_1 = np.kron(np.eye(2), h_1_m_1)
            h_1_m_2 = np.kron(np.eye(2), h_1_m_2)
            h_1_m_3 = np.kron(np.eye(2), h_1_m_3)
            h_2_m_1 = np.kron(np.eye(2), h_2_m_1)
            h_2_m_2 = np.kron(np.eye(2), h_2_m_2)
            h_2_m_3 = np.kron(np.eye(2), h_2_m_3)
            h_2_c_1 = np.kron(np.eye(2), h_2_c_1)
            h_2_c_2 = np.kron(np.eye(2), h_2_c_2)
            h_2_c_3 = np.kron(np.eye(2), h_2_c_3)
            h_3_m_1 = np.kron(np.eye(2), h_3_m_1)
            h_3_m_2 = np.kron(np.eye(2), h_3_m_2)
            h_3_m_3 = np.kron(np.eye(2), h_3_m_3)
        if self.single_orbital:
            if not self.lat4:
                for i in range(n_m):
                    for j in range(n_m):
                        lat.add_hoppings(([1, 0], m[i], m[j], h_2_m_1.T[i, j]),
                                         ([0, 1], m[i], m[j], h_2_m_2.T[i, j]),
                                         ([-1, -1], m[i], m[j], h_2_m_3.T[i, j]))
                    for j in range(n_c):
                        lat.add_hoppings(([-1, -1], m[i], c[j], h_1_m_1.T[i, j]),
                                         ([0, 0], m[i], c[j], h_1_m_2.T[i, j]),
                                         ([-1, 0], m[i], c[j], h_1_m_3.T[i, j]),
                                         ([0, 1], m[i], c[j], h_3_m_1.T[i, j]),
                                         ([-2, -1], m[i], c[j], h_3_m_2.T[i, j]),
                                         ([0, -1], m[i], c[j], h_3_m_3.T[i, j]))
                for i in range(n_c):
                    for j in range(n_c):
                        lat.add_hoppings(([1, 0], c[i], c[j], h_2_c_1.T[i, j]),
                                         ([0, 1], c[i], c[j], h_2_c_2.T[i, j]),
                                         ([-1, -1], c[i], c[j], h_2_c_3.T[i, j]))
            else:
                for i in range(n_m):
                    for j in range(n_m):
                        lat.add_hoppings(([1, 0], m[i], m[j], h_2_m_1.T[i, j]),
                                         ([1, 0], m2[i], m2[j], h_2_m_1.T[i, j]),
                                         ([-1, 0], m[i], m2[j], h_2_m_2.T[i, j]),
                                         ([0, 1], m2[i], m[j], h_2_m_2.T[i, j]),
                                         ([-1, -1], m[i], m2[j], h_2_m_3.T[i, j]),
                                         ([0, 0], m2[i], m[j], h_2_m_3.T[i, j]))
                    for j in range(n_c):
                        lat.add_hoppings(([0, 0], m2[i], c[j], h_1_m_1.T[i, j]),
                                         ([0, -1], m[i], c2[j], h_1_m_1.T[i, j]),
                                         ([0, 0], m[i], c[j], h_1_m_2.T[i, j]),
                                         ([1, 0], m2[i], c2[j], h_1_m_2.T[i, j]),
                                         ([-1, 0], m[i], c[j], h_1_m_3.T[i, j]),
                                         ([0, 0], m2[i], c2[j], h_1_m_3.T[i, j]),
                                         ([0, 1], m2[i], c[j], h_3_m_1.T[i, j]),
                                         ([0, 0], m[i], c2[j], h_3_m_1.T[i, j]),
                                         ([-1, -1], m[i], c2[j], h_3_m_2.T[i, j]),
                                         ([-1, 0], m2[i], c[j], h_3_m_2.T[i, j]),
                                         ([1, -1], m[i], c2[j], h_3_m_3.T[i, j]),
                                         ([1, 0], m2[i], c[j], h_3_m_3.T[i, j]))
                for i in range(n_c):
                    for j in range(n_c):
                        lat.add_hoppings(([1, 0], c[i], c[j], h_2_c_1.T[i, j]),
                                         ([1, 0], c2[i], c2[j], h_2_c_1.T[i, j]),
                                         ([0, 0], c[i], c2[j], h_2_c_2.T[i, j]),
                                         ([-1, 1], c2[i], c[j], h_2_c_2.T[i, j]),
                                         ([0, -1], c[i], c2[j], h_2_c_3.T[i, j]),
                                         ([-1, 0], c2[i], c[j], h_2_c_3.T[i, j]))
        else:
            lat.register_hopping_energies({
                'h_1_m_1': h_1_m_1.T,
                'h_1_m_2': h_1_m_2.T,
                'h_1_m_3': h_1_m_3.T,
                'h_2_m_1': h_2_m_1.T,
                'h_2_c_1': h_2_c_1.T,
                'h_2_m_2': h_2_m_2.T,
                'h_2_c_2': h_2_c_2.T,
                'h_2_m_3': h_2_m_3.T,
                'h_2_c_3': h_2_c_3.T,
                'h_3_m_1': h_3_m_1.T,
                'h_3_m_2': h_3_m_2.T,
                'h_3_m_3': h_3_m_3.T
            })

            m, c = metal_name, chalcogenide_name

            if not self.lat4:
                lat.add_hoppings(([-1, -1], m, c, 'h_1_m_1'),
                                 ([0, 0], m, c, 'h_1_m_2'),
                                 ([-1, 0], m, c, 'h_1_m_3'),
                                 ([1, 0], m, m, 'h_2_m_1'),
                                 ([1, 0], c, c, 'h_2_c_1'),
                                 ([0, 1], m, m, 'h_2_m_2'),
                                 ([0, 1], c, c, 'h_2_c_2'),
                                 ([-1, -1], m, m, 'h_2_m_3'),
                                 ([-1, -1], c, c, 'h_2_c_3'),
                                 ([0, 1], m, c, 'h_3_m_1'),
                                 ([-2, -1], m, c, 'h_3_m_2'),
                                 ([0, -1], m, c, 'h_3_m_3'))
            else:
                m1 = m
                c1 = c
                m2 = m + "2"
                c2 = c + "2"
                lat.add_hoppings(([0, 0], m2, c1, 'h_1_m_1'),
                                 ([0, -1], m1, c2, 'h_1_m_1'),
                                 ([0, 0], m1, c1, 'h_1_m_2'),
                                 ([1, 0], m2, c2, 'h_1_m_2'),
                                 ([-1, 0], m1, c1, 'h_1_m_3'),
                                 ([0, 0], m2, c2, 'h_1_m_3'),
                                 ([1, 0], m1, m1, 'h_2_m_1'),
                                 ([1, 0], m2, m2, 'h_2_m_1'),
                                 ([-1, 0], m1, m2, 'h_2_m_2'),
                                 ([0, 1], m2, m1, 'h_2_m_2'),
                                 ([-1, -1], m1, m2, 'h_2_m_3'),
                                 ([0, 0], m2, m1, 'h_2_m_3'),
                                 ([1, 0], c1, c1, 'h_2_c_1'),
                                 ([1, 0], c2, c2, 'h_2_c_1'),
                                 ([0, 0], c1, c2, 'h_2_c_2'),
                                 ([-1, 1], c2, c1, 'h_2_c_2'),
                                 ([0, -1], c1, c2, 'h_2_c_3'),
                                 ([-1, 0], c2, c1, 'h_2_c_3'),
                                 ([0, 1], m2, c1, 'h_3_m_1'),
                                 ([0, 0], m1, c2, 'h_3_m_1'),
                                 ([-1, -1], m1, c2, 'h_3_m_2'),
                                 ([-1, 0], m2, c1, 'h_3_m_2'),
                                 ([1, -1], m1, c2, 'h_3_m_3'),
                                 ([1, 0], m2, c1, 'h_3_m_3'))
        return lat

    def test1(self):
        # plot all
        print("Doing every calculation with high precision,\n this can take a minute.")
        self.lat4 = False
        for soc in [False, True]:
            self.soc = soc

            grid = plt.GridSpec(2, 2, hspace=0.4)
            plt.figure(figsize=(6.7, 8))

            for square, name in zip(grid, ["MoS2", "WS2", "MoSe2", "WSe2"]):
                self.name = name
                model = pb.Model(self.lattice(), pb.translational_symmetry())
                solver = pb.solver.lapack(model)

                k_points = model.lattice.brillouin_zone()
                gamma = [0, 0]
                k = k_points[0]
                m = (k_points[0] + k_points[1]) / 2

                plt.subplot(square, title=name)
                bands = solver.calc_bands(gamma, k, m, gamma, step=0.005)
                bands.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"], lw=1.5)

    def test2(self, u=None, soc=False):
        plt.figure()
        # plot bands comparing lat4=False and lat4=True with a certain strain vector, giving the largest error between
        # the two models

        bands = []
        for lat4 in [False, True]:
            self.lat4 = lat4
            self.soc = soc
            model = pb.Model(self.lattice(),
                             pb.translational_symmetry())
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

"""Tight-binding models for group 1 transition metal dichalcogenides (TMD), 6 band."""
import re
import math
import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag

_default_11band_params_SK = {  # from https://journals.aps.org/prb/abstract/10.1103/PhysRevB.88.075409
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


class Group1Tmd11Band:
    def __init__(self, **kwargs):
        self.single_orbital = False
        self.single_name_m = ["-dz2", "-dx2-y2", "-dxy", "-dxz", "-dyz"]
        self.single_name_x = ["-pxe", "-pye", "-pze", "-pxo", "-pyo", "-pzo"]
        # make a list of all the parameters we import during the creation of the various matrices for the hoppings from
        # the table from Fang's papers
        self._rt3 = np.sqrt(3)
        # d: (even-[z², x²-y², xy]; odd-[xz, yz])
        self._ur_m = block_diag(np.array([[1]]), self.rot_mat(2 * (2 * np.pi / 3)), self.rot_mat(2 * np.pi / 3))
        # p: (even-[x, y, z]; odd-[x, y, z])
        self._ur_c = block_diag(self.rot_mat(2 * np.pi / 3), np.array([[1]]),
                                self.rot_mat(2 * np.pi / 3), np.array([[1]]))
        self._name = "MoS2"
        self._lat4 = False
        self._even_odd = True
        self.use_theta = False
        # defaults; also needed to import the matrices from Fang
        self.params = _default_11band_params_SK
        self.soc = False
        self.soc_eo_flip = True
        self.soc_polarized = False
        self.sz = 1.
        self._berry_phase_factor = -1
        # see if there are parameters to overwrite
        # make the lattice vectors
        # add other default parameters
        # see if there are more parameters to overwrite; also the matrices from Fang could be overwritten
        [setattr(self, var, kwargs[var]) for var in [*kwargs]]

    def rot_mat(self, phi=0):
        return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

    @property
    def params(self):
        return self.__params

    @params.setter
    def params(self, params):
        self.__name_list = [*params]
        self.__params = params
        self.name = [*params][0] if self.name not in [*params] else self.name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name not in self.__name_list:
            print("The name %s is not in the params-list" % name)
        else:
            self._name = name
            self._generate_matrices()

    @property
    def lat4(self):
        return self._lat4

    @lat4.setter
    def lat4(self, lat4):
        self._lat4 = lat4
        self._generate_matrices()

    @property
    def even_odd(self):
        return self._even_odd

    @even_odd.setter
    def even_odd(self, even_odd):
        self._even_odd = even_odd

    @property
    def a1(self):
        return np.array([1, 0]) * self.a

    @property
    def a2(self):
        return (np.array([0, self._rt3]) if self.lat4 else np.array([-1 / 2, self._rt3 / 2])) * self.a

    @property
    def soc_doubled_ham(self):
        return not self.soc_polarized if self.soc else False

    @property
    def soc_eo_flip_used(self):
        return self.soc and self.soc_eo_flip and not self.soc_polarized

    @property
    def n_valence_band(self):
        return 13 if self.soc_doubled_ham else 6

    @property
    def n_bands(self):
        return 22 if self.soc_doubled_ham else 11

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
                 [self._rt3 * np.cos(theta) * np.sin(theta)**2 * t1
                  - np.cos(theta) * (np.sin(theta)**2 - 1 / 2 * np.cos(theta)**2) * t2,
                  np.cos(theta) * np.sin(theta)**2 * t1 + self._rt3 / 2 * np.cos(theta)**3 * t2, 0.0, 0.0,
                  np.sin(theta) * (1 - 2 * np.cos(theta) ** 2) * t1
                  + self._rt3 * np.cos(theta) ** 2 * np.sin(theta) * t2],
                 [self._rt3 * np.sin(theta) * np.cos(theta) ** 2 * t1
                  + np.sin(theta) * (np.sin(theta) ** 2 - 1 / 2 * np.cos(theta) ** 2) * t2,
                  -np.sin(theta) * np.cos(theta)**2 * (-t1 + self._rt3/2 * t2), 0.0, 0.0,
                  - np.cos(theta) * (1 - 2 * np.sin(theta) ** 2) * t1
                  - self._rt3 * np.sin(theta) ** 2 * np.cos(theta) * t2]]
            ) if theta is not None and self.use_theta else 1 / (7 * np.sqrt(7)) * np.array(
                [[0.0, 0.0, -14 * t1, 7 * self._rt3 * t1, 0.0],
                 [6 * self._rt3 * t1 - 2 * t2, 6 * t1 + 4 * self._rt3 * t2, 0.0, 0.0, -self._rt3 * t1 + 12 * t2],
                 [12 * t1 + self._rt3 * t2, 4 * self._rt3 * t1 - 6 * t2, 0.0, 0.0, -2 * t1 - 6 * self._rt3 * t2]])


        h_0_m = np.diag((delta_0, delta_2, delta_2, delta_1, delta_1))
        h_1_m = h_1_mat(v_pdp, v_pds)
        if self.even_odd:
            h_0_c_o = np.diag((delta_p - v_ppp_0, delta_p - v_ppp_0, delta_z + v_pps_0))
            h_0_c_e = np.diag((delta_p + v_ppp_0, delta_p + v_ppp_0, delta_z - v_pps_0))
            h_0_c = block_diag(h_0_c_e, h_0_c_o)
            h_1_m_e = np.sqrt(2) * h_1_m[:, :3]
            h_1_m_o = np.sqrt(2) * h_1_m[:, 3:]
            h_1_m = block_diag(h_1_m_e, h_1_m_o)
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
        ur_mat = block_diag(self.rot_mat(2 * np.pi / 3), np.array([[1]]))
        h_2_m_3 = np.dot(ur_mat, np.dot(np.diag((v_dds, v_ddd, v_ddp)), ur_mat.T))
        h_2_m_2 = np.diag((v_ddp, v_ddd))
        h_2_m = block_diag(h_2_m_3, h_2_m_2)
        keys = ["h_0_m", "h_0_c", "h_1_m", "h_2_m", "h_2_c", "a", "lamb_m", "lamb_c"]
        values = [h_0_m, h_0_c, h_1_m, h_2_m, h_2_c, a, lamb_m, lamb_c]
        [setattr(self, key, value) for key, value in zip(keys, values)]

    def ham(self, h, ur_l, ur_r):
        return h, ur_l.T.dot(h.dot(ur_r)), ur_l.dot(h.dot(ur_r.T))

    def lattice(self):
        lat = pb.Lattice(a1=self.a1, a2=self.a2)

        metal_name, chalcogenide_name = re.findall("[A-Z][a-z]*", self.name)

        def h_0_m_sz(sz):
            s_part_o = sz * self.lamb_m * 1j * np.array([[0, -1 / 2], [1 / 2, 0]])
            s_part_e = sz * self.lamb_m * 1j * np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
            return self.h_0_m + block_diag(s_part_e, s_part_o)

        def h_0_c_sz(sz):
            s_part_c = sz * self.lamb_c * 1j / 2 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
            return self.h_0_c + block_diag(s_part_c, s_part_c)

        if self.soc:
            if self.soc_polarized:
                h_0_m = h_0_m_sz(self.sz)
                h_0_c = h_0_c_sz(self.sz)
            else:
                soc_part_m = np.zeros((5, 5)) * 1j
                soc_part_c = np.zeros((6, 6)) * 1j
                if self.soc_eo_flip:
                    soc_part_m[3:, :3] = self.sz * self.lamb_m * np.array(
                        [[self._rt3 / 2, -1 / 2, 1j / 2],
                         [-1j / 2 * self._rt3, -1j / 2, -1 / 2]])
                    soc_part_m[:3, 3:] = -soc_part_m[3:, :3].T

                    soc_part_c[:3, 3:] = -1j*self.sz * self.lamb_c * np.array(
                        [[0, 0, 1/2],
                         [0, 0, -1j/2],
                         [-1/2, 1j / 2, 0]])
                    soc_part_c[3:, :3] = -soc_part_c[:3, 3:].T
                h_0_m = np.concatenate((np.concatenate((h_0_m_sz(self.sz), soc_part_m.conj().T), axis=1),
                                        np.concatenate((soc_part_m, h_0_m_sz(-self.sz)), axis=1)), axis=0)
                h_0_c = np.concatenate((np.concatenate((h_0_c_sz(self.sz), soc_part_c.conj().T), axis=1),
                                        np.concatenate((soc_part_c, h_0_c_sz(-self.sz)), axis=1)), axis=0)
        else:
            h_0_m = h_0_m_sz(0.)
            h_0_c = h_0_c_sz(0.)

        if self.single_orbital:
            m = [metal_name + mi for mi in self.single_name_m]
            if self.soc_doubled_ham:
                m = [mi + "u" for mi in m] + [mi + "d" for mi in m]
            n_m = len(m)
            for i in range(n_m):
                lat.add_one_sublattice(m[i], [0, 0], np.real(h_0_m[i, i]))
            for i in range(n_m):
                for j in np.arange(i + 1, n_m):
                    lat.add_one_hopping([0, 0], m[i], m[j], h_0_m[i, j])  # could be the transpose
        else:
            lat.add_one_sublattice(metal_name, [0, 0], h_0_m.T)

        if self.lat4:
            if self.single_orbital:
                m2 = [metal_name + "2" + mi for mi in self.single_name_m]
                if self.soc_doubled_ham:
                    m2 = [mi + "u" for mi in m2] + [mi + "d" for mi in m2]
                for i in range(n_m):
                    lat.add_one_sublattice(m2[i], [0, 0], np.real(h_0_m[i, i]))
                for i in range(n_m):
                    for j in np.arange(i + 1, n_m):
                        lat.add_one_hopping([0, 0], m2[i], m2[j], h_0_m[i, j])  # could be the transpose
            else:
                lat.add_one_sublattice(metal_name + "2", [self.a / 2, self.a * self._rt3 / 2], h_0_m.T)

        if self.single_orbital:
            c = [chalcogenide_name + ci for ci in self.single_name_x]
            if self.soc_doubled_ham:
                c = [ci + "u" for ci in c] + [ci + "d" for ci in c]
            n_c = len(c)
            for i in range(n_c):
                lat.add_one_sublattice(c[i], [self.a / 2, self.a * self._rt3 / 6], np.real(h_0_c[i, i]))
            for i in range(n_c):
                for j in np.arange(i + 1, n_c):
                    lat.add_one_hopping([0, 0], c[i], c[j], h_0_c[i, j])  # could be the transpose
        else:
            lat.add_one_sublattice(chalcogenide_name, [self.a / 2, self.a * self._rt3 / 6], h_0_c.T)

        if self.lat4:
            if self.single_orbital:
                c2 = [chalcogenide_name + "2" + ci for ci in self.single_name_x]
                if self.soc_doubled_ham:
                    c2 = [ci + "u" for ci in c2] + [ci + "d" for ci in c2]
                for i in range(n_c):
                    lat.add_one_sublattice(c2[i], [self.a / 2, self.a * self._rt3 / 6], np.real(h_0_c[i, i]))
                for i in range(n_c):
                    for j in np.arange(i + 1, n_c):
                        lat.add_one_hopping([0, 0], c2[i], c2[j], h_0_c[i, j])  # could be the transpose
            else:
                lat.add_one_sublattice(chalcogenide_name + "2", [0, self.a * 2 * self._rt3 / 3], h_0_c.T)

        (h_1_m_1, h_1_m_2, h_1_m_3) = self.ham(self.h_1_m, self._ur_c, self._ur_m)
        (h_2_m_1, h_2_m_2, h_2_m_3) = self.ham(self.h_2_m, self._ur_m, self._ur_m)
        (h_2_c_1, h_2_c_2, h_2_c_3) = self.ham(self.h_2_c, self._ur_c, self._ur_c)

        if self.soc_doubled_ham:
            h_1_m_1 = np.kron(np.eye(2), h_1_m_1)
            h_1_m_2 = np.kron(np.eye(2), h_1_m_2)
            h_1_m_3 = np.kron(np.eye(2), h_1_m_3)
            h_2_m_1 = np.kron(np.eye(2), h_2_m_1)
            h_2_m_2 = np.kron(np.eye(2), h_2_m_2)
            h_2_m_3 = np.kron(np.eye(2), h_2_m_3)
            h_2_c_1 = np.kron(np.eye(2), h_2_c_1)
            h_2_c_2 = np.kron(np.eye(2), h_2_c_2)
            h_2_c_3 = np.kron(np.eye(2), h_2_c_3)
        if self.single_orbital:
            if not self.lat4:
                for i in range(n_m):
                    for j in range(n_m):
                        lat.add_hoppings(([1, 0], m[i], m[j], h_2_m_1.T[i, j]),
                                         ([0, 1], m[i], m[j], h_2_m_2.T[i, j]),
                                         ([-1, -1], m[i], m[j], h_2_m_3.T[i, j]))
                    for j in range(n_c):
                        lat.add_hoppings(([-1, -1], m[i], c[j], h_1_m_1.T[i, j]),
                                         ([0, 0], m[i], c[j], h_1_m_2.T[i, j]),
                                         ([-1, 0], m[i], c[j], h_1_m_3.T[i, j]))
                for i in range(n_c):
                    for j in range(n_c):
                        lat.add_hoppings(([1, 0], c[i], c[j], h_2_c_1.T[i, j]),
                                         ([0, 1], c[i], c[j], h_2_c_2.T[i, j]),
                                         ([-1, -1], c[i], c[j], h_2_c_3.T[i, j]))
            else:
                for i in range(n_m):
                    for j in range(n_m):
                        lat.add_hoppings(([1, 0], m[i], m[j], h_2_m_1.T[i, j]),
                                         ([1, 0], m2[i], m2[j], h_2_m_1.T[i, j]),
                                         ([-1, 0], m[i], m2[j], h_2_m_2.T[i, j]),
                                         ([0, 1], m2[i], m[j], h_2_m_2.T[i, j]),
                                         ([-1, -1], m[i], m2[j], h_2_m_3.T[i, j]),
                                         ([0, 0], m2[i], m[j], h_2_m_3.T[i, j]))
                    for j in range(n_c):
                        lat.add_hoppings(([0, 0], m2[i], c[j], h_1_m_1.T[i, j]),
                                         ([0, -1], m[i], c2[j], h_1_m_1.T[i, j]),
                                         ([0, 0], m[i], c[j], h_1_m_2.T[i, j]),
                                         ([1, 0], m2[i], c2[j], h_1_m_2.T[i, j]),
                                         ([-1, 0], m[i], c[j], h_1_m_3.T[i, j]),
                                         ([0, 0], m2[i], c2[j], h_1_m_3.T[i, j]))
                for i in range(n_c):
                    for j in range(n_c):
                        lat.add_hoppings(([1, 0], c[i], c[j], h_2_c_1.T[i, j]),
                                         ([1, 0], c2[i], c2[j], h_2_c_1.T[i, j]),
                                         ([0, 0], c[i], c2[j], h_2_c_2.T[i, j]),
                                         ([-1, 1], c2[i], c[j], h_2_c_2.T[i, j]),
                                         ([0, -1], c[i], c2[j], h_2_c_3.T[i, j]),
                                         ([-1, 0], c2[i], c[j], h_2_c_3.T[i, j]))
        else:
            lat.register_hopping_energies({
                'h_1_m_1': h_1_m_1.T,
                'h_1_m_2': h_1_m_2.T,
                'h_1_m_3': h_1_m_3.T,
                'h_2_m_1': h_2_m_1.T,
                'h_2_m_2': h_2_m_2.T,
                'h_2_m_3': h_2_m_3.T,
                'h_2_c_1': h_2_c_1.T,
                'h_2_c_2': h_2_c_2.T,
                'h_2_c_3': h_2_c_3.T
            })

            m, c = metal_name, chalcogenide_name

            if not self.lat4:
                lat.add_hoppings(([-1, -1], m, c, 'h_1_m_1'),
                                 ([0, 0], m, c, 'h_1_m_2'),
                                 ([-1, 0], m, c, 'h_1_m_3'),
                                 ([1, 0], m, m, 'h_2_m_1'),
                                 ([1, 0], c, c, 'h_2_c_1'),
                                 ([0, 1], m, m, 'h_2_m_2'),
                                 ([0, 1], c, c, 'h_2_c_2'),
                                 ([-1, -1], m, m, 'h_2_m_3'),
                                 ([-1, -1], c, c, 'h_2_c_3'))
            else:
                m1 = m
                c1 = c
                m2 = m + "2"
                c2 = c + "2"
                lat.add_hoppings(([0, 0], m2, c1, 'h_1_m_1'),
                                 ([0, -1], m1, c2, 'h_1_m_1'),
                                 ([0, 0], m1, c1, 'h_1_m_2'),
                                 ([1, 0], m2, c2, 'h_1_m_2'),
                                 ([-1, 0], m1, c1, 'h_1_m_3'),
                                 ([0, 0], m2, c2, 'h_1_m_3'),
                                 ([1, 0], m1, m1, 'h_2_m_1'),
                                 ([1, 0], m2, m2, 'h_2_m_1'),
                                 ([-1, 0], m1, m2, 'h_2_m_2'),
                                 ([0, 1], m2, m1, 'h_2_m_2'),
                                 ([-1, -1], m1, m2, 'h_2_m_3'),
                                 ([0, 0], m2, m1, 'h_2_m_3'),
                                 ([1, 0], c1, c1, 'h_2_c_1'),
                                 ([1, 0], c2, c2, 'h_2_c_1'),
                                 ([0, 0], c1, c2, 'h_2_c_2'),
                                 ([-1, 1], c2, c1, 'h_2_c_2'),
                                 ([0, -1], c1, c2, 'h_2_c_3'),
                                 ([-1, 0], c2, c1, 'h_2_c_3'))
        return lat

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

