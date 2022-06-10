"""Tight-binding models for group 4 transition metal dichalcogenides (TMD), 11 band."""
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from .tmd_abstract_lattice import AbstractLattice

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

_bert_11band_params_fang = {  # from https://link.aps.org/doi/10.1103/PhysRevB.92.205108
    #"name"         a,    eps1,    eps3,    eps4,    eps6,    eps7,    eps9,   eps10,  t1_1_1,  t1_2_2,
    #          t1_3_3,  t1_4_4,  t1_5_5,  t1_6_6,  t1_7_7,  t1_8_8,  t1_9_9,t1_10_10,t1_11_11,  t1_3_5,
    #          t1_6_8,  1_9_11,  t1_1_2,  t1_3_4,  t1_4_5,  t1_6_7,  t1_7_8, t1_9_10,t1_10_11,  t5_4_1,
    #          t5_3_2,  t5_5_2,  t5_9_6, t5_11_6, t5_10_7,  t5_9_8, t5_11_8,  t6_9_6, t6_11_6,  t6_9_8,
    #         t6_11_8,       c,     dXX,     dXM,   lambM,   lambX
    "MoS2":  [  0.318,  0.618, -1.179, -0.55 , -0.023,  0.209, -2.964, -1.79 ,
               -0.451,  0.367, -0.101,  1.06 , -0.533, -0.193,  0.309, -0.571,
               -0.25 ,  0.878,  0.003, -0.002,  0.504,  0.011, -0.587, -0.101,
               -0.288, -0.393, -0.366, -0.054, -0.028, -0.652, -1.259,  2.113,
               -0.8  , -1.009,  1.548, -0.898,  0.332, -0.005, -0.147, -0.379,
               -0.068, 12.29 ,  3.13 ,  2.41 ,  0.084,  0.056],
}

class Group4Tmd6Band(AbstractLattice):
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
    """
    def __init__(self, **kwargs):

        lattice_orbital_dict = {"l": {"Mo": [2, -2, 0], "S": [-1, 1, 0]},
                                "orbs": {"Mo": ["dxy", "dx2y2", "dz2"],
                                         "S": ["pxe", "pye", "pze"]},
                                "group": {"Mo": [2, 2, 0], "S": [2, 2, 3]}}
        super().__init__(orbital=lattice_orbital_dict, n_v=6, n_b=11)
        self.single_orbital = False
        self.params = _default_11band_params
        self._berry_phase_factor = -1
        [setattr(self, var, kwargs[var]) for var in [*kwargs]]

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

        def matrix_1(t_0_n, t_1_n, t_2_n, t_3_n, t_4_n):
            return np.array([[t_0_n, 0, 0], [0, t_1_n, t_2_n], [0, t_3_n, t_4_n]])

        def matrix_2(t_0_2, t_1_2, t_2_2, t_3_2, t_4_2, t_5_2):
            return np.array([[t_0_2, t_3_2, t_4_2], [-t_3_2, t_1_2, t_5_2], [-t_4_2, t_5_2, t_2_2]])

        # make the matrices
        h_0_cc = matrix_0(eps6, eps7)
        h_0_dd = matrix_0(eps9, eps10)
        h_1_dc = matrix_1(t5_10_7, t5_11_8, t5_11_6, t5_9_8, t5_9_6)
        h_2_cc = matrix_2(t1_7_7, t1_8_8, t1_6_6, t1_7_8, -t1_6_7, t1_6_8)
        h_2_dd = matrix_2(t1_10_10, t1_11_11, t1_9_9, t1_10_11, -t1_9_10, t1_9_11)
        h_0_m = h_0_cc
        h_0_c = h_0_dd
        h_1_m = h_1_dc
        h_2_m = h_2_cc
        h_2_c = h_2_dd
        keys = ["h_0_m", "h_0_c", "h_1_m", "h_2_m", "h_2_c", "a", "lamb_m", "lamb_c"]
        values = [h_0_m, h_0_c, h_1_m, h_2_m, h_2_c, a, lamb_m, lamb_c]
        self.lattice_params(**dict([(key, value) for key, value in zip(keys, values)]))

    def test1(self, soc=False, lat4=False, path=1):
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

if __name__ == "__main__":
    group4_class = Group4Tmd11Band()
    #group4_class.test1()
    #res = group4_class.test2()