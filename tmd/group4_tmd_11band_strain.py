"""Tight-binding models for group 4 transition metal dichalcogenides (TMD), 11 band."""
import re
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from .tmd_abstract_lattice import AbstractLattice

_default_11band_strain_params_paper = {  # from https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.075106
    # "name"
    #        a,          eps_0_bb, eps_0_cc, eps_0_dd,          t_0_2_bb, t_0_2_cc, t_0_2_dd,
    #      d_0,          eps_1_bb, eps_1_cc, eps_1_dd,          t_1_2_bb, t_1_2_cc, t_1_2_dd,
    #      d_1,                                                 t_2_2_bb, t_2_2_cc, t_2_2_dd,
    #                    a_0_0_bb, a_0_0_cc, a_0_0_dd,          t_3_2_bb, t_3_2_cc, t_3_2_dd,
    #                    a_1_0_bb, a_1_0_cc, a_1_0_dd,          t_4_2_bb, t_4_2_cc, t_4_2_dd,
    # eps_1_aa,                                                 t_5_2_bb, t_5_2_cc, t_5_2_dd,
    # a_1_0_aa,          b_0_0_bb, b_0_0_cc, b_0_0_dd,
    # b_0_0_aa,          b_1_0_bb, b_1_0_cc, b_1_0_dd,          a_0_2_bb, a_0_2_cc, a_0_2_dd,
    #                                                           a_1_2_bb, a_1_2_cc, a_1_2_dd,
    # t_0_2_aa,                                                 a_2_2_bb, a_2_2_cc, a_2_2_dd,
    # t_1_2_aa,          t_0_1_ba, t_0_1_dc, t_0_3_dc,          a_3_2_bb, a_3_2_cc, a_3_2_dd,
    # t_3_2_aa,          t_1_1_ba, t_1_1_dc, t_1_3_dc,          a_4_2_bb, a_4_2_cc, a_4_2_dd,
    #                              t_2_1_dc, t_2_3_dc,          a_5_2_bb, a_5_2_cc, a_5_2_dd,
    # a_0_2_aa,          t_3_1_ba, t_3_1_dc, t_3_3_dc,
    # a_1_2_aa,                    t_4_1_dc, t_4_3_dc,          b_0_2_bb, b_0_2_cc, b_0_2_dd,
    # a_3_2_aa,                                                 b_1_2_bb, b_1_2_cc, b_1_2_dd,
    #                    a_0_1_ba, a_0_1_dc, a_0_3_dc,          b_2_2_bb, b_2_2_cc, b_2_2_dd,
    # b_0_2_aa,          a_1_1_ba, a_1_1_dc, a_1_3_dc,          b_3_2_bb, b_3_2_cc, b_3_2_dd,
    # b_1_2_aa,                    a_2_1_dc, a_2_3_dc,          b_4_2_bb, b_4_2_cc, b_4_2_dd,
    # b_3_2_aa,          a_3_1_ba, a_3_1_dc, a_3_3_dc,          b_5_2_bb, b_5_2_cc, b_5_2_dd,
    # b_6_2_aa,                    a_4_1_dc, a_4_3_dc,          b_6_2_bb, b_6_2_cc, b_6_2_dd,
    #                                                           b_7_2_bb, b_7_2_cc, b_7_2_dd,
    #                    b_0_1_ba, b_0_1_dc, b_0_3_dc,          b_8_2_bb, b_8_2_cc, b_8_2_dd,
    #   lamb_m,          b_1_1_ba, b_1_1_dc, b_1_3_dc,
    #   lamb_c,                    b_2_1_dc, b_2_3_dc,
    #                    b_3_1_ba, b_3_1_dc, b_3_3_dc,
    #                              b_4_1_dc, b_4_3_dc,
    #                    b_5_1_ba, b_5_1_dc, b_5_3_dc,
    #                              b_6_1_dc, b_6_3_dc,
    #                    b_7_1_ba, b_7_1_dc, b_7_3_dc,
    #                    b_8_1_ba, b_8_1_dc, b_8_3_dc
    "MoS2": [
        .3182, -6.720, -6.082, -8.839, 0.865, 0.275, 0.912,
        .1564, -7.235, -5.856, -7.850, -0.187, -0.558, 0.006,
        .0517, -0.174, -0.298, -0.192,
        1.623, -1.021, -0.858, -0.070, -0.249, -0.038,
        -1.500, -1.817, -3.317, 0.100, 0.114, -0.106,
        -4.873, -0.068, 0.410, 0.008,
        -2.498, -0.094, -0.370, -1.142,
        -0.890, 0.273, -0.043, 0.720, -1.841, -1.027, -1.425,
        -0.027, 1.544, -0.057,
        -0.206, 0.444, 1.032, 0.644,
        0.031, -0.789, 1.411, 0.014, -0.045, 0.206, -0.170,
        -0.257, 2.158, 0.652, -0.245, -0.210, 0.285, -0.199,
        -0.940, -0.150, 0.141, -0.738, 0.065,
        -0.258, -1.379, -0.954, -0.221,
        -0.202, -0.883, -0.069, -2.203, -0.910, -2.013,
        0.705, 0.768, 1.337, 0.828,
        0.545, -0.486, 0.173, 0.350, 0.376, 0.540,
        -0.676, -0.605, 0.843, 0.204, -0.065, -0.003, 0.143,
        -0.192, 2.178, 0.567, -0.208, 0.188, -0.056,
        0.555, 1.845, 0.446, 0.744, 0.096, -0.779, 0.082,
        -0.095, -0.208, 0.035, 0.482, -0.634, 0.744,
        -0.146, 0.288, 0.051,
        -1.076, 1.724, -0.178, -0.089, -0.152, -0.099,
        0.0836, 0.401, -0.353, -1.069,
        0.0556, -2.204, -0.070,
        -2.100, -0.682, -0.267,
        -0.850, -0.281,
        0.859, 0.899, -0.690,
        -0.542, -0.382,
        -0.377, -2.093, -0.340,
        -0.836, 1.101, 0.015],
    "MoSe2": [
        .3317, -5.986, -5.559, -8.231, 0.964, 0.251, 0.991,
        .1669, -6.502, -5.314, -7.110, -0.172, -0.473, -0.004,
        .0572, -0.211, -0.264, -0.217,
        1.396, -1.090, -0.742, -0.068, -0.201, -0.039,
        -1.440, -2.023, -3.316, 0.076, 0.096, -0.121,
        -4.547, -0.074, 0.352, 0.005,
        -2.341, -0.121, -0.296, -1.146,
        -0.810, 0.270, 0.004, 0.829, -1.979, -0.951, -1.586,
        -0.103, 1.333, -0.072,
        -0.146, 0.536, 0.885, 0.668,
        0.017, -0.695, 1.268, 0.017, -0.059, 0.195, -0.162,
        -0.191, 1.941, 0.554, -0.215, -0.123, 0.236, -0.202,
        -0.874, -0.155, 0.142, -0.596, 0.050,
        -0.309, -1.326, -0.858, -0.223,
        -0.125, -0.772, -0.069, -2.378, -0.793, -2.180,
        0.514, 0.827, 1.108, 0.884,
        0.408, -0.407, 0.175, 0.445, 0.333, 0.576,
        -0.588, -0.417, 0.825, 0.185, -0.016, 0.008, 0.155,
        -0.118, 1.928, 0.554, -0.146, 0.126, -0.026,
        0.416, 1.718, 0.272, 0.760, 0.112, -0.667, 0.073,
        -0.063, -0.298, 0.062, 0.567, -0.565, 0.777,
        -0.128, 0.255, 0.066,
        -0.897, 1.530, -0.164, -0.092, -0.110, -0.127,
        0.0836, 0.264, -0.367, -0.995,
        0.2470, -1.995, -0.093,
        -1.874, -0.510, -0.292,
        -0.727, -0.290,
        0.770, 0.761, -0.664,
        -0.475, -0.391,
        -0.469, -1.841, -0.299,
        -0.717, 1.005, 0.007],
    "WS2": [
        .3182, -6.838, -5.734, -9.078, 0.873, 0.355, 0.965,
        .1574, -7.250, -5.498, -8.033, -0.218, -0.691, 0.014,
        .0560, -0.175, -0.371, -0.212,
        1.743, -1.212, 0.158, -0.099, -0.304, -0.101,
        -1.854, -1.916, -4.290, 0.110, 0.145, -0.163,
        -4.327, -0.082, 0.488, -0.031,
        -2.631, 0.089, -0.292, -1.390,
        -0.986, 0.487, 0.036, 1.586, -1.844, -1.232, -1.122,
        -0.067, 1.947, -0.162,
        -0.198, 0.434, 1.123, 0.674,
        0.027, -0.884, 1.558, 0.010, -0.042, 0.462, -0.314,
        -0.310, 2.302, 0.664, -0.273, -0.208, 0.365, -0.333,
        -0.993, -0.154, 0.177, -0.654, 0.105,
        -0.453, -1.436, -0.943, -0.265,
        -0.213, -1.005, -0.066, -2.254, -1.068, -1.920,
        0.834, 0.772, 1.240, 1.039,
        0.585, -0.609, 0.537, 0.283, 0.522, 0.580,
        -0.942, -0.482, 1.045, 0.185, -0.054, -0.083, 0.345,
        -0.175, 2.827, 0.623, -0.198, 0.179, 0.062,
        0.649, 1.826, 0.071, 1.055, 0.127, -0.863, 0.130,
        -0.076, -0.241, -0.090, 0.467, -0.960, 0.858,
        -0.128, 0.484, 0.146,
        -1.128, 2.402, -0.345, -0.117, -0.046, -0.236,
        0.2874, 0.140, -0.900, -1.110,
        0.0556, -2.293, -0.125,
        -1.990, -0.306, -0.120,
        -1.184, -0.536,
        0.915, 0.902, -1.093,
        -0.193, -0.644,
        -0.634, -2.934, -0.535,
        -0.944, 1.427, -0.127],
    "WSe2": [
        .3316, -6.066, -5.267, -8.466, 0.977, 0.320, 1.047,
        .1680, -6.494, -5.001, -7.277, -0.198, -0.584, 0.003,
        .0611, -0.217, -0.333, -0.241,
        1.385, -1.012, -0.050, -0.092, -0.245, -0.102,
        -1.724, -1.967, -4.138, 0.079, 0.124, -0.185,
        -4.069, -0.091, 0.423, -0.038,
        -2.357, 0.059, -0.220, -1.337,
        -0.902, 0.482, -0.022, 1.507, -1.986, -1.127, -1.357,
        -0.152, 1.617, -0.159,
        -0.137, 0.557, 1.013, 0.718,
        0.013, -0.773, 1.399, 0.017, -0.074, 0.325, -0.303,
        -0.232, 2.079, 0.567, -0.242, -0.105, 0.291, -0.287,
        -0.905, -0.161, 0.188, -0.564, 0.112,
        -0.490, -1.401, -0.853, -0.263,
        -0.117, -0.896, -0.068, -2.427, -0.966, -2.086,
        0.589, 0.834, 1.179, 1.069,
        0.406, -0.493, 0.468, 0.401, 0.406, 0.556,
        -0.809, -0.322, 0.917, 0.202, 0.015, -0.044, 0.331,
        -0.090, 2.409, 0.653, -0.104, 0.129, 0.063,
        0.480, 1.764, 0.022, 1.050, 0.152, -0.727, 0.112,
        -0.037, -0.238, -0.021, 0.550, -0.776, 0.873,
        -0.157, 0.308, 0.109,
        -0.929, 1.973, -0.321, -0.129, -0.099, -0.224,
        0.2874, -0.029, -0.877, -1.094,
        0.2470, -2.153, -0.114,
        -1.879, -0.276, -0.241,
        -0.897, -0.476,
        0.798, 0.761, -1.022,
        -0.300, -0.651,
        -0.690, -2.447, -0.423,
        -0.793, 1.082, -0.058]
}

_default_11band_strain_params = {  # from https://sites.google.com/view/shiangfang/codes
    # "name"
    #        a,          eps_0_bb, eps_0_cc, eps_0_dd,          t_0_2_bb, t_0_2_cc, t_0_2_dd,
    #      d_0,          eps_1_bb, eps_1_cc, eps_1_dd,          t_1_2_bb, t_1_2_cc, t_1_2_dd,
    #      d_1,                                                 t_2_2_bb, t_2_2_cc, t_2_2_dd,
    #                    a_0_0_bb, a_0_0_cc, a_0_0_dd,          t_3_2_bb, t_3_2_cc, t_3_2_dd,
    #                    a_1_0_bb, a_1_0_cc, a_1_0_dd,          t_4_2_bb, t_4_2_cc, t_4_2_dd,
    # eps_1_aa,                                                 t_5_2_bb, t_5_2_cc, t_5_2_dd,
    # a_1_0_aa,          b_0_0_bb, b_0_0_cc, b_0_0_dd,
    # b_0_0_aa,          b_1_0_bb, b_1_0_cc, b_1_0_dd,          a_0_2_bb, a_0_2_cc, a_0_2_dd,
    #                                                           a_1_2_bb, a_1_2_cc, a_1_2_dd,
    # t_0_2_aa,                                                 a_2_2_bb, a_2_2_cc, a_2_2_dd,
    # t_1_2_aa,          t_0_1_ba, t_0_1_dc, t_0_3_dc,          a_3_2_bb, a_3_2_cc, a_3_2_dd,
    # t_3_2_aa,          t_1_1_ba, t_1_1_dc, t_1_3_dc,          a_4_2_bb, a_4_2_cc, a_4_2_dd,
    #                              t_2_1_dc, t_2_3_dc,          a_5_2_bb, a_5_2_cc, a_5_2_dd,
    # a_0_2_aa,          t_3_1_ba, t_3_1_dc, t_3_3_dc,
    # a_1_2_aa,                    t_4_1_dc, t_4_3_dc,          b_0_2_bb, b_0_2_cc, b_0_2_dd,
    # a_3_2_aa,                                                 b_1_2_bb, b_1_2_cc, b_1_2_dd,
    #                    a_0_1_ba, a_0_1_dc, a_0_3_dc,          b_2_2_bb, b_2_2_cc, b_2_2_dd,
    # b_0_2_aa,          a_1_1_ba, a_1_1_dc, a_1_3_dc,          b_3_2_bb, b_3_2_cc, b_3_2_dd,
    # b_1_2_aa,                    a_2_1_dc, a_2_3_dc,          b_4_2_bb, b_4_2_cc, b_4_2_dd,
    # b_3_2_aa,          a_3_1_ba, a_3_1_dc, a_3_3_dc,          b_5_2_bb, b_5_2_cc, b_5_2_dd,
    # b_6_2_aa,                    a_4_1_dc, a_4_3_dc,          b_6_2_bb, b_6_2_cc, b_6_2_dd,
    #                                                           b_7_2_bb, b_7_2_cc, b_7_2_dd,
    #                    b_0_1_ba, b_0_1_dc, b_0_3_dc,          b_8_2_bb, b_8_2_cc, b_8_2_dd,
    #   lamb_m,          b_1_1_ba, b_1_1_dc, b_1_3_dc,
    #   lamb_c,                    b_2_1_dc, b_2_3_dc,
    #                    b_3_1_ba, b_3_1_dc, b_3_3_dc,
    #                              b_4_1_dc, b_4_3_dc,
    #                    b_5_1_ba, b_5_1_dc, b_5_3_dc,
    #                              b_6_1_dc, b_6_3_dc,
    #                    b_7_1_ba, b_7_1_dc, b_7_3_dc,
    #                    b_8_1_ba, b_8_1_dc, b_8_3_dc
    "MoS2": [
        .31824, -6.7197, -6.0819, -8.8389, 0.86508, 0.27488, 0.91221,
        .1564, -7.2354, -5.8564, -7.8504, -0.18685, -0.5581, 0.0058403,
        .0517, -0.1738, -0.2978, -0.19181,
        1.6235, -1.0207, -0.85768, -0.070449, -0.24873, -0.03845,
        -1.4997, -1.8168, -3.317, 0.1002, 0.11447, -0.10612,
        -4.8732, -0.068037, 0.40963, 0.0075472,
        -2.4983, -0.094302, -0.3705, -1.1418,
        -0.88993, 0.27303, -0.043412, 0.71954, -1.8407, -1.0274, -1.4252,
        -0.027254, 1.5437, -0.057158,
        -0.20629, 0.44353, 1.0319, 0.64409,
        0.031333, -0.78897, 1.4114, 0.013723, -0.044772, 0.20649, -0.16967,
        -0.25663, 2.1576, 0.65152, -0.24488, -0.20986, 0.28487, -0.19901,
        -0.94039, -0.14984, 0.14135, -0.73809, 0.064619,
        -0.25792, -1.379, -0.95362, -0.22068,
        -0.20175, -0.88331, -0.068636, -2.2026, -0.91037, -2.0125,
        0.70478, 0.76758, 1.3372, 0.82844,
        0.54535, -0.48562, 0.17287, 0.34971, 0.37556, 0.54014,
        -0.67585, -0.60473, 0.84289, 0.20393, -0.065075, -.0031822, 0.14265,
        -0.19159, 2.1777, 0.5672, -0.20762, 0.18791, -0.055522,
        0.55495, 1.845, 0.44644, 0.74369, 0.095801, -0.77925, 0.082186,
        -0.094907, -0.20826, 0.035136, 0.48206, -0.63384, 0.74443,
        -0.14573, 0.28835, 0.051439,
        -1.076, 1.7241, -0.17785, -0.089115, -0.15234, -0.099394,
        0.0836, 0.4011, -0.35263, -1.0689,
        0.0556, -2.204, -0.069701,
        -2.0996, -0.68189, -0.26701,
        -0.85001, -0.2806,
        0.85883, 0.89934, -0.6901,
        -0.54217, -0.38159,
        -0.37684, -2.0932, -0.34047,
        -0.83594, 1.1011, 0.015254],
    "MoSe2": [
        .33174, -5.986, -5.5593, -8.2313, 0.96391, 0.25063, 0.99113,
        .1669, -6.5023, -5.3143, -7.11, -0.17234, -0.47342, -.0036484,
        .0572, -0.21128, -0.26353, -0.21673,
        1.3956, -1.0901, -0.74231, -0.06804, -0.20124, -0.039312,
        -1.4401, -2.0234, -3.3155, 0.075711, 0.096034, -0.12147,
        -4.5471, -0.073552, 0.352, 0.0047322,
        -2.3406, -0.12096, -0.29601, -1.1458,
        -0.81024, 0.26983, .0036307, 0.82858, -1.9794, -0.95087, -1.5863,
        -0.10266, 1.3334, -0.072352,
        -0.14568, 0.53563, 0.88478, 0.6677,
        0.017446, -0.69479, 1.2677, 0.017186, -0.058501, 0.19472, -0.16233,
        -0.19132, 1.9412, 0.55439, -0.21522, -0.1229, 0.23643, -0.20208,
        -0.87396, -0.15533, 0.14159, -0.59642, 0.049521,
        -0.30909, -1.3258, -0.85773, -0.22287,
        -0.12516, -0.77169, -0.069137, -2.3779, -0.79261, -2.1798,
        0.51424, 0.82738, 1.108, 0.88379,
        0.40837, -0.40745, 0.17505, 0.44468, 0.33323, 0.57593,
        -0.58761, -0.41742, 0.82458, 0.18475, -0.015595, 0.0075386, 0.15501,
        -0.11752, 1.9279, 0.55433, -0.14566, 0.12623, -0.02589,
        0.4162, 1.7178, 0.27171, 0.76039, 0.11244, -0.66685, 0.073023,
        -0.063236, -0.29752, 0.0622, 0.56717, -0.56452, 0.77725,
        -0.1282, 0.25457, 0.065941,
        -0.89743, 1.53, -0.16413, -0.0921, -0.10965, -0.12727,
        0.0836, 0.2643, -0.36697, -0.995,
        0.247, -1.9948, -0.092776,
        -1.8736, -0.50984, -0.29247,
        -0.72697, -0.29025,
        0.7701, 0.76051, -0.6641,
        -0.47494, -0.39124,
        -0.46918, -1.8414, -0.29868,
        -0.71679, 1.0053, 0.0074413, ],
    "WS2": [
        .31817, -6.8383, -5.7338, -9.0778, 0.8731, 0.35504, 0.9654,
        .1574, -7.2503, -5.4979, -8.033, -0.21755, -0.69109, 0.014114,
        .056, -0.17498, -0.37102, -0.21221,
        1.7428, -1.2121, 0.15842, -0.098954, -0.30434, -0.1005,
        -1.8542, -1.9156, -4.2899, 0.10965, 0.14518, -0.16335,
        -4.3265, -0.082019, 0.48759, -0.030772,
        -2.6306, 0.089036, -0.29223, -1.3896,
        -0.98583, 0.48703, 0.035947, 1.5863, -1.8438, -1.2324, -1.1218,
        -0.066617, 1.9471, -0.16221,
        -0.19759, 0.43421, 1.1234, 0.67397,
        0.026736, -0.8836, 1.5578, 0.01038, -0.041548, 0.46215, -0.3136,
        -0.31038, 2.302, 0.66419, -0.27262, -0.20778, 0.36458, -0.33326,
        -0.99291, -0.15356, 0.17734, -0.65371, 0.10512,
        -0.45268, -1.4364, -0.9433, -0.2645,
        -0.21323, -1.005, -0.06591, -2.254, -1.0683, -1.9197,
        0.83394, 0.77232, 1.2403, 1.0392,
        0.58456, -0.60878, 0.53727, 0.28316, 0.52173, 0.58035,
        -0.94205, -0.48216, 1.045, 0.18534, -0.053906, -0.083318, 0.34503,
        -0.17522, 2.8266, 0.623, -0.19763, 0.17904, 0.061744,
        0.6489, 1.8259, 0.071026, 1.0553, 0.12652, -0.86295, 0.12968,
        -0.075808, -0.24136, -0.090131, 0.46731, -0.96046, 0.85835,
        -0.12828, 0.48383, 0.14605,
        -1.1279, 2.4018, -0.34521, -0.11682, -0.046287, -0.23598,
        0.2874, 0.13987, -0.89981, -1.1101,
        0.0556, -2.2935, -0.12455,
        -1.9903, -0.30593, -0.12035,
        -1.1843, -0.5356,
        0.9155, 0.90223, -1.0934,
        -0.19305, -0.64407,
        -0.63383, -2.9341, -0.53457,
        -0.9443, 1.4265, -0.12748],
    "WSe2": [
        .33155, -6.0659, -5.2666, -8.4665, 0.97661, 0.31985, 1.0466,
        .168, -6.4941, -5.0013, -7.2774, -0.19786, -0.58379, 0.0025289,
        .0611, -0.21738, -0.33319, -0.24107,
        1.3854, -1.0117, -0.050292, -0.092069, -0.24506, -0.10196,
        -1.7239, -1.9667, -4.1385, 0.079136, 0.1243, -0.18543,
        -4.0689, -0.091306, 0.42262, -0.037656,
        -2.3566, 0.059246, -0.2204, -1.3371,
        -0.90174, 0.48213, -0.021949, 1.5068, -1.9862, -1.127, -1.3572,
        -0.15181, 1.6171, -0.1591,
        -0.13728, 0.55743, 1.0132, 0.71831,
        0.013226, -0.77303, 1.3989, 0.017183, -0.074093, 0.32502, -0.30273,
        -0.23188, 2.0793, 0.56732, -0.24178, -0.10534, 0.29133, -0.28703,
        -0.90541, -0.16134, 0.18772, -0.56448, 0.1124,
        -0.49023, -1.4007, -0.85264, -0.26338,
        -0.1174, -0.89571, -0.067849, -2.4267, -0.96639, -2.0857,
        0.58919, 0.83376, 1.1792, 1.0687,
        0.40619, -0.49269, 0.4678, 0.40082, 0.40616, 0.55627,
        -0.80917, -0.32193, 0.91675, 0.20247, 0.014994, -0.043899, 0.33131,
        -0.090072, 2.4087, 0.65337, -0.10402, 0.12859, 0.062649,
        0.48006, 1.7639, 0.022246, 1.0503, 0.15216, -0.72746, 0.11202,
        -0.036918, -0.23759, -0.021196, 0.55035, -0.77559, 0.87344,
        -0.1573, 0.30778, 0.10862,
        -0.929, 1.9729, -0.32117, -0.12899, -0.098508, -0.22441,
        0.2874, -0.029111, -0.87699, -1.094,
        0.247, -2.153, -0.11444,
        -1.8795, -0.27627, -0.24136,
        -0.8969, -0.4763,
        0.7976, 0.76079, -1.0222,
        -0.30013, -0.65121,
        -0.69028, -2.4467, -0.42347,
        -0.79342, 1.082, -0.058411]
}

_bert_params = {"MoS2": [ 0.3182, -1.179 , -0.023 , -2.964 ,  1.06  ,  0.309 ,  0.878 ,
        0.1564, -0.55  ,  0.209 , -1.79  , -0.533 , -0.571 ,  0.003 ,
        0.0517, -0.101 , -0.193 , -0.25  ,  1.6235, -1.0207, -0.8577,
       -0.288 , -0.366 , -0.028 , -1.4997, -1.8168, -3.317 , -0.101 ,
       -0.393 , -0.054 ,  0.618 , -0.002 ,  0.504 ,  0.011 , -2.4983,
       -0.0943, -0.3705, -1.1418, -0.8899,  0.273 , -0.0434,  0.7195,
       -1.8407, -1.0274, -1.4252, -0.0273,  1.5437, -0.0572, -0.451 ,
        0.4435,  1.0319,  0.6441,  0.367 , -0.652 ,  1.548 ,  0.    ,
       -0.0448,  0.2065, -0.1697, -0.587 ,  2.113 ,  0.332 , -0.068 ,
       -0.2099,  0.2849, -0.199 , -1.009 , -0.147 ,  0.1414, -0.7381,
        0.0646, -0.2579, -1.259 , -0.898 , -0.379 , -0.2018, -0.8   ,
       -0.005 , -2.2026, -0.9104, -2.0125,  0.7048,  0.7676,  1.3372,
        0.8284,  0.5454, -0.4856,  0.1729,  0.3497,  0.3756,  0.5401,
       -0.6758, -0.6047,  0.8429,  0.2039, -0.0651, -0.0032,  0.1426,
       -0.1916,  2.1777,  0.5672, -0.2076,  0.1879, -0.0555,  0.555 ,
        1.845 ,  0.4464,  0.7437,  0.0958, -0.7792,  0.0822, -0.0949,
       -0.2083,  0.0351,  0.4821, -0.6338,  0.7444, -0.1457,  0.2883,
        0.0514, -1.076 ,  1.7241, -0.1779, -0.0891, -0.1523, -0.0994,
        0.0836,  0.4011, -0.3526, -1.0689,  0.0556, -2.204 , -0.0697,
       -2.0996, -0.6819, -0.267 , -0.85  , -0.2806,  0.8588,  0.8993,
       -0.6901, -0.5422, -0.3816, -0.3768, -2.0932, -0.3405, -0.8359,
        1.1011,  0.0153]
}

class Group4Tmd11BandStrain:
    r"""Monolayer of a group 4 TMD using the second nearest-neighbor 11-band model

        Parameters
        ----------
        name : str
            Name of the TMD to model. The available options are: MoS2, WS2, MoSe2,
            WSe2. The relevant tight-binding parameters for these
            materials are given by https://link.aps.org/doi/10.1103/PhysRevB.92.205108
        override_params : Optional[dict]
            Replace or add new material parameters. The dictionary entries must
            be in the format
                `"name": [eps1,    eps3,    eps4,    eps6,    eps7,    eps9,    eps10,    t1_1_1,   t1_2_2,  t1_3_3,
                          t1_4_4,  t1_5_5,  t1_6_6,  t1_7_7,  t1_8_8,  t1_9_9,  t1_10_10, t1_11_11, t1_3_5,  t1_6_8,
                          t1_9_11, t1_1_2,  t1_3_4,  t1_4_5,  t1_6_7,  t1_7_8,  t1_9_10,  t1_10_11, t5_4_1,  t5_3_2,
                          t5_5_2,  t5_9_6,  t5_11_6, t5_10_7, t5_9_8,  t5_11_8, t6_9_6,   t6_11_6,  t6_9_8,  t6_11_8,
                          a,       c,       dcc,     dcM,     lambM,   lambc]`.

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
        self.params = _default_11band_strain_params
        self._rt3 = np.sqrt(3)
        self.soc = False
        self.soc_eo_flip = False
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
        [a, eps_0_bb, eps_0_cc, eps_0_dd, t_0_2_bb, t_0_2_cc, t_0_2_dd,
         d_0, eps_1_bb, eps_1_cc, eps_1_dd, t_1_2_bb, t_1_2_cc, t_1_2_dd,
         d_1, t_2_2_bb, t_2_2_cc, t_2_2_dd,
         a_0_0_bb, a_0_0_cc, a_0_0_dd, t_3_2_bb, t_3_2_cc, t_3_2_dd,
         a_1_0_bb, a_1_0_cc, a_1_0_dd, t_4_2_bb, t_4_2_cc, t_4_2_dd,
         eps_1_aa, t_5_2_bb, t_5_2_cc, t_5_2_dd,
         a_1_0_aa, b_0_0_bb, b_0_0_cc, b_0_0_dd,
         b_0_0_aa, b_1_0_bb, b_1_0_cc, b_1_0_dd, a_0_2_bb, a_0_2_cc, a_0_2_dd,
         a_1_2_bb, a_1_2_cc, a_1_2_dd,
         t_0_2_aa, a_2_2_bb, a_2_2_cc, a_2_2_dd,
         t_1_2_aa, t_0_1_ba, t_0_1_dc, t_0_3_dc, a_3_2_bb, a_3_2_cc, a_3_2_dd,
         t_3_2_aa, t_1_1_ba, t_1_1_dc, t_1_3_dc, a_4_2_bb, a_4_2_cc, a_4_2_dd,
         t_2_1_dc, t_2_3_dc, a_5_2_bb, a_5_2_cc, a_5_2_dd,
         a_0_2_aa, t_3_1_ba, t_3_1_dc, t_3_3_dc,
         a_1_2_aa, t_4_1_dc, t_4_3_dc, b_0_2_bb, b_0_2_cc, b_0_2_dd,
         a_3_2_aa, b_1_2_bb, b_1_2_cc, b_1_2_dd,
         a_0_1_ba, a_0_1_dc, a_0_3_dc, b_2_2_bb, b_2_2_cc, b_2_2_dd,
         b_0_2_aa, a_1_1_ba, a_1_1_dc, a_1_3_dc, b_3_2_bb, b_3_2_cc, b_3_2_dd,
         b_1_2_aa, a_2_1_dc, a_2_3_dc, b_4_2_bb, b_4_2_cc, b_4_2_dd,
         b_3_2_aa, a_3_1_ba, a_3_1_dc, a_3_3_dc, b_5_2_bb, b_5_2_cc, b_5_2_dd,
         b_6_2_aa, a_4_1_dc, a_4_3_dc, b_6_2_bb, b_6_2_cc, b_6_2_dd,
         b_7_2_bb, b_7_2_cc, b_7_2_dd,
         b_0_1_ba, b_0_1_dc, b_0_3_dc, b_8_2_bb, b_8_2_cc, b_8_2_dd,
         lamb_m, b_1_1_ba, b_1_1_dc, b_1_3_dc,
         lamb_c, b_2_1_dc, b_2_3_dc,
         b_3_1_ba, b_3_1_dc, b_3_3_dc,
         b_4_1_dc, b_4_3_dc,
         b_5_1_ba, b_5_1_dc, b_5_3_dc,
         b_6_1_dc, b_6_3_dc,
         b_7_1_ba, b_7_1_dc, b_7_3_dc,
         b_8_1_ba, b_8_1_dc, b_8_3_dc
         ] = self.params[self.name]

        # make functions for various matrices
        def matrix_0(e_0, e_1, a_0_0, a_1_0, b_0_0, b_1_0):
            return np.array([np.diag([e_1, e_1, e_0]),
                             np.diag([a_1_0, a_1_0, a_0_0]),
                             [[b_0_0, 0.000, 0.000], [0.000, -b_0_0, b_1_0], [0.000, b_1_0, 0.000]],
                             [[0.000, b_0_0, b_1_0], [b_0_0, 0.000, 0.000], [b_1_0, 0.000, 0.000]]])

        def matrix_n(t_0_n, t_1_n, t_2_n, t_3_n, t_4_n,
                     a_0_n, a_1_n, a_2_n, a_3_n, a_4_n,
                     b_0_n, b_1_n, b_2_n, b_3_n, b_4_n,
                     b_5_n, b_6_n, b_7_n, b_8_n):
            return np.array([[[t_0_n, 0, 0], [0, t_1_n, t_2_n], [0, t_3_n, t_4_n]],
                             [[a_0_n, 0, 0], [0, a_1_n, a_2_n], [0, a_3_n, a_4_n]],
                             [[b_0_n, 0, 0], [0, b_1_n, b_2_n], [0, b_3_n, b_4_n]],
                             [[0, b_5_n, b_6_n], [b_7_n, 0, 0], [b_8_n, 0, 0]]])

        def matrix_2(t_0_2, t_1_2, t_2_2, t_3_2, t_4_2, t_5_2,
                     a_0_2, a_1_2, a_2_2, a_3_2, a_4_2, a_5_2,
                     b_0_2, b_1_2, b_2_2, b_3_2, b_4_2,
                     b_5_2, b_6_2, b_7_2, b_8_2):
            return np.array([[[t_0_2, t_3_2, t_4_2], [-t_3_2, t_1_2, t_5_2], [-t_4_2, t_5_2, t_2_2]],
                             [[a_0_2, a_3_2, a_4_2], [-a_3_2, a_1_2, a_5_2], [-a_4_2, a_5_2, a_2_2]],
                             [[b_0_2, b_3_2, b_4_2], [-b_3_2, b_1_2, b_5_2], [-b_4_2, b_5_2, b_2_2]],
                             [[0, b_6_2, b_7_2], [b_6_2, 0, b_8_2], [b_7_2, -b_8_2, 0]]])

        # make the matrices
        h_0_aa = matrix_0(0.000000, eps_1_aa, 0.000000, a_1_0_aa, b_0_0_aa, 0.000000)[:, :2, :2]
        h_0_bb = matrix_0(eps_0_bb, eps_1_bb, a_0_0_bb, a_1_0_bb, b_0_0_bb, b_1_0_bb)
        h_0_cc = matrix_0(eps_0_cc, eps_1_cc, a_0_0_cc, a_1_0_cc, b_0_0_cc, b_1_0_cc)
        h_0_dd = matrix_0(eps_0_dd, eps_1_dd, a_0_0_dd, a_1_0_dd, b_0_0_dd, b_1_0_dd)
        h_1_ba = matrix_n(t_0_1_ba, t_1_1_ba, 0.000000, t_3_1_ba, 0.000000,
                          a_0_1_ba, a_1_1_ba, 0.000000, a_3_1_ba, 0.000000,
                          b_0_1_ba, b_1_1_ba, 0.000000, b_3_1_ba, 0.000000,
                          b_5_1_ba, 0.000000, b_7_1_ba, b_8_1_ba)[:, :, :2]
        h_1_dc = matrix_n(t_0_1_dc, t_1_1_dc, t_2_1_dc, t_3_1_dc, t_4_1_dc,
                          a_0_1_dc, a_1_1_dc, a_2_1_dc, a_3_1_dc, a_4_1_dc,
                          b_0_1_dc, b_1_1_dc, b_2_1_dc, b_3_1_dc, b_4_1_dc,
                          b_5_1_dc, b_6_1_dc, b_7_1_dc, b_8_1_dc)
        h_2_aa = matrix_2(t_0_2_aa, t_1_2_aa, 0.000000, t_3_2_aa, 0.000000, 0.000000,
                          a_0_2_aa, a_1_2_aa, 0.000000, a_3_2_aa, 0.000000, 0.000000,
                          b_0_2_aa, b_1_2_aa, 0.000000, b_3_2_aa, 0.000000,
                          0.000000, b_6_2_aa, 0.000000, 0.000000)[:, :2, :2]
        h_2_bb = matrix_2(t_0_2_bb, t_1_2_bb, t_2_2_bb, t_3_2_bb, t_4_2_bb, t_5_2_bb,
                          a_0_2_bb, a_1_2_bb, a_2_2_bb, a_3_2_bb, a_4_2_bb, a_5_2_bb,
                          b_0_2_bb, b_1_2_bb, b_2_2_bb, b_3_2_bb, b_4_2_bb,
                          b_5_2_bb, b_6_2_bb, b_7_2_bb, b_8_2_bb)
        h_2_cc = matrix_2(t_0_2_cc, t_1_2_cc, t_2_2_cc, t_3_2_cc, t_4_2_cc, t_5_2_cc,
                          a_0_2_cc, a_1_2_cc, a_2_2_cc, a_3_2_cc, a_4_2_cc, a_5_2_cc,
                          b_0_2_cc, b_1_2_cc, b_2_2_cc, b_3_2_cc, b_4_2_cc,
                          b_5_2_cc, b_6_2_cc, b_7_2_cc, b_8_2_cc)
        h_2_dd = matrix_2(t_0_2_dd, t_1_2_dd, t_2_2_dd, t_3_2_dd, t_4_2_dd, t_5_2_dd,
                          a_0_2_dd, a_1_2_dd, a_2_2_dd, a_3_2_dd, a_4_2_dd, a_5_2_dd,
                          b_0_2_dd, b_1_2_dd, b_2_2_dd, b_3_2_dd, b_4_2_dd,
                          b_5_2_dd, b_6_2_dd, b_7_2_dd, b_8_2_dd)
        h_3_dc = matrix_n(t_0_3_dc, t_1_3_dc, t_2_3_dc, t_3_3_dc, t_4_3_dc,
                          a_0_3_dc, a_1_3_dc, a_2_3_dc, a_3_3_dc, a_4_3_dc,
                          b_0_3_dc, b_1_3_dc, b_2_3_dc, b_3_3_dc, b_4_3_dc,
                          b_5_3_dc, b_6_3_dc, b_7_3_dc, b_8_3_dc)
        keys = ["h_0_aa", "h_0_bb", "h_0_cc", "h_0_dd", "h_1_ba", "h_1_dc", "h_2_aa", "h_2_bb", "h_2_cc", "h_2_dd",
                "h_3_dc", "a", "d_0", "d_1", "lamb_m", "lamb_c"]
        values = [h_0_aa, h_0_bb, h_0_cc, h_0_dd, h_1_ba, h_1_dc, h_2_aa, h_2_bb, h_2_cc, h_2_dd, h_3_dc,
                  a, d_0, d_1, lamb_m, lamb_c]
        [setattr(self, key, value) for key, value in zip(keys, values)]

    def ham(self, h, ur_l, ur_r):
        return h, ur_l.T.dot(h.dot(ur_r)), ur_l.dot(h.dot(ur_r.T))

    def lattice(self):
        lat = pb.Lattice(a1=self.a1, a2=self.a2)

        metal_name, chalcogenide_name = re.findall("[A-Z][a-z]*", self.name)

        def h_0_m_sz(sz):
            s_part_aa = sz * self.lamb_m * 1j * np.array([[0, -1 / 2], [1 / 2, 0]])
            s_part_cc = -sz * self.lamb_m * 1j * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
            return block_diag(self.h_0_aa[0] + s_part_aa, self.h_0_cc[0] + s_part_cc)

        def h_0_c_sz(sz):
            s_part_bb = sz * self.lamb_c * 1j * np.array([[0, -1 / 2, 0], [1 / 2, 0, 0], [0, 0, 0]])
            return block_diag(self.h_0_bb[0] + s_part_bb, self.h_0_dd[0] + s_part_bb)

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
                        [[0, 0, 1 / 2],
                         [0, 0, -1j / 2],
                         [-1 / 2, 1j / 2, 0]])
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
        ur_m = block_diag(self._ur[:2, :2], self._ur)
        ur_c = block_diag(self._ur, self._ur)

        (h_1_m_1, h_1_m_2, h_1_m_3) = self.ham(block_diag(self.h_1_ba[0], self.h_1_dc[0]), ur_c, ur_m)
        (h_2_m_1, h_2_m_2, h_2_m_3) = self.ham(block_diag(self.h_2_aa[0], self.h_2_cc[0]), ur_m, ur_m)
        (h_2_c_1, h_2_c_2, h_2_c_3) = self.ham(block_diag(self.h_2_bb[0], self.h_2_dd[0]), ur_c, ur_c)
        (h_3_m_1, h_3_m_2, h_3_m_3) = self.ham(block_diag(np.zeros([3, 2]), self.h_3_dc[0]), ur_c, ur_m)

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

    def _stitching(self, u_local, h1, h2=None, shape=None, ur=None):
        if u_local.ndim == 1:
            if h2 is None:
                h2 = np.zeros([h1.shape[0], 0, 0])
            if ur is None:
                ur = np.eye(3)
            output = np.zeros([h1.shape[1] + h2.shape[1], h1.shape[2] + h2.shape[2]])
            output[:h1.shape[1], :h1.shape[2]] = np.dot(ur.T[:h1.shape[1], :h1.shape[1]],
                                                        np.dot(np.dot(h1[1:].T, u_local).T,
                                                               ur[:h1.shape[2], :h1.shape[2]]))
            output[h1.shape[1]:, h1.shape[2]:] = np.dot(ur.T[:h2.shape[1], :h2.shape[1]],
                                                        np.dot(np.dot(h2[1:].T, u_local).T,
                                                               ur[:h2.shape[2], :h2.shape[2]]))
            if self.soc_doubled_ham:
                output = np.kron(np.eye(2), output)
            return output
        else:
            if u_local.shape[1] == 0:
                return 0 if shape is None else np.zeros([0, shape[1], shape[2]])
            return np.array([self._stitching(u_i, h1, h2, shape, ur) for u_i in u_local.T])

    def straining_hopping(self, uxx, uyy, uxy, hop_id=None, shape=None):
        len_bool = True
        if not hasattr(uxx, "__len__"):
            uxx = np.array([uxx])
            uyy = np.array([uyy])
            uxy = np.array([uxy])
            len_bool = False
        rt3 = self._rt3
        u = np.array([uxx + uyy, uxx - uyy, 2 * uxy])
        ur_1 = np.array([[1, 0, 0], [0, -1 / 2, -rt3 / 2], [0, rt3 / 2, -1 / 2]])
        ur_2 = ur_1.dot(ur_1)
        ur = self._ur
        if len_bool:
            hopping = np.zeros(shape)
            if hop_id == 'h_1_m_1':
                hopping = self._stitching(u, self.h_1_ba, self.h_1_dc, shape)
            elif hop_id == 'h_1_m_2':
                hopping = self._stitching(ur_1.dot(u), self.h_1_ba, self.h_1_dc, shape, ur)
            elif hop_id == 'h_1_m_3':
                hopping = self._stitching(ur_2.dot(u), self.h_1_ba, self.h_1_dc, shape, ur.T)
            elif hop_id == 'h_2_m_1':
                hopping = self._stitching(u, self.h_2_aa, self.h_2_cc, shape)
            elif hop_id == 'h_2_m_2':
                hopping = self._stitching(ur_1.dot(u), self.h_2_aa, self.h_2_cc, shape, ur)
            elif hop_id == 'h_2_m_3':
                hopping = self._stitching(ur_2.dot(u), self.h_2_aa, self.h_2_cc, shape, ur.T)
            elif hop_id == 'h_2_c_1':
                hopping = self._stitching(u, self.h_2_bb, self.h_2_dd, shape)
            elif hop_id == 'h_2_c_2':
                hopping = self._stitching(ur_1.dot(u), self.h_2_bb, self.h_2_dd, shape, ur)
            elif hop_id == 'h_2_c_3':
                hopping = self._stitching(ur_2.dot(u), self.h_2_bb, self.h_2_dd, shape, ur.T)
            elif hop_id == 'h_3_m_1':
                hopping = self._stitching(u, np.zeros([4, 3, 2]), self.h_3_dc, shape)
            elif hop_id == 'h_3_m_2':
                hopping = self._stitching(ur_1.dot(u), np.zeros([4, 3, 2]), self.h_3_dc, shape, ur)
            elif hop_id == 'h_3_m_3':
                hopping = self._stitching(ur_2.dot(u), np.zeros([4, 3, 2]), self.h_3_dc, shape, ur.T)
            return hopping.swapaxes(1, 2)
        else:
            hopping = {
                'h_1_m_1': self.straining_hopping(uxx, uyy, uxy, 'h_1_m_1', (1, 6, 5)),
                'h_1_m_2': self.straining_hopping(uxx, uyy, uxy, 'h_1_m_2', (1, 6, 5)),
                'h_1_m_3': self.straining_hopping(uxx, uyy, uxy, 'h_1_m_3', (1, 6, 5)),
                'h_2_m_1': self.straining_hopping(uxx, uyy, uxy, 'h_2_m_1', (1, 5, 5)),
                'h_2_c_1': self.straining_hopping(uxx, uyy, uxy, 'h_2_c_1', (1, 6, 6)),
                'h_2_m_2': self.straining_hopping(uxx, uyy, uxy, 'h_2_m_2', (1, 5, 5)),
                'h_2_c_2': self.straining_hopping(uxx, uyy, uxy, 'h_2_c_2', (1, 6, 6)),
                'h_2_m_3': self.straining_hopping(uxx, uyy, uxy, 'h_2_m_3', (1, 5, 5)),
                'h_2_c_3': self.straining_hopping(uxx, uyy, uxy, 'h_2_c_3', (1, 6, 6)),
                'h_3_m_1': self.straining_hopping(uxx, uyy, uxy, 'h_3_m_1', (1, 6, 5)),
                'h_3_m_2': self.straining_hopping(uxx, uyy, uxy, 'h_3_m_2', (1, 6, 5)),
                'h_3_m_3': self.straining_hopping(uxx, uyy, uxy, 'h_3_m_3', (1, 6, 5))
            }
            return hopping if hop_id is None else hopping[hop_id]

    def strained_onsite(self, uxx, uyy, uxy, sub_id=None, shape=None):
        onsite = []
        u = np.array([uxx + uyy, uxx - uyy, 2 * uxy])
        metal_name, chalcogenide_name = re.findall("[A-Z][a-z]*", self.name)
        if hasattr(uxx, '__len__'):
            onsite = np.zeros(shape)
            metal_bool = np.array(sub_id == metal_name).flatten()
            chalcogenide_bool = np.array(sub_id == chalcogenide_name).flatten()
            onsite[metal_bool] = self._stitching(u.T[metal_bool].T, self.h_0_aa, self.h_0_cc, shape)
            onsite[chalcogenide_bool] = self._stitching(u.T[chalcogenide_bool].T, self.h_0_bb, self.h_0_dd, shape)
        else:
            if sub_id is None:
                onsite = {
                    metal_name: self.strained_onsite(uxx, uyy, uxy, metal_name),
                    chalcogenide_name: self.strained_onsite(uxx, uyy, uxy, chalcogenide_name)
                }
            elif sub_id == metal_name:
                onsite = block_diag(np.tensordot(self.h_0_aa[1:], u, (0, 0)), np.tensordot(self.h_0_cc[1:], u, (0, 0)))
            elif sub_id == chalcogenide_name:
                onsite = block_diag(np.tensordot(self.h_0_bb[1:], u, (0, 0)), np.tensordot(self.h_0_dd[1:], u, (0, 0)))
            if self.soc_doubled_ham and sub_id is not None:
                onsite = np.kron(np.eye(2), onsite)
        return onsite

    def uniform_strain(self, uxx, uyy, uxy):
        @pb.hopping_energy_modifier
        def strained_hoppings(energy, x1, y1, z1, x2, y2, z2, hop_id):
            return energy + self.straining_hopping(uxx, uyy, uxy, hop_id=hop_id, shape=np.shape(energy))

        @pb.onsite_energy_modifier
        def strained_onsite(energy, x, y, z, sub_id):
            return energy + self.strained_onsite(uxx, uyy, uxy, sub_id=sub_id, shape=np.shape(energy))

        return strained_hoppings, strained_onsite

    def construct_hamiltonian_fang_matlab(self, k=None, u=None):
        if k is None:
            k = np.array([0, 0])
        if u is None:
            u = np.array([0, 0, 0])
        self.lat4 = False
        model = pb.Model(self.lattice(),
                         self.uniform_strain(u[0], u[1], u[2]),
                         pb.translational_symmetry(True, True))
        model.set_wave_vector(k)
        hamiltonian = np.array(model.hamiltonian.todense()).T
        delta = group4_class.a * np.array([.5, np.sqrt(3) / 6])
        phase = np.exp(1j * k.dot(delta))
        if not self.soc_doubled_ham:
            hamiltonian[0:2, 5:8] *= phase
            hamiltonian[5:8, 0:2] *= np.conjugate(phase)
            hamiltonian[2:5, 8:11] *= phase
            hamiltonian[8:11, 2:5] *= np.conjugate(phase)
            transform_list = np.array([0, 1, 5, 6, 7, 2, 3, 4, 8, 9, 10])
            transform = np.zeros((11, 11))
            for i in range(11):
                transform[i, transform_list[i]] = 1
            hamiltonian = np.dot(transform, np.dot(hamiltonian, transform.T))
        else:
            hamiltonian[0:2, 10:13] *= phase
            hamiltonian[10:13, 0:2] *= np.conjugate(phase)
            hamiltonian[5:7, 16:19] *= phase
            hamiltonian[16:19, 5:7] *= np.conjugate(phase)
            hamiltonian[2:5, 13:16] *= phase
            hamiltonian[13:16, 2:5] *= np.conjugate(phase)
            hamiltonian[7:10, 19:22] *= phase
            hamiltonian[19:22, 7:10] *= np.conjugate(phase)
            transform_list = np.array([0, 1, 12, 10, 11, 4, 2, 3, 15, 13, 14, 5, 6, 18, 16, 17, 9, 7, 8, 21, 19, 20])
            transform = np.zeros((22, 22))
            for i in range(22):
                transform[i, transform_list[i]] = 1
            hamiltonian = np.dot(transform, np.dot(hamiltonian, transform.T))
        return hamiltonian

    def test1(self):
        # plot all
        print("Doing every calculation with high precision,\n this can take a minute.")

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
        #plt.figure()
        # plot bands comparing lat4=False and lat4=True with a certain strain vector, giving the largest error between
        # the two models
        if u is None:
            u = np.array([1, .34, -.42])
        bands = []
        for lat4 in [False]:
            self.lat4 = lat4
            self.soc = soc
            model = pb.Model(self.lattice(),
                             self.uniform_strain(u[0], u[1], u[2]),
                             pb.translational_symmetry())
            solver = pb.solver.lapack(model)
            if not lat4:
                k_points = model.lattice.brillouin_zone()
                gamma = [0, 0]
                k = k_points[0]
                m = (k_points[0] + k_points[1]) / 2
            bands.append(solver.calc_bands(gamma, k, m, gamma, step=0.05))
        bands[0].plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"], lw=1.5, color="C1", ls=":")
        #bands[1].plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"], lw=1.5, color="C1", ls=":")
        #error = np.array(
        #    [[np.abs(bands[1].energy[i] - bands[0].energy[i, j]).min()
        ##      for j in range(bands[0].energy.shape[1])] for i in
        #     range(bands[0].energy.shape[0])])
        # class.test2()[0] for maximal error
        #return error.max(), error, bands


class TestClass(AbstractLattice):
    def __init__(self, **kwargs):

        lattice_orbital_dict = {"l": {"Mo": [-1, 1, 2, -2, 0], "S": [-1, 1, 0, -1, 1, 0]},
                                "orbs": {"Mo": ["dxz", "dyz", "dxy", "dx2y2", "dz2"],
                                         "S": ["pxo", "pyo", "pzo", "pxe", "pye", "pze"]},
                                "group": {"Mo": [1, 1, 2, 2, 0], "S": [0, 0, 1, 2, 2, 3]}}
        super().__init__(orbital=lattice_orbital_dict, n_v=6, n_b=11)
        self.single_orbital = False
        self.strain_params = {}
        self.params = _default_11band_strain_params
        self._berry_phase_factor = -1
        [setattr(self, var, kwargs[var]) for var in [*kwargs]]

    def _generate_matrices(self):
        # generate the matrices from Fang
        # import data
        [a, eps_0_bb, eps_0_cc, eps_0_dd, t_0_2_bb, t_0_2_cc, t_0_2_dd,
         d_0, eps_1_bb, eps_1_cc, eps_1_dd, t_1_2_bb, t_1_2_cc, t_1_2_dd,
         d_1, t_2_2_bb, t_2_2_cc, t_2_2_dd,
         a_0_0_bb, a_0_0_cc, a_0_0_dd, t_3_2_bb, t_3_2_cc, t_3_2_dd,
         a_1_0_bb, a_1_0_cc, a_1_0_dd, t_4_2_bb, t_4_2_cc, t_4_2_dd,
         eps_1_aa, t_5_2_bb, t_5_2_cc, t_5_2_dd,
         a_1_0_aa, b_0_0_bb, b_0_0_cc, b_0_0_dd,
         b_0_0_aa, b_1_0_bb, b_1_0_cc, b_1_0_dd, a_0_2_bb, a_0_2_cc, a_0_2_dd,
         a_1_2_bb, a_1_2_cc, a_1_2_dd,
         t_0_2_aa, a_2_2_bb, a_2_2_cc, a_2_2_dd,
         t_1_2_aa, t_0_1_ba, t_0_1_dc, t_0_3_dc, a_3_2_bb, a_3_2_cc, a_3_2_dd,
         t_3_2_aa, t_1_1_ba, t_1_1_dc, t_1_3_dc, a_4_2_bb, a_4_2_cc, a_4_2_dd,
         t_2_1_dc, t_2_3_dc, a_5_2_bb, a_5_2_cc, a_5_2_dd,
         a_0_2_aa, t_3_1_ba, t_3_1_dc, t_3_3_dc,
         a_1_2_aa, t_4_1_dc, t_4_3_dc, b_0_2_bb, b_0_2_cc, b_0_2_dd,
         a_3_2_aa, b_1_2_bb, b_1_2_cc, b_1_2_dd,
         a_0_1_ba, a_0_1_dc, a_0_3_dc, b_2_2_bb, b_2_2_cc, b_2_2_dd,
         b_0_2_aa, a_1_1_ba, a_1_1_dc, a_1_3_dc, b_3_2_bb, b_3_2_cc, b_3_2_dd,
         b_1_2_aa, a_2_1_dc, a_2_3_dc, b_4_2_bb, b_4_2_cc, b_4_2_dd,
         b_3_2_aa, a_3_1_ba, a_3_1_dc, a_3_3_dc, b_5_2_bb, b_5_2_cc, b_5_2_dd,
         b_6_2_aa, a_4_1_dc, a_4_3_dc, b_6_2_bb, b_6_2_cc, b_6_2_dd,
         b_7_2_bb, b_7_2_cc, b_7_2_dd,
         b_0_1_ba, b_0_1_dc, b_0_3_dc, b_8_2_bb, b_8_2_cc, b_8_2_dd,
         lamb_m, b_1_1_ba, b_1_1_dc, b_1_3_dc,
         lamb_c, b_2_1_dc, b_2_3_dc,
         b_3_1_ba, b_3_1_dc, b_3_3_dc,
         b_4_1_dc, b_4_3_dc,
         b_5_1_ba, b_5_1_dc, b_5_3_dc,
         b_6_1_dc, b_6_3_dc,
         b_7_1_ba, b_7_1_dc, b_7_3_dc,
         b_8_1_ba, b_8_1_dc, b_8_3_dc
         ] = self.params[self.name]

        # make functions for various matrices
        def matrix_0(e_0, e_1, a_0_0, a_1_0, b_0_0, b_1_0):
            return np.array([np.diag([e_1, e_1, e_0]),
                             np.diag([a_1_0, a_1_0, a_0_0]),
                             [[b_0_0, 0.000, 0.000], [0.000, -b_0_0, b_1_0], [0.000, b_1_0, 0.000]],
                             [[0.000, b_0_0, b_1_0], [b_0_0, 0.000, 0.000], [b_1_0, 0.000, 0.000]]])

        def matrix_n(t_0_n, t_1_n, t_2_n, t_3_n, t_4_n,
                     a_0_n, a_1_n, a_2_n, a_3_n, a_4_n,
                     b_0_n, b_1_n, b_2_n, b_3_n, b_4_n,
                     b_5_n, b_6_n, b_7_n, b_8_n):
            return np.array([[[t_0_n, 0, 0], [0, t_1_n, t_2_n], [0, t_3_n, t_4_n]],
                             [[a_0_n, 0, 0], [0, a_1_n, a_2_n], [0, a_3_n, a_4_n]],
                             [[b_0_n, 0, 0], [0, b_1_n, b_2_n], [0, b_3_n, b_4_n]],
                             [[0, b_5_n, b_6_n], [b_7_n, 0, 0], [b_8_n, 0, 0]]])

        def matrix_2(t_0_2, t_1_2, t_2_2, t_3_2, t_4_2, t_5_2,
                     a_0_2, a_1_2, a_2_2, a_3_2, a_4_2, a_5_2,
                     b_0_2, b_1_2, b_2_2, b_3_2, b_4_2,
                     b_5_2, b_6_2, b_7_2, b_8_2):
            return np.array([[[t_0_2, t_3_2, t_4_2], [-t_3_2, t_1_2, t_5_2], [-t_4_2, t_5_2, t_2_2]],
                             [[a_0_2, a_3_2, a_4_2], [-a_3_2, a_1_2, a_5_2], [-a_4_2, a_5_2, a_2_2]],
                             [[b_0_2, b_3_2, b_4_2], [-b_3_2, b_1_2, b_5_2], [-b_4_2, b_5_2, b_2_2]],
                             [[0, b_6_2, b_7_2], [b_6_2, 0, b_8_2], [b_7_2, -b_8_2, 0]]])

        # make the matrices
        h_0_aa = matrix_0(0.000000, eps_1_aa, 0.000000, a_1_0_aa, b_0_0_aa, 0.000000)[:, :2, :2]
        h_0_bb = matrix_0(eps_0_bb, eps_1_bb, a_0_0_bb, a_1_0_bb, b_0_0_bb, b_1_0_bb)
        h_0_cc = matrix_0(eps_0_cc, eps_1_cc, a_0_0_cc, a_1_0_cc, b_0_0_cc, b_1_0_cc)
        h_0_dd = matrix_0(eps_0_dd, eps_1_dd, a_0_0_dd, a_1_0_dd, b_0_0_dd, b_1_0_dd)
        h_1_ba = matrix_n(t_0_1_ba, t_1_1_ba, 0.000000, t_3_1_ba, 0.000000,
                          a_0_1_ba, a_1_1_ba, 0.000000, a_3_1_ba, 0.000000,
                          b_0_1_ba, b_1_1_ba, 0.000000, b_3_1_ba, 0.000000,
                          b_5_1_ba, 0.000000, b_7_1_ba, b_8_1_ba)[:, :, :2]
        h_1_dc = matrix_n(t_0_1_dc, t_1_1_dc, t_2_1_dc, t_3_1_dc, t_4_1_dc,
                          a_0_1_dc, a_1_1_dc, a_2_1_dc, a_3_1_dc, a_4_1_dc,
                          b_0_1_dc, b_1_1_dc, b_2_1_dc, b_3_1_dc, b_4_1_dc,
                          b_5_1_dc, b_6_1_dc, b_7_1_dc, b_8_1_dc)
        h_2_aa = matrix_2(t_0_2_aa, t_1_2_aa, 0.000000, t_3_2_aa, 0.000000, 0.000000,
                          a_0_2_aa, a_1_2_aa, 0.000000, a_3_2_aa, 0.000000, 0.000000,
                          b_0_2_aa, b_1_2_aa, 0.000000, b_3_2_aa, 0.000000,
                          0.000000, b_6_2_aa, 0.000000, 0.000000)[:, :2, :2]
        h_2_bb = matrix_2(t_0_2_bb, t_1_2_bb, t_2_2_bb, t_3_2_bb, t_4_2_bb, t_5_2_bb,
                          a_0_2_bb, a_1_2_bb, a_2_2_bb, a_3_2_bb, a_4_2_bb, a_5_2_bb,
                          b_0_2_bb, b_1_2_bb, b_2_2_bb, b_3_2_bb, b_4_2_bb,
                          b_5_2_bb, b_6_2_bb, b_7_2_bb, b_8_2_bb)
        h_2_cc = matrix_2(t_0_2_cc, t_1_2_cc, t_2_2_cc, t_3_2_cc, t_4_2_cc, t_5_2_cc,
                          a_0_2_cc, a_1_2_cc, a_2_2_cc, a_3_2_cc, a_4_2_cc, a_5_2_cc,
                          b_0_2_cc, b_1_2_cc, b_2_2_cc, b_3_2_cc, b_4_2_cc,
                          b_5_2_cc, b_6_2_cc, b_7_2_cc, b_8_2_cc)
        h_2_dd = matrix_2(t_0_2_dd, t_1_2_dd, t_2_2_dd, t_3_2_dd, t_4_2_dd, t_5_2_dd,
                          a_0_2_dd, a_1_2_dd, a_2_2_dd, a_3_2_dd, a_4_2_dd, a_5_2_dd,
                          b_0_2_dd, b_1_2_dd, b_2_2_dd, b_3_2_dd, b_4_2_dd,
                          b_5_2_dd, b_6_2_dd, b_7_2_dd, b_8_2_dd)
        h_3_dc = matrix_n(t_0_3_dc, t_1_3_dc, t_2_3_dc, t_3_3_dc, t_4_3_dc,
                          a_0_3_dc, a_1_3_dc, a_2_3_dc, a_3_3_dc, a_4_3_dc,
                          b_0_3_dc, b_1_3_dc, b_2_3_dc, b_3_3_dc, b_4_3_dc,
                          b_5_3_dc, b_6_3_dc, b_7_3_dc, b_8_3_dc)
        h_0_m = self.block_diag(h_0_aa, h_0_cc)
        h_0_c = self.block_diag(h_0_bb, h_0_dd)
        h_1_m = self.block_diag(h_1_ba, h_1_dc)
        h_2_m = self.block_diag(h_2_aa, h_2_cc)
        h_2_c = self.block_diag(h_2_bb, h_2_dd)
        h_3_m = self.block_diag(0 * h_1_ba, h_3_dc)
        keys_strain = ["h_0_m", "h_0_c", "h_1_m", "h_2_m", "h_2_c", "h_3_m", "a", "d_0", "d_1", "lamb_m", "lamb_c"]
        values_strain = [h_0_m, h_0_c, h_1_m, h_2_m, h_2_c, h_3_m, a, d_0, d_1, lamb_m, lamb_c]
        self.strain_params = dict([(key, value) for key, value in zip(keys_strain, values_strain)])
        keys = ["h_0_m", "h_0_c", "h_1_m", "h_2_m", "h_2_c", "h_3_m", "a", "lamb_m", "lamb_c"]
        values = [h_0_m[0], h_0_c[0], h_1_m[0], h_2_m[0], h_2_c[0], h_3_m[0], a, lamb_m, lamb_c]
        self.lattice_params(**dict([(key, value) for key, value in zip(keys, values)]))

    def _stitching(self, u_local, h1, h2=None, shape=None, ur=None):
        if u_local.ndim == 1:
            if h2 is None:
                h2 = np.zeros([h1.shape[0], 0, 0])
            if ur is None:
                ur = np.eye(3)
            output = np.zeros([h1.shape[1] + h2.shape[1], h1.shape[2] + h2.shape[2]])
            output[:h1.shape[1], :h1.shape[2]] = np.dot(ur.T[:h1.shape[1], :h1.shape[1]],
                                                        np.dot(np.dot(h1[1:].T, u_local).T,
                                                               ur[:h1.shape[2], :h1.shape[2]]))
            output[h1.shape[1]:, h1.shape[2]:] = np.dot(ur.T[:h2.shape[1], :h2.shape[1]],
                                                        np.dot(np.dot(h2[1:].T, u_local).T,
                                                               ur[:h2.shape[2], :h2.shape[2]]))
            if self.soc_doubled_ham:
                output = np.kron(np.eye(2), output)
            return output
        else:
            if u_local.shape[1] == 0:
                return 0 if shape is None else np.zeros([0, shape[1], shape[2]])
            return np.array([self._stitching(u_i, h1, h2, shape, ur) for u_i in u_local.T])

    def straining_hopping(self, uxx, uyy, uxy, hop_id):
        if not hasattr(uxx, "__len__"):
            uxx = np.array([uxx])
            uyy = np.array([uyy])
            uxy = np.array([uxy])

        u_matrix = np.array((uxx + uyy, uxx - uyy, 2 * uxy))

        ur_1 = np.array([[1, 0, 0],
                         [0, -1 / 2, -np.sqrt(3) / 2],
                         [0, np.sqrt(3) / 2, -1 / 2]])
        ur_2 = ur_1.dot(ur_1)

        ur = self.orbital.ur
        ur_names = [*ur]
        h_name, n_i, from_name, to_name = self._separate_name(str(hop_id))
        h_matrix = self.strain_params[h_name]

        ur_r_b = np.where([name in from_name for name in ur_names])
        ur_l_b = np.where([name in to_name for name in ur_names])
        assert len(ur_r_b) == 1, "There seems to be a problem with the names in the hopping and the lattice for 'from'"
        assert len(ur_l_b) == 1, "There seems to be a problem with the names in the hopping and the lattice for 'to'"
        ur_r = ur[ur_names[ur_r_b[0][0]]]
        ur_l = ur[ur_names[ur_l_b[0][0]]]

        if n_i == 0:
            ur_r = np.eye(*ur_r.shape)
            ur_l = np.eye(*ur_l.shape)
        elif n_i == 1:
            ur_l = ur_l.T
            u_matrix = ur_1.dot(u_matrix)
        elif n_i == 2:
            ur_r = ur_r.T
            u_matrix = ur_2.dot(u_matrix)
        else:
            assert False, "n_i for the hopping is incorrect, n_i=" + str(n_i)

        hopping = np.einsum("cj,kjl,lb,ka->abc", ur_l, h_matrix[1:], ur_r, u_matrix)

        if self.single_orbital:
            def find_idx(find_name, error_name):
                if find_name in self._m_orbs:
                    idx = np.where([name == find_name for name in self._m_orbs])[0][0]
                    idx_max = len(self._m_orbs)
                elif find_name in self._c_orbs:
                    idx = np.where([name == find_name for name in self._c_orbs])[0][0]
                    idx_max = len(self._c_orbs)
                elif self.lat4:
                    if find_name in self._m2_orbs:
                        idx = np.where([name == find_name for name in self._m2_orbs])[0][0]
                        idx_max = len(self._m2_orbs)
                    elif find_name in self._c2_orbs:
                        idx = np.where([name == find_name for name in self._c2_orbs])[0][0]
                        idx_max = len(self._c2_orbs)
                    else:
                        assert False, error_name + " of hopping not in the list of possible orbital names in lat4 case"
                else:
                    assert False, error_name + " of hopping not in the list of possible orbital names in not lat4 case"
                return idx, idx_max
            idx_from, idx_max_from = find_idx(from_name, "from-name")
            idx_to, idx_max_to = find_idx(to_name, "to-name")
            if self.soc_doubled_ham:
                if idx_from >= idx_max_from / 2 and idx_to >= idx_max_to / 2:
                    idx_from = idx_from % int(idx_max_from / 2)
                    idx_to = idx_to % int(idx_max_to / 2)
                    hopping = hopping[:, idx_from, idx_to]
                elif idx_from >= idx_max_from / 2 or idx_to >= idx_max_to / 2:
                    hopping = 0
                else:
                    hopping = hopping[:, idx_from, idx_to] #might be the transpose
            else:
                hopping = hopping[:, idx_from, idx_to]
        else:
            if self.soc_doubled_ham:
                hopping = np.kron(np.eye(2), hopping)
        return hopping

    def strained_onsite(self, uxx, uyy, uxy, sub_id=None):
        s_id = str(sub_id)
        if self.single_orbital:
            if s_id in self._m_orbs:
                h_name = "h_0_m"
            elif s_id in self._c_orbs:
                h_name = "h_0_c"
            elif self.lat4:
                if s_id in self._m2_orbs:
                    h_name = "h_0_m"
                elif s_id in self._c2_orbs:
                    h_name = "h_0_c"
                else:
                    assert False, "Onsite name not in list for lat4 single-orbital"
            else:
                assert False, "Onsite name not in list for not lat4 single-orbital"
        else:
            if s_id == self.m_name:
                h_name = "h_0_m"
            elif s_id == self.c_name:
                h_name = "h_0_c"
            elif self.lat4:
                if s_id == self.m_name + "2":
                    h_name = "h_0_m"
                elif s_id == self.c_name + "2":
                    h_name = "h_0_c"
                else:
                    assert False, "Onsite name not in list for lat4"
            else:
                assert False, "Onsite name not in list for not lat4"
        name_out = self._make_name(h_name, 1, s_id, s_id)
        return self.straining_hopping(uxx, uyy, uxy, name_out)

    def uniform_strain(self, uxx, uyy, uxy):
        @pb.hopping_energy_modifier
        def strained_hoppings(energy, x1, y1, z1, x2, y2, z2, hop_id):
            additional = self.straining_hopping(uxx, uyy, uxy, hop_id=hop_id)
            return energy + additional

        @pb.onsite_energy_modifier
        def strained_onsite(energy, x, y, z, sub_id):
            additional = self.strained_onsite(uxx, uyy, uxy, sub_id=sub_id)
            return energy + additional

        return strained_hoppings, strained_onsite

    def test1(self, soc=False, lat4=False, path=1):
        # plot all
        print("Doing every calculation with high precision,\n this can take a minute.")

        for soc in [False]:
            self.soc = soc

            #grid = plt.GridSpec(2, 2, hspace=0.4)
            #plt.figure(figsize=(6.7, 8))

            for name in zip(["MoS2", "WS2", "MoSe2", "WSe2"]):
                self.name = name
                model = pb.Model(self.lattice(), pb.translational_symmetry())
                solver = pb.solver.lapack(model)

                k_points = model.lattice.brillouin_zone()
                gamma = [0, 0]
                k = k_points[0]
                m = (k_points[0] + k_points[1]) / 2

                plt.plot(title=name)
                bands = solver.calc_bands(gamma, k, m, gamma, step=0.005)
                bands.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"], lw=1.5)

    def test2(self, u=None, soc=False):
        plt.figure()
        # plot bands comparing lat4=False and lat4=True with a certain strain vector, giving the largest error between
        # the two models
        if u is None:
            u = np.array([1, .34, -.42])
        bands = []
        model = pb.Model(self.lattice(), pb.translational_symmetry())
        k_points = model.lattice.brillouin_zone()
        gamma = [0, 0]
        k = k_points[0]
        m = (k_points[0] + k_points[1]) / 2
        for lat4 in [False]:
            self.single_orbital = lat4
            self.soc = False
            self.lat4 = True
            model = pb.Model(self.lattice(),
                             self.uniform_strain(u[0], u[1], u[2]),
                             pb.translational_symmetry())
            solver = pb.solver.lapack(model)

            bands.append(solver.calc_bands(gamma, k, m, gamma, step=0.5))
        bands[0].plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"], lw=1.5, color="C0")
        bands[1].plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"], lw=1.5, color="C1", ls=":")
        #error = np.array(
        #    [[np.abs(bands[1].energy[i] - bands[0].energy[i, j]).min()
        #      for j in range(bands[0].energy.shape[1])] for i in#
        #     range(bands[0].energy.shape[0])])
        # class.test2()[0] for maximal error
        #return error.max(), error, bands


if __name__ == "__main__":
    group4_class = Group4Tmd11BandStrain()
    group4_class.test2(u=np.array([0, 0, 0]))
    # res = group4_class.test2()
    ham = group4_class.construct_hamiltonian_fang_matlab(np.array([1, 2]), np.array([1, .34, -.42]))
