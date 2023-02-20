"""Import a Wannier90 file to a pb.Lattice"""
import numpy as np
import warnings

from typing import List, Union, Optional, Tuple

bohr = 1.8897259886


class Units:
    def __init__(self, scale=1.):
        self.scale = scale

IntList = Optional[List[int]]
IntVector = Optional[Tuple[int, int, int]]
IntVector5 = Optional[Tuple[int, int, int, int, int]]
Vectors = Optional[List[Tuple[float, float, float]]]
NamedVectors = Optional[List[Tuple[str, float, float, float]]]

UnitsLnVectors = Optional[Union[Units, Vectors]]
NamedUnitsLnVectors = Optional[Union[Units, NamedVectors]]

UnitsRsVectors = Optional[Union[Units, Vectors]]
UnitsEnVectors = Optional[Union[Units, Vectors]]
Int = Optional[int]
Bool = Optional[bool]
Real = Optional[float]
String = Optional[str]

# name_of_var: (type, default value [-1 if required, None if optional, str oif other param], can't be defined at the same time)
input_types_win = {
    "num_wann": (Int, -1, None),
    "num_bands": (Int, "num_wann", None),
    "unit_cell_cart": (UnitsLnVectors, -1, None),
    "atoms_cart": (NamedUnitsLnVectors, -1, ["atoms_frac"]),
    "atoms_frac": (NamedVectors, -1, ["atoms_cart"]),
    "mp_grid": (IntVector, -1, None),
    "kpoints": (Vectors, -1, None),
    "gamma_only": (Bool, False, None),
    "spinors": (Bool, False, None),
    "shell_list": (IntList, None, None),
    "search_shells": (Int, 36, None),
    "skip_b1_tests": (Bool, False, None),
    "nnkpts": (IntVector5, None, None),
    "kmesh_tol": (Real, None, None),
    "postproc_setup": Bool,
    "exclude_bands": Int,
    "select_projections": Int,
    "auto_projections": Bool,
    "restart": String,
    "iprint": Int,
    "length_unit": String,
    "wvfn_formatted": Bool,
    "spin": String,
    "devel_flag": String,
    "timing_level": Int,
    "optimisation": Int,
    "translate_home_cell": Bool,
    "write_xyz": Bool,
    "write_vdw_data": Bool,
    "write_hr_diag": Bool,
    "wannier_plot": Bool,
    "wannier_plot_list": Int,
    "wannier_plot_supercell": Int,
    "wannier_plot_format": String,
    "wannier_plot_mode": String,
    "wannier_plot_radius": Units,
    "wannier_plot_scale": Units,
    "wannier_plot_spinor_mode": String,
    "wannier_plot_spinor_phase": Bool,
    "bands_plot": Bool,
    "kpoint_path": look_into_this,
    "bands_num_points": Int,
    "bands_plot_format": String,
    "bands_plot_project": Int,
    "bands_plot_mode": String,
    "bands_plot_dim": Int,
    "fermi_surface_plot": Bool,
    "fermi_surface_num_points": Int,
    "fermi_energy": Units_energy,
    "fermi_energy_min": Units_energy,
    "fermi_energy_max": Units_energy,
    "fermi_energy_step": Real,
    "fermi_surface_plot_format": String,
    "hr_plot": Bool,
    "write_hr": Bool,
    "write_rmn": Bool,
    "write_bvec": Bool,
    "write_tb": Bool,
    "hr_cutoff": units_energy,
    "dist_cutoff": units_length,
    "dist_cutoff_mode": String,
    "translation_centre_frac": Real,
    "use_ws_distance": Bool,
    "ws_distance_tol": Real,
    "ws_search_size": Int,
    "write_u_matrices": Bool,
    "dis_win_min": units_len,
    "dis_win_max": units_len,
    "dis_froz_min": units_len,
    "dis_froz_max": units_len,
    "dis_num_iter": Int,
    "dis_mix_ratio": Real,
    "dis_conv_tol": Real,
    "dis_conv_window": Int,
    "dis_spheres_num": Int,
    "dis_spheres_first_wann": Int,
    "dis_spheres": Real,
    "num_iter": Int,
    "num_cg_steps": Int,
    "conv_window": Int,
    "conv_tol": units_energy,
    "precond": Bool,
    "conv_noise_amp": Real,
    "conv_noise_num": Int,
    "num_dump_cycles": Int,
    "num_print_cycles": Int,
    "write_r2mn": Bool,
    "guiding_centres": Bool,
    "num_guide_cycles": Int,
    "num_no_guide_iter": Int,
    "trial_step": Real,
    "fixed_step": Real,
    "use_bloch_phases": Bool,
    "site_symmetry": Bool,
    "symmetrize_eps": Real,
    "slwf_num": Int,
    "slwf_constrain": Bool,
    "slwf_lambda": Real,
    "slwf_centres": units_lenth,
    "transport": Bool,
    "transport_mode": String,
    "tran_win_min": units_energy,
    "tran_win_max": units_energy,
    "tran_energy_step": Real,
    "fermi_energy": Real,
    "tran_num_bb": Int,
    "tran_num_ll": Int,
    "tran_num_rr": Int,
    "tran_num_cc": Int,
    "tran_num_lc": Int,
    "tran_num_cr": Int,
    "tran_num_cell_ll": Int,
    "tran_num_cell_rr": Int,
    "tran_num_bandc": Int,
    "tran_write_ht": Bool,
    "tran_read_ht": Bool,
    "tran_use_same_lead": Bool,
    "tran_group_threshold": Real,
    "hr_cutoff": units_energy,
    "dist_cutoff": units_lkength,
    "dist_cutoff_mode": String,
    "one_dim_axis": String,
    "translation_centre_frac": Real
}

def to_int(int_str):
    try:
        return int(int_str)
    except ValueError:
        return None


def to_float(float_str):
    try:
        float_str = float_str[0] + float_str[1:].replace("-", "e-").replace("+", "e+")
        return float(float_str)
    except ValueError:
        return None


def parse_win_file(filename):
    keys, values, tmp_value = [], [], []
    tmp_key = ""

    for line in open(filename):
        if line.strip()[:5] == "begin":
            tmp_key = line.split("begin")[1].split("#")[0].split("!")[0].strip().lower()
            tmp_value = []
        elif line.strip()[:3] == "end":
            end_key = line.split("end")[1].split("#")[0].split("!")[0].strip().lower()
            assert tmp_key == end_key, "The segment didn't close properly, {0} != {1}".format(tmp_key, end_key)
            if tmp_key in ["unit_cell_cart", "atoms_cart", "atoms_frac", "kpoints"]:
                tmp_scale = 1 if tmp_key in ["atoms_frac", "kpoints"] else .1
                if tmp_value[0].lower() == "bohr":
                    tmp_scale /= bohr
                    tmp_value = tmp_value[1:]
                elif tmp_value[0].lower() in ["ang", "angstrom"]:
                    tmp_value = tmp_value[1:]
                if tmp_key in ["atoms_cart", "atoms_frac"]:
                    tmp_names = [lat_line.split(" ", 1)[0] for lat_line in tmp_value]
                    tmp_value = [lat_line.split(" ", 1)[1] for lat_line in tmp_value]
                tmp_value = tmp_scale * np.array([np.fromstring(lat_line, sep=" ") for lat_line in tmp_value])
                if tmp_key in ["atoms_cart", "atoms_frac"]:
                    tmp_value = dict(zip(["names", "positions"], [tmp_names, tmp_value]))
            if tmp_key == "kpoint_path":
                tmp_names_from = [lat_line.split()[0] for lat_line in tmp_value]
                tmp_value_from = [" ".join(lat_line.split(maxsplit=1)[1].split(maxsplit=3)[:3])
                                  for lat_line in tmp_value]
                tmp_names_to = [lat_line.split()[4] for lat_line in tmp_value]
                tmp_value_to = ["".join(lat_line.split(maxsplit=5)[5]) for lat_line in tmp_value]
                tmp_value_from = np.array([np.fromstring(lat_line, sep=" ") for lat_line in tmp_value_from])
                tmp_value_to = np.array([np.fromstring(lat_line, sep=" ") for lat_line in tmp_value_to])
                tmp_value = dict(zip(["names_from", "kpoints_from", "names_to", "kpoints_to"],
                                     [tmp_names_from, tmp_value_from, tmp_names_to, tmp_value_to]))
            keys.append(tmp_key)
            values.append(tmp_value)
            tmp_key = ""
        else:
            if not tmp_key == "":
                tmp_value.append(line.strip())
            else:
                line_uncomment = line.split("#")[0].split("!")[0].strip()
                split_symb = " "
                if "=" in line_uncomment:
                    split_symb = "="
                elif ":" in line_uncomment:
                    split_symb = ":"
                line_split = line_uncomment.split(split_symb)
                assert len(line_split) in (1, 2), "Encountered an unreadable line: \n----\n{0}----".format(line)
                if len(line_split) == 2:
                    key, value = line_split
                    key = key.strip().lower()
                    keys.append(key)
                    value = value.strip().lower()
                    if value in ["true", "t", ".true.."]:
                        value = True
                    elif value in ["false", "f", ".false.."]:
                        value = False
                    elif to_int(value) is not None:
                        value = to_int(value)
                    elif to_float(value) is not None:
                        value = to_float(value)
                    if key in ["mp_grid"]:
                        value = np.fromstring(value, sep=" ", dtype=int).tolist()
                    values.append(value)
                elif not line_split[0].strip() == "":
                    assert False, "Encountered an unreadable line: \n----\n{0}----".format(line)
    return dict(zip(keys, values))


def parse_hr_dat_file(filename):
    keys, values = [], []
    num_wann, nrpts, line_i = 0, 0, 0
    ws_deg_points = np.array([], dtype=int)

    for line_i, line in enumerate(open(filename)):
        value = line.strip()
        if line_i == 0:
            keys.append("comment")
            values.append(value)
        elif line_i == 1:
            assert len(value.split()) == 1 or str(int(value)) == value,\
                "The number of Wannier orbitals should be a single number, got this: \n----\n{0}----".format(value)
            keys.append("num_wann")
            num_wann = int(value)
            values.append(num_wann)
        elif line_i == 2:
            assert len(value.split()) == 1 or str(int(value)) == value, \
                "The number of Wigner-Seitz grid-points nrpts should be a single number, got this:" \
                "\n----\n{0}----".format(value)
            keys.append("nrpts")
            nrpts = int(value)
            values.append(nrpts)
        elif line_i == 3 or ws_deg_points.shape[0] < nrpts:
            value = np.fromstring(value, sep=" ", dtype=int)
            ws_deg_points = np.hstack((ws_deg_points, value))
            if (not value.shape[0] == 15) and (not ws_deg_points.shape[0] == nrpts):
                warnings.warn("There were more enetries on the WS-degs. Should be 15 (except last),"
                              "but {0} were passed".format(value.shape[0]), UserWarning)
        elif ws_deg_points.shape[0] == nrpts:
            keys.append("ws_deg")
            values.append(ws_deg_points.tolist())
            break
        else:
            assert False, "Apperently, the loop didn't stop. Maybe too many Wigner-Seitz degenerate points? " \
                          "({0}), {1}!={2}".format(ws_deg_points, ws_deg_points.shape[0], nrpts)
    keys.append("hr_columns")
    ham_elements = np.loadtxt(filename, skiprows=line_i)
    assert ham_elements.shape[0] == num_wann ** 2 * nrpts,\
        "The number of matrix elements has not the expected length: {0} != {1}".format(
            ham_elements.shape[0], num_wann ** 2 * nrpts
        )
    values.append(ham_elements)

    # TODO: compare with the input and .xyz and check if consistent. Also build Lattice.
    return dict(zip(keys, values))
