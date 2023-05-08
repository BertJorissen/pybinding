"""Import a Wannier90 file to a pb.Lattice"""
import numpy as np
from .lattice import Lattice
from .support.parse import win, hr_dat, xyz

from scipy.spatial import KDTree
from typing import List, Union, Optional, Tuple


"""
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

"""


class Wannier:
    """Some docstring etc."""

    def __init__(self, win_file: str, hr_dat_file: str, centres_xyz_file: Optional[str] = None):
        self._atoms_pos: np.ndarray
        self._atoms_names: List[str]
        self._atoms_names_unique: List[str]
        self._hamiltonian_parts: Tuple[np.ndarray, np.ndarray]
        self._atoms_n_orb: np.ndarray
        self._orbs_names_xyz_from_atoms: List[str]
        self._orbs_names_xyz_from_atoms_unique: List[str]

        self.single_orbital = False

        self.win_file = win_file
        self.hr_dat_file = hr_dat_file
        self.centres_xyz_file = centres_xyz_file

    def _calc_unit_cell(self):
        return self.win_file["unit_cell_cart"]

    def _calc_atoms_cart(self):
        if "atoms_cart" in self.win_file.keys():
            return self.win_file["atoms_cart"]["positions"]
        else:
            return np.transpose(self.unit_cell.T @ self.win_file["atoms_frac"]["positions"].T)

    def _calc_names(self):
        if "atoms_cart" in self.win_file.keys():
            return self.win_file["atoms_cart"]["names"]
        else:
            return self.win_file["atoms_frac"]["names"]

    def _calc_names_unique(self):
        return self._calc_names_unique_list(self.atoms_names)

    @staticmethod
    def _calc_names_unique_list(names_list: List[str]):
        names_out = []
        for name_i, name in enumerate(names_list):
            an_indices = np.array([an_i for an_i, an in enumerate(names_list) if an == name])
            if an_indices.shape[0] == 1:
                names_out.append(name)
            elif an_indices.shape[0] > 1:
                names_out.append(name + "_" + str(np.argmin(np.abs(an_indices - name_i))))
            else:
                assert False, "This shouldn't happen."
        return names_out

    def _calc_orbs_names_xyz_from_atoms(self):
        return [self.atoms_names_unique[atom_i] for atom_i in KDTree(self.atoms_pos).query(self.orbs_pos_xyz)[1]]

    def _calc_orbs_names_xyz_from_atoms_unique(self):
        return self._calc_names_unique_list(self.orbs_names_xyz_from_atoms)

    def _calc_hamiltonian_parts(self):
        r_mat = np.array(self.hr_dat_file["hr_columns"][:, :3], dtype=int)
        nm_pos = np.array(self.hr_dat_file["hr_columns"][:, 3:5], dtype=int) - 1
        h_val = (np.array(self.hr_dat_file["hr_columns"][:, 5:], dtype=complex) @ np.transpose([[1, 1j]])).T[0]
        r_mat_unique = np.unique(r_mat, axis=0)
        h_out = np.zeros((r_mat_unique.shape[0], self.num_wann, self.num_wann), dtype=complex)
        for r_k, nm, h_v in zip(r_mat, nm_pos, h_val):
            r_k_i = np.where(np.all(r_mat_unique == r_k, axis=1))[0][0]
            h_out[r_k_i, nm[0], nm[1]] = h_v / self.hr_dat_file["ws_deg"][r_k_i]
        return h_out, r_mat_unique

    def _calc_atoms_n_orb(self):
        projections = self.win_file["projections"]
        spinors = 2 if self.win_file["spinors"] else 1
        parsed_projections = dict(zip(
            [name.split(":")[0].strip() for name in projections],
            [spinors * self._orb_dict[orb.split(":")[1].strip()] for orb in projections],
        ))
        return [parsed_projections[atom] for atom in self.atoms_names]

    @property
    def _orb_dict(self):
        return dict(zip(["s", "p", "d", "f"], [1, 3, 5, 7]))

    @property
    def win_file(self):
        return self._win_file

    @property
    def hr_dat_file(self):
        return self._hr_dat_file

    @property
    def centres_xyz_file(self):
        return self._centres_xyz_file

    @property
    def unit_cell(self):
        return self.win_file["unit_cell_cart"]

    @property
    def atoms_pos(self):
        return self._atoms_pos

    @property
    def atoms_names(self):
        return self._atoms_names

    @property
    def atoms_names_unique(self):
        return self._atoms_names_unique

    @property
    def atoms_n_orb(self):
        return self._atoms_n_orb

    @property
    def orbs_pos_xyz(self):
        return self.centres_xyz_file["xyz"][:self.num_wann]

    @property
    def orbs_names_xyz(self):
        return self.centres_xyz_file["atoms"][:self.num_wann]

    @property
    def orbs_names_xyz_from_atoms(self):
        return self._orbs_names_xyz_from_atoms

    @property
    def orbs_names_xyz_from_atoms_unique(self):
        return self._orbs_names_xyz_from_atoms_unique

    @property
    def atoms_i_orb(self):
        # TODO: add more projections, parse them, and test if they give the right result
        return [np.arange(ani) + ano for ani, ano in
                zip(self._atoms_n_orb, np.hstack((0, np.cumsum(self._atoms_n_orb)[:-1])))]

    @property
    def num_wann(self):
        return self.win_file["num_wann"]

    @property
    def hamiltonian_parts(self):
        return self._hamiltonian_parts

    def lattice(self, only_2d=True):

        # Define lattice-object with the unit-cell vectors
        lat = Lattice(self.unit_cell[0], self.unit_cell[1]) if only_2d else Lattice(*self.unit_cell)

        # Define the onsite terms and the onsite-names
        i_onsite = np.where(np.all(self.hamiltonian_parts[1] == [0, 0, 0], axis=1))[0][0]
        # onsite terms

        onsite_energy = self.hamiltonian_parts[0][i_onsite]
        onsite_energy_diag = np.real(onsite_energy) if self.single_orbital else onsite_energy
        atoms_i_orb = np.arange(self.num_wann).reshape((-1, 1)) if self.single_orbital else self.atoms_i_orb
        atoms_n_unq = self.orbs_names_xyz_from_atoms_unique if self.single_orbital else self.atoms_names_unique
        pos = self.orbs_pos_xyz if self.single_orbital else self.atoms_pos

        lat.add_sublattices(
            *[(sub_name, sub_pos[:2] if only_2d else sub_pos, onsite_energy_diag[:, i_orb][i_orb])
              for sub_name, sub_pos, i_orb in zip(atoms_n_unq, pos, atoms_i_orb)]
        )

        lat.add_hoppings(*[([0, 0], s_f, s_t, onsite_energy[:, atoms_i_orb[s_i + s_j + 1]][atoms_i_orb[s_i]])
                           for s_i, s_f in enumerate(atoms_n_unq[:-1]) for s_j, s_t in enumerate(atoms_n_unq[(s_i+1):])]
        )

        for r_k_i in range(len(self.hamiltonian_parts[0])):
            if r_k_i == i_onsite:
                continue
            r_k = self.hamiltonian_parts[1][r_k_i]
            if np.sum(np.all(-r_k == self.hamiltonian_parts[1][:r_k_i], axis=1)) > 0:
                continue
            ham = self.hamiltonian_parts[0][r_k_i]
            lat.add_hoppings(
                *[(r_k, s_f, s_t, ham[:, atoms_i_orb[s_j]][atoms_i_orb[s_i]])
                  for s_i, s_f in enumerate(atoms_n_unq) for s_j, s_t in enumerate(atoms_n_unq)]
            )
        return lat

    @win_file.setter
    def win_file(self, filename: str):
        self._win_file = win(filename)
        self._atoms_pos = self._calc_atoms_cart()
        self._atoms_names = self._calc_names()
        self._atoms_names_unique = self._calc_names_unique()
        self._atoms_n_orb = self._calc_atoms_n_orb()

    @hr_dat_file.setter
    def hr_dat_file(self, filename: str):
        self._hr_dat_file = hr_dat(filename)
        self._hamiltonian_parts = self._calc_hamiltonian_parts()
        # TODO: compare with the input and .xyz and check if consistent. Also build Lattice.

    @centres_xyz_file.setter
    def centres_xyz_file(self, filename: Optional[str]):
        self._centres_xyz_file = xyz(filename) if filename is not None else None
        self._orbs_names_xyz_from_atoms = self._calc_orbs_names_xyz_from_atoms() if filename is not None else self.atoms_names
        self._orbs_names_xyz_from_atoms_unique = self._calc_orbs_names_xyz_from_atoms_unique() if filename is not None else self.atoms_names_unique
        self.single_orbital = True if filename is not None else False