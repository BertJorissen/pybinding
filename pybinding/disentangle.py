"""Code for disentangling data based on the overlap of wavefunctions."""

import numpy as np
from typing import Optional, Tuple, List
from .support.pickle import pickleable

__all__ = ['Disentangle']


@pickleable
class Disentangle:
    def __init__(self, overlap_matrix: np.ndarray):
        """
        A Class to store the product matrix for a wavefunction, not the wavefunction itself.
            Main application is for disentangling for the band structure.

        Parameters
        ----------
        overlap_matrix : np.ndarray
            Array of the product of the wave function between two k-points.
        """
        self.overlap_matrix: np.ndarray = overlap_matrix
        self.threshold: float = np.abs(2 * np.shape(overlap_matrix)[1]) ** -0.25
        self._disentangle_matrix: Optional[np.ndarray] = None
        self._routine: int = 1
        self._no_reorder_indx: List[int] = []

    @property
    def disentangle_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Give back the reordering for the band structure.

        Returns : Tuple[np.ndarray(), np.ndarray()]
            2D array with the [to-index, relative changes index] of the band for each step
        """
        if self._disentangle_matrix is None:
            self._disentangle_matrix = self._calc_disentangle_matrix()
        return self._disentangle_matrix

    def __call__(self, matrix: np.ndarray) -> np.ndarray:
        """ Apply the disentanglement on a matrix, wrapper for Disentangle._apply_disentanglement()

        usage :
            energy_sorted = Disentangle(energy_unsorted)

        Parameters : np.ndarray
                The matrix to reorder
        Returns : np.ndarray
            The reordered matrix
        """
        return self._apply_disentanglement(matrix)

    @property
    def routine(self) -> int:
        """ Give back the routine for the reordering.

        Returns : int
            The integer for the routine:
                0 -> The bands are ordered from low to high
                1 -> The scipy.optimize.linear_sum_assignment
        """
        return self._routine

    @routine.setter
    def routine(self, use: int):
        """ Set the routine for the reordering

        Parameters : int
            The integer for the routine:
                0 -> The bands are ordered from low to high
                1 -> The scipy.optimize.linear_sum_assignment
        """
        self._routine = use
        self._disentangle_matrix = None

    @property
    def no_reorder_idx(self) -> List[int]:
        """ Give back the list of indices that aren't going to be disentangled.

        This function can be useful for high-symmetry points, where it isn't clear
        what is the best band for a overlap.
        """
        return self._no_reorder_indx

    @no_reorder_idx.setter
    def no_reorder_idx(self, idx: List[int]):
        """ Give the list of indices that aren't going to be disentangled.

        This function can be useful for high-symmetry points, where it isn't cloer
        what is the best band for a overlap.
        """
        self._no_reorder_indx = idx
        self._disentangle_matrix = None

    def _calc_disentangle_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Calculate the changes in index for the band structure of which the overlap matrix is given

        Parameters
        Returns : Tuple[np.ndarray(), np.ndarray()]
            2D array with the [to-index, relative changes index] of the band for each step
        """
        assert len(self.overlap_matrix.shape) == 3, \
            "The overlap has the wrong shape, {0} and not 3".format(len(self.overlap_matrix.shape))

        n_k, n_b, n_b_2 = self.overlap_matrix.shape
        assert n_b == n_b_2, "currently, only square matrices can be used, {0} != {1}".format(n_b, n_b_2)
        # there is one more k-point
        n_k += 1

        # matrix to store the changes of the index i
        ind = np.zeros((n_k, n_b), dtype=int)

        # matrix to store value of the overlap
        keep = np.zeros((n_k, n_b), dtype=bool)

        if self.routine == 0:
            func = self._linear_sum_approx
        elif self.routine == 1:
            func = self._linear_sum_scipy
        else:
            assert False, "The value for ise_scipy of {0} doesn't even exist".format(self.routine)

        # loop over all the k-points
        for i_k in range(n_k):
            if i_k == 0:
                ind[i_k], keep[i_k] = np.arange(n_b, dtype=int), np.full(n_b, True, dtype=bool)
            else:
                if i_k in self._no_reorder_indx:
                    ind[i_k], keep[i_k] = ind[i_k - 1], np.full(n_b, True, dtype=bool)
                else:
                    ind[i_k], keep[i_k] = func(self.overlap_matrix[i_k - 1, ind[i_k - 1], :])

        working_indices = np.zeros((n_k, n_b), dtype=int)
        tmp_w_i = np.arange(n_b, dtype=int)
        for i_k in range(n_k):
            for i_b in range(n_b):
                if not keep[i_k, i_b] and not self.threshold == 0:
                    tmp_w_i[i_b] = np.max(tmp_w_i) + 1
            working_indices[i_k] = tmp_w_i
        if self.threshold == 0:
            assert np.max(working_indices) + 1 == n_b, "This shouldn't happen, the system should not increase in size."
        return ind, working_indices

    def _linear_sum_approx(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Function to calculate the equivalent of scipy.optimize.lineair_sum_assignment"""
        assert len(matrix.shape) == 2, "The matrix must have two dimensions"

        n_b, n_b_2 = matrix.shape
        assert n_b == n_b_2, "currently, only square matrices can be used, {0} != {1}".format(n_b, n_b_2)

        # matrix to store the changes of the index i
        ind = np.zeros(n_b, dtype=int)
        # matrix to store value of the overlap
        keep = np.zeros(n_b, dtype=bool)

        index_all = np.arange(n_b, dtype=int).tolist()
        for i_b in range(n_b):
            # find the new index where the value is the largest, considering only the new indices. The result
            # of 'i_max' is the relative index of the 'new indices' that aren't chosen yet
            i_max = np.argmax([matrix[i_b][i] for i in index_all])
            # first convert the relative new index to the new index, and find with old index this corresponds
            i_new = index_all.pop(i_max)
            ind[i_b] = i_new
            keep[i_b] = matrix[i_b, i_new] > self.threshold
        return ind, keep

    def _linear_sum_scipy(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Wrapper for scipy.optimize.lineair_sum_assignment"""
        from scipy.optimize import linear_sum_assignment
        orig, perm = linear_sum_assignment(-matrix)
        assert np.all(orig == np.arange(matrix.shape[0])), \
            "The orig should be a list from 1 to number of bands, but is {0}".format(orig)
        return np.array(perm, dtype=int), np.array(matrix[orig, perm] > self.threshold, dtype=bool)

    def _apply_disentanglement(self, matrix: np.ndarray) -> np.ndarray:
        """ Apply the disentanglement on a matrix

        Parameters:
            matrix : np.ndarray
                The matrix to reorder
        Returns : np.ndarray
            The reordered matrix
        """
        assert len(np.shape(matrix)) >= 2, \
            "The wavefunction has the wrong shape, {0} is smaller than 2".format(len(np.shape(matrix)))

        ind, working_indices = self.disentangle_matrix
        assert len(ind.shape) == 2, \
            "The ind matrix has the wrong shape, {0} and not 2".format(len(ind.shape))
        assert len(working_indices.shape) == 2, \
            "The working_indices matrix has the wrong shape, {0} and not 2".format(len(working_indices.shape))

        assert np.shape(matrix)[:2] == ind.shape, \
            "The shapes of the matrices don't agree (energy - ind), {0} != {1}".format(
                np.shape(matrix)[:2], ind.shape)
        assert np.shape(matrix)[:2] == ind.shape, \
            "The shapes of the matrices don't agree (energy - working_indices), {0} != {1}".format(
                np.shape(matrix)[:2], working_indices.shape)
        n_k, n_b = working_indices.shape
        size = np.array(np.shape(matrix))
        size[1] = np.max(working_indices) + 1
        out_values = np.full(size, np.nan)
        for i_k in range(n_k):
            out_values[i_k, working_indices[i_k]] = matrix[i_k, ind[i_k]]
        return out_values






