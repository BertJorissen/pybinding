"""Processing and presentation of computed wavefunction data"""
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Union, List, Callable
from ..support.alias import AliasArray
from .bands import Bands, BandsArea, FatBands, FatBandsArea
from .series import Series
from .spatial import SpatialLDOS
from ..disentangle import Disentangle

__all__ = ['Wavefunction', 'WavefunctionArea']


class Wavefunction:
    def __init__(self, bands: Bands, wavefunction: np.ndarray, sublattices: Optional[AliasArray] = None,
                 system=None):
        """ Class to store the results of a Wavefunction.

        Parameters:
            bands : bands
                The band structure, with eigenvalues and k_path, of the wavefunction
            wavefunction : np.ndarray()
                ND-array. The first dimension corresponds with the k-point, the second with the band (sorted values),
                the last index with the relative dimension of the wavefunction. The wavefunction is complex,
                and already rescaled to give a norm of 1. The np.dot-function is used to calculate the overlap
                with the hermitian conjugate.
        """
        self.bands: Bands = bands
        self.wavefunction: np.ndarray = wavefunction
        self._overlap_matrix: Optional[np.ndarray] = None
        self._disentangle: Optional[Disentangle] = None
        self._sublattices: Optional[AliasArray] = sublattices
        self._system = system

    @property
    def overlap_matrix(self) -> np.ndarray:
        """ Give back the overlap matrix

        Returns : np.ndarray
            The overlap matrix between the different k-points.
        """
        if self._overlap_matrix is None:
            self._overlap_matrix = self._calc_overlap_matrix()
        return self._overlap_matrix

    def _calc_overlap_matrix(self) -> np.ndarray:
        """ Calculate the overlap of all the wavefunctions with each other

            Parameters
            Returns : np.ndarray()
                3D array with the relative coverlap between the k-point and the previous k-point
            """
        assert len(self.wavefunction.shape) == 3, \
            "The favefunction has the wrong shape, {0} and not 3".format(len(self.wavefunction.shape))
        n_k = self.wavefunction.shape[0]
        assert n_k > 1, "There must be more than one k-point, first dimension is not larger than 1."
        return np.array([np.abs(self.wavefunction[i_k] @ self.wavefunction[i_k + 1].T.conj())
                         for i_k in range(n_k - 1)])

    @property
    def disentangle(self):
        """ Give back a Disentanlement-class, and save the class for further usage.

        Returns : Disentangle
            Class to perform disentanglement
        """
        if self._disentangle is None:
            self._disentangle = Disentangle(self.overlap_matrix)
        return self._disentangle

    @property
    def bands_disentangled(self) -> Bands:
        """ Disentangle the bands from the wavefunction.

        Returns : Bands
            The reordered eigenvalues in a Bands-class."""
        return Bands(self.bands.k_path, self.disentangle(self.bands.energy))

    def _fatbands(self, suborbital: bool = False) -> FatBands:
        """ Return FatBands with the pDOS for each sublattice.

        Parameters
        ----------
        suborbital : bool
            If the pDOS should be calculated for each orbital. Default: False.

        Returns : FatBands
            The (unsorted) bands with the pDOS.
        """
        probablitiy = np.abs(self.wavefunction ** 2)
        labels = {"data": "pDOS", "columns": "orbital"}
        if self._sublattices is not None:
            mapping = self._sublattices.mapping
            keys = mapping.keys()
            data_len = np.shape(probablitiy)[2] if suborbital else len(keys)
            data = np.zeros((self.bands.energy.shape[0], self.bands.energy.shape[1], data_len))
            data_index = 0
            orbital_labels = []
            for key in keys:
                probablitiy_sliced = probablitiy[:, :, self._sublattices == key]
                if suborbital:
                    for suborbital_index in range(probablitiy_sliced.shape[2]):
                        data[:, :, len(orbital_labels)] = probablitiy_sliced[:, :, suborbital_index]
                        orbital_labels.append(str(key) + "_" + str(suborbital_index))
                        data_index += 1
                else:
                    data[:, :, len(orbital_labels)] = np.sum(probablitiy_sliced, axis=2)
                    orbital_labels.append(str(key))
            labels["orbitals"] = orbital_labels
        else:
            data = probablitiy
        return FatBands(Bands(self.bands.k_path, self.bands.energy), data, labels)

    @property
    def fatbands(self) -> FatBands:
        """ Return FatBands with the pDOS for each sublattice.

        Returns : FatBands
            The (unsorted) bands with the pDOS.
        """
        return self._fatbands()

    @property
    def fatbands_suborbital(self) -> FatBands:
        """ Return FatBands with the pDOS for each sublattice.

        Returns : FatBands
            The (unsorted) bands with the pDOS.
        """
        return self._fatbands(suborbital=True)

    @property
    def fatbands_disentangled(self) -> FatBands:
        """ Return FatBands with the pDOS for each sublattice.

        Returns : FatBands
            The (sorted) bands with the pDOS.
        """
        fatbands = self.fatbands
        return FatBands(Bands(self.bands.k_path, self.bands_disentangled.energy),
                        self.disentangle(fatbands.data), fatbands.labels)

    @property
    def fatbands_suborbital_disentangled(self) -> FatBands:
        """ Return FatBands with the pDOS for each sublattice.

        Returns : FatBands
            The (sorted) bands with the pDOS.
        """
        fatbands = self.fatbands_suborbital
        return FatBands(Bands(self.bands.k_path, self.bands_disentangled.energy),
                        self.disentangle(fatbands.data), fatbands.labels)

    def spatial_ldos(self, energies: Optional[ArrayLike] = None,
                     broadening: Optional[float] = None) -> Union[Series, SpatialLDOS]:
        r"""Calculate the spatial local density of states at the given energy

        .. math::
            \text{LDOS}(r) = \frac{1}{c \sqrt{2\pi}}
                             \sum_n{|\Psi_n(r)|^2 e^{-\frac{(E_n - E)^2}{2 c^2}}}

        for each position :math:`r` in `system.positions`, where :math:`E` is `energy`,
        :math:`c` is `broadening`, :math:`E_n` is `eigenvalues[n]` and :math:`\Psi_n(r)`
        is `eigenvectors[:, n]`.

        Parameters
        ----------
        energies : Arraylike
            The energy value for which the spatial LDOS is calculated.
        broadening : float
            Controls the width of the Gaussian broadening applied to the DOS.

        Returns
        -------
        :class:`~pybinding.StructureMap`
        """
        if energies is None:
            energies = np.linspace(np.nanmin(self.bands.energy), np.nanmax(self.bands.energy), 1000)
        if broadening is None:
            broadening = (np.nanmax(self.bands.energy) - np.nanmin(self.bands.energy)) / 1000
        scale = 1 / (broadening * np.sqrt(2 * np.pi) * self.bands.energy.shape[0])
        ldos = np.zeros((self.wavefunction.shape[2], len(energies)))
        for i_k, eigenvalue in enumerate(self.bands.energy):
            delta = np.nan_to_num(eigenvalue)[:, np.newaxis] - energies
            gauss = np.exp(-0.5 * delta**2 / broadening**2)
            psi2 = np.nan_to_num(np.abs(self.wavefunction[i_k].T)**2)
            ldos += scale * np.sum(psi2[:, :, np.newaxis] * gauss[np.newaxis, :, :], axis=1)
        if self._system is not None:
            return SpatialLDOS(ldos.T, energies, self._system)
        else:
            labels = {"variable": "E (eV)", "data": "sLDOS", "columns": "orbitals"}
            if self._sublattices is not None:
                mapping = self._sublattices.mapping
                keys = mapping.keys()
                data = np.zeros((len(energies), len(keys)))
                for i_k, key in enumerate(keys):
                    data[:, i_k] = np.sum(ldos[self._sublattices == key, :], axis=0)
                labels["orbitals"] = [str(key) for key in keys]
                ldos = data
            else:
                labels["orbitals"] = [str(i) for i in range(self.wavefunction_1d.shape[2])]
                ldos = ldos.T
            return Series(energies, ldos, labels=labels)

    def operator(self, operator: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]], disentangle: bool = False,
                 names: Union[Optional[List[str]], str] = None) -> FatBands:
        """Apply the operator on the wavefunction. Only from-to same band, no correlations between bands.

        Parameters
        ----------
        operator : np.ndarray or Callable
            The operators to apply on the system. This should be in the shape of (ham_idx, ham_idx).
            Optionally, this can be 3D, (op_idx, ham_idx, ham_idx), with op_idx different operators.
            If this is a function (Callable), the function should accept the k-vector to build the operator
            for that k-vector, and return a np.ndarray as would be provided if it were not a Callable mentioned above.
        disentangle : bool
            If the bands and the results should be disentangled. Defualt: False.
        names : Union[Optional[List[str]], str]
            The names for the labels of the operators.

        Returns
        -------
        :class: ~pybinding.FatBands
        """
        data_mat = None  # only determine size after the first call
        operator_dims = None  # only determine size after the first call
        for wfc_i, wfc in enumerate(self.wavefunction):  # loop over every k-point
            operator_loc = operator(self.bands.k_path[wfc_i]) if callable(operator) else operator  # get the operator
            if operator_loc.ndim == 2:  # see if we are working with a single operator
                operator_loc = [operator_loc]  # add another ghost dimension
            if data_mat is None:  # construct the data matrix with the first call
                operator_dims = (len(operator_loc), *operator_loc[0].shape)  # get the size of the operator
                assert operator_dims[1] == self.wavefunction.shape[2], \
                    "The first dimension of the operator doesn't match the the wavefunction, {0} != {1}".format(
                        operator_dims[1], self.wavefunction.shape[2]
                    )  # check the size of the operator
                assert operator_dims[2] == self.wavefunction.shape[2], \
                    "The second dimension of the operator doesn't match the the wavefunction, {0} != {1}".format(
                        operator_dims[2], self.wavefunction.shape[2]
                    )  # check the size of the operator
                data_mat = np.zeros((self.wavefunction.shape[0], self.wavefunction.shape[1], operator_dims[0]))
                # reserve the memory
            for opera_i, opera in enumerate(operator_loc):  # loop over the operators
                for i_b in range(self.wavefunction.shape[1]):  # loop over the bands
                    data_mat[wfc_i, i_b, opera_i] = (wfc.conj()[i_b, :] @ opera @ wfc.T[:, i_b]).real  # calculate value
        fatbands_out = self.fatbands_disentangled if disentangle else self.fatbands  # generate the fatbands object
        fatbands_out.data = self.disentangle(data_mat) if disentangle else data_mat  # append the data
        if names is None:  # determine the right name
            fatbands_out.labels["orbitals"] = [str(name_i) for name_i in range(operator_dims[0])]
        elif isinstance(names, str):  # if only a string is given, wrap it with a list
            fatbands_out.labels["orbitals"] = [names]
        else:  # the correct shape should be given, List[str] with length the amount of operators
            fatbands_out.labels["orbitals"] = names
        return fatbands_out


class WavefunctionArea(Wavefunction):
    def __init__(self, bands: BandsArea, wavefunction: np.ndarray, sublattices: Optional[AliasArray] = None,
                 system=None):
        """ Class to store the results of a Wavefunction for an Area.

        Parameters:
            bands : BandsArea
                The band structure, with eigenvalues and k_path, of the wavefunction
            wavefunction : np.ndarray()
                ND-array. The first dimension corresponds with the k-point, the second with the band (sorted values),
                the last index with the relative dimension of the wavefunction. The wavefunction is complex,
                and already rescaled to give a norm of 1. The np.dot-function is used to calculate the overlap
                with the hermitian conjugate.
        """

        super().__init__(bands, bands.area_to_list(wavefunction), sublattices, system)
        self.bands: BandsArea = bands

    @property
    def wavefunction_area(self) -> np.ndarray:
        return self.bands.list_to_area(self.wavefunction)

    @wavefunction_area.setter
    def wavefunction_area(self, wavefunction: np.ndarray):
        self.wavefunction = self.bands.area_to_list(wavefunction)

    @property
    def fatbandsarea(self) -> FatBandsArea:
        return FatBandsArea(self.bands, self.bands.list_to_area(self.fatbands.data), self.fatbands.labels)

    @property
    def fatbandsarea_suborbital(self) -> FatBandsArea:
        return FatBandsArea(self.bands, self.bands.list_to_area(self.fatbands_suborbital.data), self.fatbands_suborbital.labels)

    @property
    def fatbandsarea_disentangled(self) -> FatBandsArea:
        return FatBandsArea(
            BandsArea(self.bands.k_area, self.bands.list_to_area(self.bands_disentangled.energy)),
            self.bands.list_to_area(self.disentangle(self.fatbands.data)), self.fatbands.labels
        )

    @property
    def fatbandsarea_suborbital_disentangled(self) -> FatBandsArea:
        return FatBandsArea(
            BandsArea(self.bands.k_area, self.bands.list_to_area(self.bands_disentangled.energy)),
            self.bands.list_to_area(self.disentangle(self.fatbands_suborbital.data)), self.fatbands_suborbital.labels
        )
