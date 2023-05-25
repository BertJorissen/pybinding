import numpy as np
from .results import SeriesArea, WavefunctionArea


def _wf_dpr(wf1, wf2):
    """calculate dot product between two wavefunctions.
    wf1 and wf2 are of the form [orbital,spin]"""
    return np.dot(wf1.flatten().conjugate(), wf2.flatten())


def _no_pi(x,clos):
    "Make x as close to clos by adding or removing pi"
    while abs(clos-x)>.5*np.pi:
        if clos-x>.5*np.pi:
            x+=np.pi
        elif clos-x<-.5*np.pi:
            x-=np.pi
    return x


def _one_phase_cont(pha, clos):
    """Reads in 1d array of numbers *pha* and makes sure that they are
    continuous, i.e., that there are no jumps of 2pi. First number is
    made as close to *clos* as possible."""
    ret=np.copy(pha)
    # go through entire list and "iron out" 2pi jumps
    for i in range(len(ret)):
        # which number to compare to
        if i == 0:
            cmpr = clos
        else:
            cmpr = ret[i-1]
        # make sure there are no 2pi jumps
        ret[i]=_no_pi(ret[i], cmpr)
    return ret


def _one_berry_loop(wf):
    nocc = wf.shape[1]
    # temporary matrices
    prd = np.identity(nocc, dtype=complex)
    ovr = np.zeros([nocc, nocc], dtype=complex)
    # go over all pairs of k-points, assuming that last point is overcounted!
    for i in range(wf.shape[0] - 1):
        # generate overlap matrix, go over all bands
        for j in range(nocc):
            for k in range(nocc):
                ovr[j, k] = _wf_dpr(wf[i, j, :], wf[i + 1, k, :])
        # multiply overlap matrices
        prd = np.dot(prd, ovr)
    det = np.linalg.det(prd)
    pha = (-1.0) * np.angle(det)
    return pha


def calc_berry(wfc: WavefunctionArea, rescale=True) -> SeriesArea:
    wfs2d = np.array(wfc.wavefunction_area, dtype=complex)
    all_phases = np.zeros((wfs2d.shape[0], wfs2d.shape[1]), dtype=float)
    for i in range(wfs2d.shape[0] - 1):
        for j in range(wfs2d.shape[1] - 1):
            all_phases[i, j] = _one_berry_loop(np.array([
                wfs2d[i, j], wfs2d[i + 1, j], wfs2d[i + 1, j + 1], wfs2d[i, j + 1], wfs2d[i, j]
            ], dtype=complex))
    if rescale:
        all_phases = all_phases / (np.max(all_phases) - np.min(all_phases)) * 2
    # ToDo: check if the orderings etc. are correct
    return SeriesArea(wfc.bands.k_area, all_phases)
