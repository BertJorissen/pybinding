import numpy as np
import matplotlib.pyplot as plt
from .results import make_path
from matplotlib.colors import ListedColormap


def colors():
    return {"yellow": np.array([242, 186, 76]) / 256,
            "blue": np.array([0, 102, 170]) / 256,
            "red": np.array([234, 44, 56]) / 256,
            "green": np.array([114, 174, 36]) / 256,
            "pink": np.array([183, 20, 160]) / 256,
            "orange": np.array([127, 127, 207]) / 256,
            "white": np.array([256, 256, 256]) / 256,
            "black": np.array([0, 0, 0]) / 256}


def colormap():
    n = 128
    colors_list = colors()
    c1 = colors_list["blue"]
    c2 = colors_list["white"]
    c3 = colors_list["yellow"]
    top = ListedColormap(np.transpose([np.ones(n) if i == 3 else np.linspace(c1[i], c2[i], n) for i in range(4)]))
    bottom = ListedColormap(np.transpose([np.ones(n) if i == 3 else np.linspace(c2[i], c3[i], n) for i in range(4)]))
    return ListedColormap(np.vstack((top(np.linspace(0, 1, n)), bottom(np.linspace(0, 1, n)))))


def make_k_path(k0, k1, *ks, step=0.1):
    k_points = [np.atleast_1d(k) for k in (k0, k1) + ks]
    return make_path(*k_points, step=step)


def _wf_dpr(wf1, wf2):
    """calculate dot product between two wavefunctions.
    wf1 and wf2 are of the form [orbital,spin]"""
    return np.dot(wf1.flatten().conjugate(), wf2.flatten())


def no_pi(x,clos):
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
        ret[i]=no_pi(ret[i], cmpr)
    return ret


def _chop(c_list, tol=10**-6):
    c_array = np.array(c_list)
    c_array.real[abs(c_array.real) < tol] = 0.0
    c_array.imag[abs(c_array.imag) < tol] = 0.0
    return c_array


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


def m_dot(a, b):
    return np.einsum('ijk, ijk->ij', a, b)


def calc_loop_path(solver, k_path, occ=None):
    wavefunction_array = []
    energy_array = []
    n = len(k_path)

    for i in range(n):
        print("{ii:n} / {iii:n}".format(ii=i + 1, iii=n), end="")
        solver.set_wave_vector(k_path[i])
        wavefunction_array.append(solver.eigenvectors[:, occ].T)
        energy_array.append(solver.eigenvalues)
        print(end="\r")
    print("")

    wfs2d = np.array(wavefunction_array, dtype=complex)
    all_phases = _one_berry_loop(wfs2d)
    return all_phases


def calc_loop_direction(solver, dir=0, occ=None, n=31, nd=None):
    if nd is None:
        nd = n
    k_u = solver.model.lattice.reciprocal_vectors()
    k_x, k_y = k_u[0][:2], k_u[1][:2]
    nx = n if dir == 1 else nd
    ny = nd if dir == 1 else n
    wavefunction_array, ee, kk = calc_wavefunction_region(solver, k=make_k_region(k_x, k_y, ny, nx,
                                                                                  k_add=-.5*(k_x+k_y)), occ=occ)
    wfs2d = np.array(wavefunction_array, dtype=complex)
    all_phases = [_one_berry_loop(wfs2d[:, i, :, :] if dir == 0 else wfs2d[i, :, :, :])
                  for i in range(n-1)]
    return _one_phase_cont(all_phases, all_phases[0]), ee, kk


def make_k_region(k_x=None, k_y=None, nx=10, ny=10, k_add=None):
    if k_x is None:
        k_x = np.array([1, 0])
    if k_y is None:
        k_y = np.array([0, 1])
    if k_add is None:
        k_add = np.array([0, 0])
    x_list, y_list = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    k_x_ret = x_list[:, :, np.newaxis] * k_x[np.newaxis, np.newaxis, :]
    k_y_ret = y_list[:, :, np.newaxis] * k_y[np.newaxis, np.newaxis, :]
    return k_x_ret + k_y_ret + k_add[np.newaxis, np.newaxis, :]


def add_phase(solver, w_vec, k_vec):
    x = solver.model.system.expanded_positions.x
    y = solver.model.system.expanded_positions.y
    phase = np.exp(1j * np.array([x, y]).T.dot(k_vec))
    return (w_vec.T * phase.conj()).T


def calc_phase(k, positions):
    distance_mesh = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    phase_mesh = np.exp(1j * np.dot(distance_mesh, k))
    return phase_mesh


def calc_wavefunction_region(solver, k=None, occ=None, verbose=False):
    if k is None:
        k_u = solver.model.lattice.reciprocal_vectors()
        k = make_k_region(k_x=k_u[0][:2], k_y=k_u[1][:2])
    nx, ny = np.shape(k)[:2]
    energy_array = []
    wavefunction_array = []
    for i in range(nx):
        energy_list = []
        wavefunction_list = []
        for j in range(ny):
            if verbose:
                print("i = {ii:n} / {iii:n} -- j = {jj:n} / {jjj:n}".format(ii=i + 1, iii=nx, jj=j + 1, jjj=ny), end="")
            k_vec = k[i][j]
            solver.set_wave_vector(k_vec)
            hamiltonian = np.array(solver.model.hamiltonian.todense()).T
            if occ is None:
                occ = np.arange(solver.eigenvalues.shape[0])
            positions = np.transpose([solver.model.system.expanded_positions.x,
                                      solver.model.system.expanded_positions.y])
            hamiltonian *= calc_phase(k_vec, positions)
            hamiltonian = _chop(hamiltonian)
            eigs, eigv = np.linalg.eigh(hamiltonian)
            wavefunction_list.append(eigv[:, occ].T)
            energy_list.append(eigs[occ])
            if verbose:
                print(end="\r")
        wavefunction_array.append(wavefunction_list)
        energy_array.append(energy_list)
    if verbose:
        print("")
    return wavefunction_array, energy_array, k


def calc_berry(wavefunction, rescale=True):
    wfs2d = np.array(wavefunction, dtype=complex)
    all_phases = np.array([[_one_berry_loop(
        np.array([wfs2d[i, j], wfs2d[i + 1, j], wfs2d[i + 1, j + 1], wfs2d[i, j + 1], wfs2d[i, j]], dtype=complex))
        for j in range(wfs2d.shape[1] - 1)] for i in range(wfs2d.shape[0] - 1)], dtype=float)
    if rescale:
        all_phases = all_phases / (np.max(all_phases) - np.min(all_phases)) * 2
    return all_phases


def plot_berry_square(solver, n=10, center=None, width=1, occ=None, rescale=True, factor=1):
    if center is None:
        center = np.eye(2) * width
    if occ is None:
        occ = [0]
    k_x = center + np.array([1, 0]) * width
    k_y = center + np.array([0, 1]) * width
    wavefunction, _, k = calc_wavefunction_region(solver,
                                                  k=make_k_region(k_x=k_x, k_y=k_y, nx=n, ny=n,
                                                                  k_add=center - width / 2),
                                                  occ=occ)
    all_phases = calc_berry(wavefunction, rescale=rescale) * factor
    solver.model.lattice.plot_brillouin_zone(decorate=False)
    return plot_berry_square_func(all_phases, k)


def plot_berry_square_func(all_phases, k, ax=None, p_max=1):
    if ax is None:
        ax = plt.gca()
    a = ax.contourf(k[1:, 1:, 0], k[1:, 1:, 1], all_phases, levels=np.linspace(-p_max, p_max, 256), cmap=colormap())
    width = k.max(axis=(0, 1)) - k.min(axis=(0, 1))
    center = (k.max(axis=(0, 1)) + k.min(axis=(0, 1))) / 2
    ax.set_ylim([center[1] - width[1] / 2, center[1] + width[1] / 2])
    ax.set_xlim([center[0] - width[0] / 2, center[0] + width[0] / 2])
    return a
