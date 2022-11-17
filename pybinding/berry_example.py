import numpy as np
import pybinding as pb
import matplotlib.pyplot as plt
import pybinding.berry
import pybinding.repository.group6_tmd


def run_code():
    solver = pb.solver.lapack(pb.Model(pybinding.repository.group6_tmd.monolayer_3band(name="MoS2"), pb.translational_symmetry()))
    occ=[0]
    n=400
    nd=400
    nkki = int(2*n / 12)
    nkkd = int(1 * n / 12)
    nkx = []
    nky = []
    nkx.append(nkkd+np.arange(nkki))
    nky.append(nkkd+0 * np.arange(nkki))
    nkx.append(nkkd+nkki + 0 * np.arange(nkki))
    nky.append(nkkd+np.arange(nkki))
    nkx.append(nkkd+nkki - np.arange(nkki))
    nky.append(nkkd+nkki + 0 * np.arange(nkki))
    nkx.append(nkkd+0 * np.arange(nkki))
    nky.append(nkkd+nkki - np.arange(nkki))
    nkx = np.array(nkx).flatten()
    nky = np.array(nky).flatten()
    k_u = solver.model.lattice.reciprocal_vectors()
    k_x, k_y = k_u[0][:2], k_u[1][:2]
    nx = n if dir == 1 else nd
    ny = nd if dir == 1 else n
    wavefunction_array, ee, kk = pybinding.berry.calc_wavefunction_region(solver, k=pybinding.berry.make_k_region(k_x, k_y, ny, nx, k_add=-.5*(k_x+k_y)), occ=occ)
    wfs2d = np.array(wavefunction_array, dtype=complex)
    wflist = [wfs2d[nx, ny, :, :] for nx, ny in zip(nkx, nky)]
    wflist.append(wfs2d[nkx[0], nky[0], :, :])
    result = pybinding.berry._one_berry_loop(np.array(wflist, dtype=complex))
    print(result)
    plt.scatter(np.array(kk[:, :, 0]).flatten(), np.array(kk[:, :, 1]).flatten())
    solver.model.lattice.plot_brillouin_zone()
    plt.scatter(np.array([kk[nkxi, nkyi, 0] for nkxi, nkyi in zip(nkx, nky)]).flatten(), np.array([kk[nkxi, nkyi, 1] for nkxi, nkyi in zip(nkx, nky)]).flatten())

if __name__ == "__main__":
    run_code()
