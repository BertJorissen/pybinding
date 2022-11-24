import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt

import pybinding.solver
from pybinding.repository.graphene import a, t
from pybinding.results import make_path


def make_mutliband_graphene():
    lat = pb.Lattice(
        a1=a * np.array([1, 0]),
        a2=a * np.array([-.5, np.sqrt(3) / 2])
    )
    lat.add_sublattices(
        ("Aa", a * np.array([0, 0]), -20),
        ("Ab", a * np.array([.5, np.sqrt(3) / 6]), 20),
        ("Ba", a * np.array([0, 0]), 0),
        ("Bb", a * np.array([.5, np.sqrt(3) / 6]), 0),
        ("Ca", a * np.array([0, 0]), 20),
        ("Cb", a * np.array([.5, np.sqrt(3) / 6]), 20),
    )
    lat.add_hoppings(
        ([0, 0], "Aa", "Ab", t),
        ([-1, 0], "Aa", "Ab", t),
        ([-1, -1], "Aa", "Ab", t),
        ([0, 0], "Ba", "Bb", t),
        ([-1, 0], "Ba", "Bb", t),
        ([-1, -1], "Ba", "Bb", t),
        ([0, 0], "Ca", "Cb", t),
        ([-1, 0], "Ca", "Cb", t),
        ([-1, -1], "Ca", "Cb", t),
    )
    return lat


def get_bands(solver: pybinding.solver.Solver, k_path: pybinding.results.Path):
    eigenvalues = []
    from tqdm import tqdm
    for n_k, k_point in enumerate(tqdm(k_path)):
        solver.set_wave_vector(k_point)
        eigenvalues.append(solver.eigenvalues)
    return eigenvalues


def calc_band_graphene_pydacp(l=5):
    solver = pb.solver.dacp(
        pb.Model(
            make_mutliband_graphene(),
            pb.primitive(l, l),
            pb.translational_symmetry(l * a, l * a)
        ),
        window=[-3., 3],
        random_vectors=5,
        filter_order=12,
        tol=1e-4
    )
    bz = solver.model.lattice.brillouin_zone()
    gamma = bz[3] * 0
    k = bz[3] / l
    m = (bz[3] + bz[4]) / 2 / l
    path = make_path(gamma, k, m, gamma)
    return get_bands(solver, path), solver, path

def calc_band_graphene_arpack(l=5):
    solver = pb.solver.arpack(
        pb.Model(
            make_mutliband_graphene(),
            pb.primitive(l, l),
            pb.translational_symmetry(l * a, l * a)
        ),
        k=12,
        sigma=2
    )
    bz = solver.model.lattice.brillouin_zone()
    gamma = bz[3] * 0
    k = bz[3] / l
    m = (bz[3] + bz[4]) / 2 / l
    path = make_path(gamma, k, m, gamma)
    return get_bands(solver, path), solver, path

def calc_band_graphene_lapack(l=5):
    solver = pb.solver.lapack(
        pb.Model(
            make_mutliband_graphene(),
            pb.primitive(l, l),
            pb.translational_symmetry(l * a, l * a)
        )
    )
    bz = solver.model.lattice.brillouin_zone()
    gamma = bz[3] * 0
    k = bz[3] / l
    m = (bz[3] + bz[4]) / 2 / l
    path = make_path(gamma, k, m, gamma)
    return get_bands(solver, path), solver, path


if __name__ == "__main__":
    l = 20
    dacp = False
    arpack = True
    lapack = True
    if dacp:
        eigenvalues, solver_test, k_path_test = calc_band_graphene_pydacp(l=l)
    if arpack:
        eigenvalues_a, solver_test, k_path_test = calc_band_graphene_arpack(l=l)
    if lapack:
        eigenvalues_l, solver_test, k_path_test = calc_band_graphene_lapack(l=l)
    plt.close('all')
    fig = plt.figure(figsize=(12, 4))
    grid = plt.GridSpec(1, 3, hspace=0, wspace=0)
    plt.subplot(grid[0], title="Large-scale structure")
    solver_test.model.plot()
    plt.subplot(grid[1], title="Brillouin-zone")
    solver_test.model.lattice.plot_brillouin_zone()
    k_path_test.plot(point_labels=[r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"])
    plt.subplot(grid[2], title="Low-energy bands")

    if dacp:
        for n_energy, energy in enumerate(eigenvalues):
            plt.scatter(n_energy * np.ones(len(energy)), energy, color="C0", s=50)
    # bands.plot(point_labels=[r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"])
    if arpack:
        for n_energy, energy in enumerate(eigenvalues_a):
            plt.scatter(n_energy * np.ones(len(energy)), energy, color="C1", s=20)
    if lapack:
        for n_energy, energy in enumerate(eigenvalues_l):
            plt.scatter(n_energy * np.ones(len(energy)), energy, color="C2", s=5)
    plt.show()
