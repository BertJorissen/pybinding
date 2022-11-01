import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository.graphene import monolayer, a, t

# import statement to run under WSL, it can be that you enabled QT5, then use 'use("Qt5Agg")'
from matplotlib import use
use("TkAGG")

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


def calc_band_graphene_pydacp(l=5):
    solver = pb.solver.dacp(
        pb.Model(
            make_mutliband_graphene(),
            pb.primitive(l, l),
            pb.translational_symmetry(l * a, l * a)
        ),
        window=[-8.5, 8.5],
        random_vectors=20,
        filter_order=12,
        tol=1e-3
    )
    use("TkAgg")
    bz = solver.model.lattice.brillouin_zone()
    gamma = bz[3] * 0
    k = bz[3] / l
    m = (bz[3] + bz[4]) / 2 / l
    return solver.calc_bands(gamma, k, m, gamma), solver


if __name__ == "__main__":
    bands, solver = calc_band_graphene_pydacp(l=4)
    plt.close('all')
    fig = plt.figure(figsize=(12, 4))
    grid = plt.GridSpec(1, 3, hspace=0, wspace=0)
    plt.subplot(grid[0], title="Large-scale structure")
    solver.model.plot()
    plt.subplot(grid[1], title="Brillouin-zone")
    solver.model.lattice.plot_brillouin_zone()
    bands.k_path.plot(point_labels=[r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"])
    plt.subplot(grid[2], title="Low-energy bands")
    bands.plot(point_labels=[r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"])
    plt.show()
