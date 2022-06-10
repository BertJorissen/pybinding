import numpy as np
import pybinding as pb

from matplotlib import use
from sys import platform
if platform == "linux" or platform == "linux2":
    use('Qt5Agg')
import matplotlib.pyplot as plt
import tmd

if __name__ == "__main__":
    a_lat = 1
    lat = pb.Lattice(a1=a_lat * np.array([1, 0]),
                     a2=a_lat * np.array([-.5, np.sqrt(3)/2]))
    lat.add_one_sublattice("C",
                           a_lat * np.array([0, np.sqrt(3)/3]),
                           np.array([[1, -2.5j, 3.5j],
                                     [2.5j, 2, 4.5j],
                                     [-3.5j, -4.5j, 4]]))
    lat.add_one_sublattice("A",
                           a_lat * np.array([0, 0]),
                           1)
    lat.add_one_sublattice("B",
                           a_lat * np.array([.5, np.sqrt(3)/6]),
                           np.array([[1, 0],
                                     [-0.,4]]))
    lat.add_one_hopping([0, 0], "A", "B", np.array([[1, 2]]))
    lat.add_one_hopping([-1, 0], "A", "B", np.array([[1, -2]]))
    lat.add_one_hopping([-1, -1], "A", "B", np.array([[0, 2]]))
    lat.add_one_hopping([0, 0], "A", "C", np.array([[.05, 2, 3]]))
    lat.add_one_hopping([0, -1], "A", "C", np.array([[1, 6, .3]]))
    lat.add_one_hopping([-1, -1], "A", "C", np.array([[.1, -0, 3]]))
    lat.add_one_hopping([0, 0], "B", "C", np.array([[1, 2, 0],
                                                    [4, 5, 6]]))
    lat.add_one_hopping([1, 0], "B", "C", np.array([[10, 5, 3],
                                                    [4, 9, -0.]]))
    lat.add_one_hopping([0, -1], "B", "C", np.array([[1, 0, 3],
                                                    [4, 5, 6]]))

    group_class = tmd.Group1Tmd11Band()
    lattice = lat if False else group_class.lattice()
    model = pb.Model(lattice,
                     pb.primitive(50, 50))
    #model.plot()
    #plt.figure()
    kpm = pb.kpm(model)
    dos = kpm.calc_dos(np.linspace(-10, 10, 100),
                       0.01,
                       10)
    print(dos.data)
    dos.plot()
    plt.show()
