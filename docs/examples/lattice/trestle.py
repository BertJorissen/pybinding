"""One dimensional lattice with complex hoppings"""
import pybinding as pb
import matplotlib.pyplot as plt

pb.pltutils.use_style()


def trestle(d=0.2, t1=0.8 + 0.6j, t2=2):
    lat = pb.Lattice(a1=1.3*d)
    lat.add_sublattices(
        ('A', [0,   0], 0),
        ('B', [d/2, d], 0)
    )
    lat.add_hoppings(
        (0, 'A', 'B', t1),
        (1, 'A', 'B', t1),
        (1, 'A', 'A', t2),
        (1, 'B', 'B', t2)
    )
    return lat

lattice = trestle()
lattice.plot()
plt.show()

lattice.plot_brillouin_zone()
plt.show()