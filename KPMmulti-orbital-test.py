import numpy as np
import pybinding as pb

import matplotlib.pyplot as plt
import tmd


def toy_lat(length=50):
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
                                     [0.,4]]))
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
    lat.add_one_hopping([0, 1], "B", "B", np.array([[1, 0],
                                                    [4, 5]]))
    lat.add_one_hopping([1, 1], "C", "C", np.array([[1, 0, 0],
                                                    [0, 0, 0],
                                                    [0, 0, 1]]))
    return pb.Model(lat, pb.primitive(length, length))


def toy_lat_blok_shift(length=50):
    a_lat = 1
    lat = pb.Lattice(
        a1=a_lat * np.array([1, 0]),
        a2=a_lat * np.array([-.5, np.sqrt(3)/2]))
    ham_a = np.random.random((3, 3))
    ham_b = np.random.random((3, 3))
    ham_c = np.random.random((2, 2))
    ham_d = np.random.random((3, 3))
    lat.add_one_sublattice("A", a_lat * np.array([0, 0]), ham_a + ham_a.T)
    lat.add_one_sublattice("C", a_lat * np.array([0, np.sqrt(3)/3]), ham_c + ham_c.T)
    lat.add_one_sublattice("B", a_lat * np.array([0, 0]), ham_b + ham_b.T)
    lat.add_one_sublattice("D", a_lat * np.array([0, np.sqrt(3)/3]), ham_d + ham_d.T)

    # 1NN
    lat.add_one_hopping([0, 0], "A", "C", np.random.random((3, 2)))
    lat.add_one_hopping([0, 0], "B", "D", np.random.random((3, 3)))
    lat.add_one_hopping([0, -1], "A", "C", np.random.random((3, 2)))
    lat.add_one_hopping([0, -1], "B", "D", np.random.random((3, 3)))
    lat.add_one_hopping([-1, -1], "A", "C", np.random.random((3, 2)))
    lat.add_one_hopping([-1, -1], "B", "D", np.random.random((3, 3)))

    # 2NN
    lat.add_one_hopping([1, 0], "A", "A", np.random.random((3, 3)))
    lat.add_one_hopping([1, 0], "B", "B", np.random.random((3, 3)))
    lat.add_one_hopping([1, 0], "C", "C", np.random.random((2, 2)))
    lat.add_one_hopping([1, 0], "D", "D", np.random.random((3, 3)))
    lat.add_one_hopping([0, -1], "A", "A", np.random.random((3, 3)))
    lat.add_one_hopping([0, -1], "B", "B", np.random.random((3, 3)))
    lat.add_one_hopping([0, -1], "C", "C", np.random.random((2, 2)))
    lat.add_one_hopping([0, -1], "D", "D", np.random.random((3, 3)))
    lat.add_one_hopping([-1, -1], "A", "A", np.random.random((3, 3)))
    lat.add_one_hopping([-1, -1], "B", "B", np.random.random((3, 3)))
    lat.add_one_hopping([-1, -1], "C", "C", np.random.random((2, 2)))
    lat.add_one_hopping([-1, -1], "D", "D", np.random.random((3, 3)))
    return pb.Model(lat, pb.primitive(length, length))

def toy_lat_blok_shift_single(length=50):
    a_lat = 1
    lat = pb.Lattice(
        a1=a_lat * np.array([1, 0]),
        a2=a_lat * np.array([-.5, np.sqrt(3)/2]))
    ham_a = np.random.random(1)[0]
    ham_b = np.random.random(1)[0]
    ham_c = np.random.random(1)[0]
    ham_d = np.random.random(1)[0]
    lat.add_one_sublattice("A", a_lat * np.array([0, 0]), ham_a + ham_a.T)
    lat.add_one_sublattice("C", a_lat * np.array([0, np.sqrt(3)/3]), ham_c + ham_c.T)
    lat.add_one_sublattice("B", a_lat * np.array([0, 0]), ham_b + ham_b.T)
    lat.add_one_sublattice("D", a_lat * np.array([0, np.sqrt(3)/3]), ham_d + ham_d.T)

    # 1NN
    lat.add_one_hopping([0, 0], "A", "C", np.random.random(1)[0])
    lat.add_one_hopping([0, 0], "B", "D", np.random.random(1)[0])
    lat.add_one_hopping([0, -1], "A", "C", np.random.random(1)[0])
    lat.add_one_hopping([0, -1], "B", "D", np.random.random(1)[0])
    lat.add_one_hopping([-1, -1], "A", "C", np.random.random(1)[0])
    lat.add_one_hopping([-1, -1], "B", "D", np.random.random(1)[0])

    # 2NN
    lat.add_one_hopping([1, 0], "A", "A", np.random.random(1)[0])
    lat.add_one_hopping([1, 0], "B", "B", np.random.random(1)[0])
    lat.add_one_hopping([1, 0], "C", "C", np.random.random(1)[0])
    lat.add_one_hopping([1, 0], "D", "D", np.random.random(1)[0])
    lat.add_one_hopping([0, -1], "A", "A", np.random.random(1)[0])
    lat.add_one_hopping([0, -1], "B", "B", np.random.random(1)[0])
    lat.add_one_hopping([0, -1], "C", "C", np.random.random(1)[0])
    lat.add_one_hopping([0, -1], "D", "D", np.random.random(1)[0])
    lat.add_one_hopping([-1, -1], "A", "A", np.random.random(1)[0])
    lat.add_one_hopping([-1, -1], "B", "B", np.random.random(1)[0])
    lat.add_one_hopping([-1, -1], "C", "C", np.random.random(1)[0])
    lat.add_one_hopping([-1, -1], "D", "D", np.random.random(1)[0])
    return pb.Model(lat, pb.primitive(length, length))


def toy_lat_multi_orb_bump(mutli=True, length=10, height=1, sigma=1):
    from pybinding.repository.graphene import a_cc, a, t

    def gaussian_bump_strain(h, s):
        """Out-of-plane deformation (bump)"""
        @pb.site_position_modifier
        def displacement(x, y, z):
            dz = h * np.exp(-(x**2 + y**2) / s**2)  # gaussian
            return x, y, z + dz  # only the height changes

        @pb.hopping_energy_modifier
        def strained_hoppings(energy, x1, y1, z1, x2, y2, z2):
            d = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)  # strained neighbor distance
            return energy * np.exp(-3.37 * (d / a_cc - 1))  # see strain section
        return displacement, strained_hoppings

    lat = pb.Lattice(a1=[a, 0], a2=[a/2, a/2 * np.sqrt(3)])
    if mutli:
        lat.add_sublattices(
            ('A', [0, -a_cc/2], np.array([[1, 0.], [0., -1]])),
            ('B', [0,  a_cc/2], np.array([[1, 0.], [0., -1]]))
        )

        lat.register_hopping_energies({'t': np.array([[t, 0.], [0., t]])})

        lat.add_hoppings(
            ([0,  0], 'A', 'B', 't'),
            ([1, -1], 'A', 'B', 't'),
            ([0, -1], 'A', 'B', 't')
        )
    else:
        lat.add_sublattices(
            ('A1', [0, -a_cc/2], 1),
            ('A2', [0, -a_cc/2], -1),
            ('B1', [0,  a_cc/2], 1),
            ('B2', [0,  a_cc/2], -1)
        )

        lat.register_hopping_energies({'t': t})

        lat.add_hoppings(
            ([0,  0], 'A1', 'B1', 't'),
            ([1, -1], 'A1', 'B1', 't'),
            ([0, -1], 'A1', 'B1', 't'),
            ([0,  0], 'A2', 'B2', 't'),
            ([1, -1], 'A2', 'B2', 't'),
            ([0, -1], 'A2', 'B2', 't')
        )

    model_out = pb.Model(
        lat.with_offset([-a / 2, 0]),
        pb.primitive(length * 3, length * 3),
        gaussian_bump_strain(h=height, s=sigma)
    )
    return model_out


def gc_lat(so=False, length=50):
    gc = tmd.Group4Tmd11Band(single_orbital=so)
    model = pb.Model(
        gc.lattice(),
        pb.primitive(length, length),
        pb.translational_symmetry(length * gc.lattice_params.a, length * gc.lattice_params.a)
    )
    return model


def graphene_lat(length=50):
    from pybinding.repository.graphene import monolayer
    return pb.Model(monolayer(), pb.primitive(length, length))


def deformed_lat(length=50, height=1) -> pb.Model:
    def make_deformation(vectors, group_class, h=1):
        def displacement(r):
            return [np.sin(r_res * 2 * np.pi) for r_res in r]

        def displacement_d(r, v_i=0):
            return [np.cos(r_res * 2 * np.pi) * 2 * np.pi for r_res in r]

        def dz_i(r, v_i=0):
            d_o = displacement(r)
            d_d = displacement_d(r, v_i)
            return height * (d_d[0] * d_o[1] + d_o[0] * d_d[1])

        @pb.site_position_modifier
        def dz(x, y, z):
            r_in = np.array([x, y, z])
            r = np.linalg.solve(vectors.T, r_in)
            return x, y, z + height * np.dot([1, 1, 0], displacement(r))

        @pb.hopping_energy_modifier
        def strained_hoppings(energy, x1, y1, z1, x2, y2, z2, hop_id):
            r_in = np.array([x1.flatten() + x2.flatten(), y1.flatten() + y2.flatten(), z1.flatten() * 0]) / 2
            r = np.linalg.solve(vectors.T, r_in)
            ux = dz_i(r, 0)
            uy = dz_i(r, 1)
            uxx = (ux * ux) / 2
            uxy = (ux * uy) / 2
            uyy = (uy * uy) / 2
            return energy + group_class.straining_hopping(uxx, uyy, uxy, hop_id=hop_id)

        @pb.onsite_energy_modifier
        def strained_onsite(energy, x, y, z, sub_id):
            r_in = np.array([x.flatten(), y.flatten(), z.flatten() * 0])
            r = np.linalg.solve(vectors.T, r_in)
            ux = dz_i(r, 0)
            uy = dz_i(r, 1)
            uxx = (ux * ux) / 2
            uxy = (ux * uy) / 2
            uyy = (uy * uy) / 2
            return energy + group_class.strained_onsite(uxx, uyy, uxy, sub_id=sub_id)
        return dz, strained_hoppings, strained_onsite
    gc = tmd.TestClass(single_orbital=False)
    large_vectors = np.array([*gc.lattice().vectors, [0, 0, 1]]) * length
    model = pb.Model(
        gc.lattice(),
        pb.primitive(length, length),
        make_deformation(
            large_vectors,
            gc,
            height
        ),
        pb.translational_symmetry(np.linalg.norm(large_vectors[0]), np.linalg.norm(large_vectors[1]))
    )
    return model


if __name__ == "__main__":
    model = gc_lat(so=False, length=2)
    # model = toy_lat_blok_shift(100)
    # model = toy_lat_blok_shift_single(1)
    # model = toy_lat_blok_shift(1)
    # testbool = True
    # model = toy_lat_multi_orb_bump(testbool, 100, 2, 2)
    kpm = pb.kpm(model)
    # dos = kpm.calc_dos(np.linspace(-10, 10, 1000), 0.02, 10)
    # ldosm = kpm.calc_ldos(np.linspace(-10, 10, 1000), 0.02, [0, 0], "Mo")
    # ldoss = kpm.calc_ldos(np.linspace(-10, 10, 1000), 0.02, [0, 0], "S")
    sldos = kpm.calc_spatial_ldos(np.linspace(-10, 10, 1000), 0.05, pb.circle(radius=1.2))
    # dldos = kpm.deferred_ldos(np.linspace(-10, 10, 1000), 0.02, [0, 0], "Mo")
    # greens = kpm.calc_greens(10, 20, np.linspace(-10, 10, 100), .1)
    # sigma = kpm.calc_conductivity(chemical_potential=np.linspace(-1.5, 1.5, 300),
    #                               broadening=0.1, direction="xx", temperature=0,
    #                               volume=20**2, num_random=10)
    # sigma.data *= 4  # to account for spin and valley degeneracy
    import matplotlib
    matplotlib.use('qt5agg')
    # if testbool:
    #    sldos.data = sldos.data[:, :model.system.num_sites]
    # sldos.structure_map(-.75).plot()
    # dos.plot()
    # testbool = False
    model = gc_lat(so=False, length=7)
    kpm = pb.kpm(model)
    sldos2 = kpm.calc_spatial_ldos(np.linspace(-10, 10, 1000), 0.05, pb.circle(radius=1.2))

