import pytest

import numpy as np
import pybinding as pb
from pybinding.repository import graphene
from scipy.sparse import csr_matrix


def point_to_same_memory(a, b):
    """Check if two numpy arrays point to the same data in memory"""
    return a.data == b.data


@pytest.fixture(scope='module')
def model():
    return pb.Model(graphene.monolayer())


def test_api():
    lattice = graphene.monolayer()
    shape = pb.rectangle(1)
    model = pb.Model(lattice, shape)

    assert model.lattice is lattice
    assert model.shape is shape

    # empty sequences are no-ops
    model.add(())
    model.add([])

    with pytest.raises(RuntimeError) as excinfo:
        model.add(None)
    assert "None" in str(excinfo.value)


def test_report(model):
    report = model.report()
    assert "2 lattice sites" in report
    assert "2 non-zero values" in report


def test_hamiltonian(model):
    """Must be in the correct format and point to memory allocated in C++ (no copies)"""
    h = model.hamiltonian
    assert isinstance(h, csr_matrix)
    assert h.dtype == np.float32
    assert h.shape == (2, 2)
    assert pytest.fuzzy_equal(h.data, [graphene.t] * 2)
    assert pytest.fuzzy_equal(h.indices, [1, 0])
    assert pytest.fuzzy_equal(h.indptr, [0, 1, 2])

    assert h.data.flags['OWNDATA'] is False
    assert h.data.flags['WRITEABLE'] is False

    with pytest.raises(ValueError) as excinfo:
        h.data += 1
    assert "read-only" in str(excinfo.value)

    h2 = model.hamiltonian
    assert h2.data is not h.data
    assert point_to_same_memory(h2.data, h.data)


def test_multiorbital_hamiltonian():
    """For multi-orbital lattices the Hamiltonian size is larger than the number of sites"""
    def lattice():
        lat = pb.Lattice([1])
        lat.add_sublattices(("A", [0], [[1, 3j],
                                        [0, 2]]))
        lat.register_hopping_energies({
            "t22": [[0, 1],
                    [2, 3]],
            "t11": 1,  # incompatible hopping - it's never used so it shouldn't raise any errors
        })
        lat.add_hoppings(([1], "A", "A", "t22"))
        return lat

    model = pb.Model(lattice(), pb.primitive(3))
    h = model.hamiltonian.toarray()

    assert model.system.num_sites == 3
    assert h.shape[0] == 6
    assert pytest.fuzzy_equal(h, h.T.conjugate())
    assert pytest.fuzzy_equal(h[:2, :2], h[-2:, -2:])
    assert pytest.fuzzy_equal(h[:2, :2], [[  1, 3j],
                                          [-3j, 2]])
    assert pytest.fuzzy_equal(h[:2, 2:4], [[0, 1],
                                           [2, 3]])

    @pb.onsite_energy_modifier
    def onsite(energy, x, sub_id):
        return 3 * energy + sub_id.eye * 0 * x

    @pb.hopping_energy_modifier
    def hopping(energy):
        return 2 * energy

    model = pb.Model(lattice(), pb.primitive(3), onsite, hopping)
    h = model.hamiltonian.toarray()

    assert model.system.num_sites == 3
    assert h.shape[0] == 6
    assert pytest.fuzzy_equal(h, h.T.conjugate())
    assert pytest.fuzzy_equal(h[:2, :2], h[-2:, -2:])
    assert pytest.fuzzy_equal(h[:2, :2], [[  3, 9j],
                                          [-9j,  6]])
    assert pytest.fuzzy_equal(h[:2, 2:4], [[0, 2],
                                           [4, 6]])
    assert pytest.fuzzy_equal(h[2:4, 4:6], [[0, 2],
                                            [4, 6]])

    def lattice_with_zero_diagonal():
        lat = pb.Lattice([1])
        lat.add_sublattices(("A", [0], [[0, 3j],
                                        [0,  0]]))
        return lat

    model = pb.Model(lattice_with_zero_diagonal(), pb.primitive(3))
    h = model.hamiltonian.toarray()

    assert model.system.num_sites == 3
    assert h.shape[0] == 6
    assert pytest.fuzzy_equal(h, h.T.conjugate())
    assert pytest.fuzzy_equal(h[:2, :2], h[-2:, -2:])
    assert pytest.fuzzy_equal(h[:2, :2], [[0, 3j],
                                          [-3j, 0]])


def test_complex_multiorbital_hamiltonian():
    def checkerboard_lattice(delta, t):
        lat = pb.Lattice(a1=[1, 0], a2=[0, 1])
        lat.add_sublattices(('A', [0, 0],    -delta),
                            ('B', [1/2, 1/2], delta))
        lat.add_hoppings(
            ([0,   0], 'A', 'B', t),
            ([0,  -1], 'A', 'B', t),
            ([-1,  0], 'A', 'B', t),
            ([-1, -1], 'A', 'B', t),
        )
        return lat

    hopp_t = np.array([[2 + 2j, 3 + 3j],
                       [4 + 4j, 5 + 5j]])      # multi-orbital hopping
    onsite_en = np.array([[1, 1j], [-1j, 1]])  # onsite energy

    model = pb.Model(checkerboard_lattice(onsite_en, hopp_t),
                     pb.translational_symmetry(True, True))
    h = model.hamiltonian.toarray()

    assert model.system.num_sites == 2
    assert h.shape[0] == 4
    assert pytest.fuzzy_equal(h, h.T.conjugate())  # check if Hermitian
    assert pytest.fuzzy_equal(h[:2, :2], -h[-2:, -2:])  # onsite energy on A and B is opposite
    assert pytest.fuzzy_equal(h[:2, 2:4], 4 * hopp_t)  # hopping A <-> B is 4 * hopp_t


def test_wave_vector():
    def hexagonal_lattice(ons_1, ons_2, t_map):
        lat = pb.Lattice(a1=[1, 0], a2=[-1/2, np.sqrt(3)/2])
        lat.add_sublattices(('A', [0, 0],    ons_1),
                            ('B', [1/2, np.sqrt(3)/6], ons_2))
        lat.register_hopping_energies(t_map)
        lat.add_hoppings(
            ([0, 0], 'A', 'B', 't1'),
            ([-1, 0], 'A', 'B', 't2'),
            ([-1, -1], 'A', 'B', 't3'),
            ([[1, 0], 'A', 'A', 't4']),
            ([[1, 1], 'B', 'B', 't5']),
        )
        return lat

    # first test, just floats
    ons_a, ons_b, hop_t = 1, 2, {'t1': 1, 't2': 2, 't3': 3, 't4': 4, 't5': 5}
    model = pb.Model(hexagonal_lattice(ons_a, ons_b, hop_t), pb.translational_symmetry(),
                     pb.force_complex_numbers(), pb.force_double_precision())
    k_vector = np.array([0.123, -4.567, 0.])
    model.set_wave_vector(k_vector)
    ham = model.hamiltonian.todense()
    assert ham.shape == (2, 2)
    a1, a2 = model.lattice.vectors
    d1 = 0 * a1
    d2, d3 = d1 - a1, d1 - a1 - a2
    hop_term = hop_t["t1"] * np.exp(1j * k_vector @ d1)
    hop_term += hop_t["t2"] * np.exp(1j * k_vector @ d2)
    hop_term += hop_t["t3"] * np.exp(1j * k_vector @ d3)
    ons_term_a = ons_a + hop_t["t4"] * (np.exp(1j * k_vector @ a1) + np.exp(1j * k_vector @ -a1))
    ons_term_b = ons_b + hop_t["t5"] * (np.exp(1j * k_vector @ (a1+a2)) + np.exp(1j * k_vector @ -(a1+a2)))
    expected_ham = np.diag((ons_term_a, ons_term_b)) + np.array([[0, hop_term], [np.conj(hop_term), 0]])
    assert np.sum(np.abs(ham - expected_ham)) < 1e-10

    # second test, just floats, change phase
    model = pb.Model(hexagonal_lattice(ons_a, ons_b, hop_t), pb.translational_symmetry(),
                     pb.force_complex_numbers(), pb.force_double_precision(), pb.force_phase())
    model.set_wave_vector(k_vector)
    ham = model.hamiltonian.todense()
    assert ham.shape == (2, 2)
    a1, a2 = model.lattice.vectors
    d1 = np.array(model.lattice.sublattices["B"].position-model.lattice.sublattices["A"].position)
    d2, d3 = d1 - a1, d1 - a1 - a2
    hop_term = hop_t["t1"] * np.exp(1j * k_vector @ d1)
    hop_term += hop_t["t2"] * np.exp(1j * k_vector @ d2)
    hop_term += hop_t["t3"] * np.exp(1j * k_vector @ d3)
    ons_term_a = ons_a + hop_t["t4"] * (np.exp(1j * k_vector @ a1) + np.exp(1j * k_vector @ -a1))
    ons_term_b = ons_b + hop_t["t5"] * (np.exp(1j * k_vector @ (a1+a2)) + np.exp(1j * k_vector @ -(a1+a2)))
    expected_ham = np.diag((ons_term_a, ons_term_b)) + np.array([[0, hop_term], [np.conj(hop_term), 0]])
    print(ham, expected_ham)
    assert np.sum(np.abs(ham - expected_ham)) < 1e-10
