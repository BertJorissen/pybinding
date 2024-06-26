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
    def hexagonal_lattice(ons_1, ons_2, t_map, k_vec, phase=False):
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

        model = pb.Model(lat, pb.translational_symmetry(),
                         pb.force_complex_numbers(), pb.force_double_precision())
        if phase:
            model.add(pb.force_phase())
        model.set_wave_vector(k_vec)
        a1, a2 = lat.vectors
        pos_ab = np.array(lat.sublattices["B"].position-lat.sublattices["A"].position)
        ham = model.hamiltonian.todense()
        a_idx = model.system.to_hamiltonian_indices(model.lattice.sublattices["A"].unique_id)
        b_idx = model.system.to_hamiltonian_indices(model.lattice.sublattices["B"].unique_id)
        return ham, a1, a2, pos_ab, a_idx, b_idx

    def calc_anal(ons_1, ons_2, t_map, k_vec, a1, a2, pos_ab, a_idx, b_idx):
        d1 = pos_ab
        d2, d3 = d1 - a1, d1 - a1 - a2
        hop_term = t_map["t1"].T * np.exp(1j * k_vec @ d1)
        hop_term += t_map["t2"].T * np.exp(1j * k_vec @ d2)
        hop_term += t_map["t3"].T * np.exp(1j * k_vec @ d3)
        ons_term_a = ons_1.T + (t_map["t4"].T * np.exp(1j * k_vec @ a1) + np.conj(t_map["t4"].T).T * np.exp(1j * k_vec @ -a1))
        ons_term_b = ons_2.T + (t_map["t5"].T * np.exp(1j * k_vec @ (a1+a2)) + np.conj(t_map["t5"].T).T * np.exp(1j * k_vec @ -(a1+a2)))
        imax = len(a_idx) + len(b_idx)
        expected_ham = np.zeros((imax, imax), dtype=np.complex128)
        expected_ham[a_idx[0]:(a_idx[-1]+1), a_idx[0]:(a_idx[-1]+1)] = ons_term_a
        expected_ham[b_idx[0]:(b_idx[-1]+1), b_idx[0]:(b_idx[-1]+1)] = ons_term_b
        expected_ham[a_idx[0]:(a_idx[-1]+1), b_idx[0]:(b_idx[-1]+1)] = np.conj(hop_term).T
        expected_ham[b_idx[0]:(b_idx[-1]+1), a_idx[0]:(a_idx[-1]+1)] = hop_term
        return expected_ham

    # For the Hamiltonian, we hop from A to B with t1, this goes in the bottom left corner
    # USUAL CASE: <to|H|from>
    #      from:  |A> |B>
    #     to: <A| 0   t1*
    #     to: <B| t1  0
    # PYBINDING CASE: <from|H|to>
    #        to:  |A> |B>
    #   from: <A| 0   t1
    #   from: <B| t1*  0
    # => compare with the transpose for the complex valued Hamiltonian
    #             (this preserves the observables as this is just a phase change)


    # first test, just floats
    ons_a, ons_b = np.array([1]), np.array([2])
    hop_t = {
        't1': np.array([1]),
        't2': np.array([2]),
        't3': np.array([3]),
        't4': np.array([4]),
        't5': np.array([5])
    }
    k_vector = np.array([0.123, -4.567, 0.])
    hamilton, aa1, aa2, pab, ai, bi = hexagonal_lattice(ons_a, ons_b, hop_t, k_vector, False)
    eham = calc_anal(ons_a, ons_b, hop_t, k_vector, aa1, aa2, pab * 0, ai, bi)
    assert hamilton.shape == (2, 2)
    assert np.sum(np.abs(hamilton.T - eham)) < 1e-10


    hamilton, aa1, aa2, pab, ai, bi = hexagonal_lattice(ons_a, ons_b, hop_t, k_vector, True)
    eham = calc_anal(ons_a, ons_b, hop_t, k_vector, aa1, aa2, pab, ai, bi)
    assert hamilton.shape == (2, 2)
    assert np.sum(np.abs(hamilton.T - eham)) < 1e-10

    # second test, just floats, change phase
    model = pb.Model(hexagonal_lattice(ons_a, ons_b, hop_t), pb.translational_symmetry(),
                     pb.force_complex_numbers(), pb.force_double_precision(), pb.force_phase())
    model.set_wave_vector(k_vector)
    ham = model.hamiltonian.todense().T
    assert ham.shape == (2, 2)
    d1 = np.array(model.lattice.sublattices["B"].position-model.lattice.sublattices["A"].position)
    d2, d3 = d1 - a1, d1 - a1 - a2
    hop_term = hop_t["t1"] * np.exp(1j * k_vector @ d1)
    hop_term += hop_t["t2"] * np.exp(1j * k_vector @ d2)
    hop_term += hop_t["t3"] * np.exp(1j * k_vector @ d3)
    ons_term_a = ons_a + hop_t["t4"] * (np.exp(1j * k_vector @ a1) + np.exp(1j * k_vector @ -a1))
    ons_term_b = ons_b + hop_t["t5"] * (np.exp(1j * k_vector @ (a1+a2)) + np.exp(1j * k_vector @ -(a1+a2)))
    expected_ham = np.diag((ons_term_a, ons_term_b)) + np.array([[0, np.conj(hop_term)], [hop_term, 0]])
    assert np.sum(np.abs(ham - expected_ham)) < 1e-10

    # complex hoppings
    hop_t = {'t1': 1+.1j, 't2': 2+.2j, 't3': 3+.3j, 't4': 4+.4j, 't5': 5+.5j}
    model = pb.Model(hexagonal_lattice(ons_a, ons_b, hop_t), pb.translational_symmetry(),
                     pb.force_complex_numbers(), pb.force_double_precision())
    model.set_wave_vector(k_vector)
    ham = model.hamiltonian.todense().T
    assert ham.shape == (2, 2)
    d1 = 0 * a1
    d2, d3 = d1 - a1, d1 - a1 - a2
    hop_term = hop_t["t1"] * np.exp(1j * k_vector @ d1)
    hop_term += hop_t["t2"] * np.exp(1j * k_vector @ d2)
    hop_term += hop_t["t3"] * np.exp(1j * k_vector @ d3)
    ons_term_a = ons_a + hop_t["t4"] * np.exp(1j * k_vector @ a1) + np.conj(hop_t["t4"]) * np.exp(1j * k_vector @ -a1)
    ons_term_b = ons_b + hop_t["t5"] * np.exp(1j * k_vector @ (a1+a2)) + np.conj(hop_t["t5"]) * np.exp(1j * k_vector @ -(a1+a2))
    expected_ham = np.diag((ons_term_a, ons_term_b)) + np.array([[0, np.conj(hop_term)], [hop_term, 0]])
    assert np.sum(np.abs(ham - expected_ham)) < 1e-10

    # complex hoppings, with phase
    hop_t = {'t1': 1+.1j, 't2': 2+.2j, 't3': 3+.3j, 't4': 4+.4j, 't5': 5+.5j}
    model = pb.Model(hexagonal_lattice(ons_a, ons_b, hop_t), pb.translational_symmetry(),
                     pb.force_complex_numbers(), pb.force_double_precision(), pb.force_phase())
    model.set_wave_vector(k_vector)
    ham = model.hamiltonian.todense().T
    assert ham.shape == (2, 2)
    d1 = np.array(model.lattice.sublattices["B"].position-model.lattice.sublattices["A"].position)
    d2, d3 = d1 - a1, d1 - a1 - a2
    hop_term = hop_t["t1"] * np.exp(1j * k_vector @ d1)
    hop_term += hop_t["t2"] * np.exp(1j * k_vector @ d2)
    hop_term += hop_t["t3"] * np.exp(1j * k_vector @ d3)
    ons_term_a = ons_a + hop_t["t4"] * np.exp(1j * k_vector @ a1) + np.conj(hop_t["t4"]) * np.exp(1j * k_vector @ -a1)
    ons_term_b = ons_b + hop_t["t5"] * np.exp(1j * k_vector @ (a1+a2)) + np.conj(hop_t["t5"]) * np.exp(1j * k_vector @ -(a1+a2))
    expected_ham = np.diag((ons_term_a, ons_term_b)) + np.array([[0, np.conj(hop_term)], [hop_term, 0]])
    assert np.sum(np.abs(ham - expected_ham)) < 1e-10

    # multi-orbital case
    ons_a = np.array([[1, 2], [2, 3]])
    ons_b = np.array([[-1, -2, -3], [-2, -3, -4], [-3, -4, -5]])
    hop_t = {
        't1': np.array([[1, 2, 3], [4, 5, 6]]),
        't2': np.array([[7, 8, 9], [10, 11, 12]]),
        't3': np.array([[13, 14, 15], [16, 17, 18]]),
        't4': np.array([[19, 20], [21, 22]]),
        't5': np.array([[23, 24, 25], [26, 27, 28], [29, 30, 31]]),
    }
    model = pb.Model(hexagonal_lattice(ons_a.T, ons_b.T, hop_t), pb.translational_symmetry(),
                     pb.force_complex_numbers(), pb.force_double_precision())
    model.set_wave_vector(k_vector)
    ham = model.hamiltonian.todense().T
    assert ham.shape == (5, 5)
    a_idx = model.system.to_hamiltonian_indices(model.lattice.sublattices["A"].unique_id)
    b_idx = model.system.to_hamiltonian_indices(model.lattice.sublattices["B"].unique_id)
    d1 = 0 * a1
    d2, d3 = d1 - a1, d1 - a1 - a2
    hop_term = hop_t["t1"].T * np.exp(1j * k_vector @ d1)
    hop_term += hop_t["t2"].T * np.exp(1j * k_vector @ d2)
    hop_term += hop_t["t3"].T * np.exp(1j * k_vector @ d3)
    ons_term_a = ons_a + (hop_t["t4"].T * np.exp(1j * k_vector @ a1) + np.conj(hop_t["t4"].T).T * np.exp(1j * k_vector @ -a1))
    ons_term_b = ons_b + (hop_t["t5"].T * np.exp(1j * k_vector @ (a1+a2)) + np.conj(hop_t["t5"].T).T * np.exp(1j * k_vector @ -(a1+a2)))
    expected_ham = np.zeros((5, 5), dtype=np.complex128)
    expected_ham[a_idx[0]:(a_idx[-1]+1), a_idx[0]:(a_idx[-1]+1)] = ons_term_a
    expected_ham[b_idx[0]:(b_idx[-1]+1), b_idx[0]:(b_idx[-1]+1)] = ons_term_b
    expected_ham[a_idx[0]:(a_idx[-1]+1), b_idx[0]:(b_idx[-1]+1)] = np.conj(hop_term).T
    expected_ham[b_idx[0]:(b_idx[-1]+1), a_idx[0]:(a_idx[-1]+1)] = hop_term
    assert np.sum(np.abs(ham - expected_ham)) < 1e-10

    # multi-orbital case, phase
    ons_a = np.array([[1, 2], [2, 3]])
    ons_b = np.array([[-1, -2, -3], [-2, -3, -4], [-3, -4, -5]])
    hop_t = {
        't1': np.array([[1, 2, 3], [4, 5, 6]]),
        't2': np.array([[7, 8, 9], [10, 11, 12]]),
        't3': np.array([[13, 14, 15], [16, 17, 18]]),
        't4': np.array([[19, 20], [21, 22]]),
        't5': np.array([[23, 24, 25], [26, 27, 28], [29, 30, 31]]),
    }
    model = pb.Model(hexagonal_lattice(ons_a.T, ons_b.T, hop_t), pb.translational_symmetry(),
                     pb.force_complex_numbers(), pb.force_double_precision(), pb.force_phase())
    model.set_wave_vector(k_vector)
    ham = model.hamiltonian.todense().T
    assert ham.shape == (5, 5)
    a_idx = model.system.to_hamiltonian_indices(model.lattice.sublattices["A"].unique_id)
    b_idx = model.system.to_hamiltonian_indices(model.lattice.sublattices["B"].unique_id)
    d1 = np.array(model.lattice.sublattices["B"].position-model.lattice.sublattices["A"].position)
    d2, d3 = d1 - a1, d1 - a1 - a2
    hop_term = hop_t["t1"].T * np.exp(1j * k_vector @ d1)
    hop_term += hop_t["t2"].T * np.exp(1j * k_vector @ d2)
    hop_term += hop_t["t3"].T * np.exp(1j * k_vector @ d3)
    ons_term_a = ons_a + (hop_t["t4"].T * np.exp(1j * k_vector @ a1) + np.conj(hop_t["t4"].T).T * np.exp(1j * k_vector @ -a1))
    ons_term_b = ons_b + (hop_t["t5"].T * np.exp(1j * k_vector @ (a1+a2)) + np.conj(hop_t["t5"].T).T * np.exp(1j * k_vector @ -(a1+a2)))
    expected_ham = np.zeros((5, 5), dtype=np.complex128)
    expected_ham[a_idx[0]:(a_idx[-1]+1), a_idx[0]:(a_idx[-1]+1)] = ons_term_a
    expected_ham[b_idx[0]:(b_idx[-1]+1), b_idx[0]:(b_idx[-1]+1)] = ons_term_b
    expected_ham[a_idx[0]:(a_idx[-1]+1), b_idx[0]:(b_idx[-1]+1)] = np.conj(hop_term).T
    expected_ham[b_idx[0]:(b_idx[-1]+1), a_idx[0]:(a_idx[-1]+1)] = hop_term
    assert np.sum(np.abs(ham - expected_ham)) < 1e-10

    # multi-orbital case-complex-onsite
    ons_a = np.array([[1, 2], [2, 3]]) + 1j * np.array([[0, 1], [-1, 0]])
    ons_b = np.array([[-1, -2, -3], [-2, -3, -4], [-3, -4, -5]]) + 1j * np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    hop_t = {
        't1': np.array([[1, 2, 3], [4, 5, 6]]),
        't2': np.array([[7, 8, 9], [10, 11, 12]]),
        't3': np.array([[13, 14, 15], [16, 17, 18]]),
        't4': np.array([[19, 20], [21, 22]]),
        't5': np.array([[23, 24, 25], [26, 27, 28], [29, 30, 31]]),
    }
    model = pb.Model(hexagonal_lattice(ons_a.T, ons_b.T, hop_t), pb.translational_symmetry(),
                     pb.force_complex_numbers(), pb.force_double_precision())
    model.set_wave_vector(k_vector)
    ham = model.hamiltonian.todense().T
    assert ham.shape == (5, 5)
    a_idx = model.system.to_hamiltonian_indices(model.lattice.sublattices["A"].unique_id)
    b_idx = model.system.to_hamiltonian_indices(model.lattice.sublattices["B"].unique_id)
    d1 = 0 * a1
    d2, d3 = d1 - a1, d1 - a1 - a2
    hop_term = hop_t["t1"].T * np.exp(1j * k_vector @ d1)
    hop_term += hop_t["t2"].T * np.exp(1j * k_vector @ d2)
    hop_term += hop_t["t3"].T * np.exp(1j * k_vector @ d3)
    ons_term_a = ons_a + (hop_t["t4"].T * np.exp(1j * k_vector @ a1) + np.conj(hop_t["t4"].T).T * np.exp(1j * k_vector @ -a1))
    ons_term_b = ons_b + (hop_t["t5"].T * np.exp(1j * k_vector @ (a1+a2)) + np.conj(hop_t["t5"].T).T * np.exp(1j * k_vector @ -(a1+a2)))
    expected_ham = np.zeros((5, 5), dtype=np.complex128)
    expected_ham[a_idx[0]:(a_idx[-1]+1), a_idx[0]:(a_idx[-1]+1)] = ons_term_a
    expected_ham[b_idx[0]:(b_idx[-1]+1), b_idx[0]:(b_idx[-1]+1)] = ons_term_b
    expected_ham[a_idx[0]:(a_idx[-1]+1), b_idx[0]:(b_idx[-1]+1)] = np.conj(hop_term).T
    expected_ham[b_idx[0]:(b_idx[-1]+1), a_idx[0]:(a_idx[-1]+1)] = hop_term
    assert np.sum(np.abs(ham - expected_ham)) < 1e-10
    assert np.sum(np.abs(ham - expected_ham)) < 1e-10

    # multi-orbital case-complex-onsite, phase
    ons_a = np.array([[1, 2], [2, 3]]) + 1j * np.array([[0, 1], [-1, 0]])
    ons_b = np.array([[-1, -2, -3], [-2, -3, -4], [-3, -4, -5]]) + 1j * np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    hop_t = {
        't1': np.array([[1, 2, 3], [4, 5, 6]]),
        't2': np.array([[7, 8, 9], [10, 11, 12]]),
        't3': np.array([[13, 14, 15], [16, 17, 18]]),
        't4': np.array([[19, 20], [21, 22]]),
        't5': np.array([[23, 24, 25], [26, 27, 28], [29, 30, 31]]),
    }
    model = pb.Model(hexagonal_lattice(ons_a.T, ons_b.T, hop_t), pb.translational_symmetry(),
                     pb.force_complex_numbers(), pb.force_double_precision(), pb.force_phase())
    model.set_wave_vector(k_vector)
    ham = model.hamiltonian.todense().T
    assert ham.shape == (5, 5)
    a_idx = model.system.to_hamiltonian_indices(model.lattice.sublattices["A"].unique_id)
    b_idx = model.system.to_hamiltonian_indices(model.lattice.sublattices["B"].unique_id)
    d1 = np.array(model.lattice.sublattices["B"].position-model.lattice.sublattices["A"].position)
    d2, d3 = d1 - a1, d1 - a1 - a2
    hop_term = hop_t["t1"].T * np.exp(1j * k_vector @ d1)
    hop_term += hop_t["t2"].T * np.exp(1j * k_vector @ d2)
    hop_term += hop_t["t3"].T * np.exp(1j * k_vector @ d3)
    ons_term_a = ons_a + (hop_t["t4"].T * np.exp(1j * k_vector @ a1) + np.conj(hop_t["t4"].T).T * np.exp(1j * k_vector @ -a1))
    ons_term_b = ons_b + (hop_t["t5"].T * np.exp(1j * k_vector @ (a1+a2)) + np.conj(hop_t["t5"].T).T * np.exp(1j * k_vector @ -(a1+a2)))
    expected_ham = np.zeros((5, 5), dtype=np.complex128)
    expected_ham[a_idx[0]:(a_idx[-1]+1), a_idx[0]:(a_idx[-1]+1)] = ons_term_a
    expected_ham[b_idx[0]:(b_idx[-1]+1), b_idx[0]:(b_idx[-1]+1)] = ons_term_b
    expected_ham[a_idx[0]:(a_idx[-1]+1), b_idx[0]:(b_idx[-1]+1)] = np.conj(hop_term).T
    expected_ham[b_idx[0]:(b_idx[-1]+1), a_idx[0]:(a_idx[-1]+1)] = hop_term
    assert np.sum(np.abs(ham - expected_ham)) < 1e-10

    # complex-multi-orbital case
    ons_a = np.array([[1, 2], [2, 3]]) + 1j * np.array([[0, 1], [-1, 0]])
    ons_b = np.array([[-1, -2, -3], [-2, -3, -4], [-3, -4, -5]]) + 1j * np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    hop_t = {
        't1': np.array([[1, 2, 3], [4, 5, 6]]) + 1j * np.array([[.1, .2, .3], [.4, .5, .6]]),
        't2': np.array([[7, 8, 9], [10, 11, 12]]) + 1j * np.array([[.7, .8, .9], [.10, .11, .12]]),
        't3': np.array([[13, 14, 15], [16, 17, 18]]) + 1j * np.array([[.13, .14, .15], [.16, .17, .18]]),
        't4': np.array([[19, 20], [21, 22]]) + 1j * np.array([[.19, .20], [.21, .22]]),
        't5': np.array([[23, 24, 25], [26, 27, 28], [29, 30, 31]]) + 1j * np.array([[.23, .24, .25], [.26, .27, .28], [.29, .30, .31]]),
    }
    model = pb.Model(hexagonal_lattice(ons_a.T, ons_b.T, hop_t), pb.translational_symmetry(),
                     pb.force_complex_numbers(), pb.force_double_precision())
    model.set_wave_vector(k_vector)
    ham = model.hamiltonian.todense().T
    assert ham.shape == (5, 5)
    a_idx = model.system.to_hamiltonian_indices(model.lattice.sublattices["A"].unique_id)
    b_idx = model.system.to_hamiltonian_indices(model.lattice.sublattices["B"].unique_id)
    d1 = 0 * a1
    d2, d3 = d1 - a1, d1 - a1 - a2
    hop_term = hop_t["t1"].T * np.exp(1j * k_vector @ d1)
    hop_term += hop_t["t2"].T * np.exp(1j * k_vector @ d2)
    hop_term += hop_t["t3"].T * np.exp(1j * k_vector @ d3)
    ons_term_a = ons_a + (hop_t["t4"].T * np.exp(1j * k_vector @ a1) + np.conj(hop_t["t4"].T).T * np.exp(1j * k_vector @ -a1))
    ons_term_b = ons_b + (hop_t["t5"].T * np.exp(1j * k_vector @ (a1+a2)) + np.conj(hop_t["t5"].T).T * np.exp(1j * k_vector @ -(a1+a2)))
    expected_ham = np.zeros((5, 5), dtype=np.complex128)
    expected_ham[a_idx[0]:(a_idx[-1]+1), a_idx[0]:(a_idx[-1]+1)] = ons_term_a
    expected_ham[b_idx[0]:(b_idx[-1]+1), b_idx[0]:(b_idx[-1]+1)] = ons_term_b
    expected_ham[a_idx[0]:(a_idx[-1]+1), b_idx[0]:(b_idx[-1]+1)] = np.conj(hop_term).T
    expected_ham[b_idx[0]:(b_idx[-1]+1), a_idx[0]:(a_idx[-1]+1)] = hop_term
    assert np.sum(np.abs(ham - expected_ham)) < 1e-10

    # complex-multi-orbital case, phase
    ons_a = np.array([[1, 2], [2, 3]]) + 1j * np.array([[0, 1], [-1, 0]])
    ons_b = np.array([[-1, -2, -3], [-2, -3, -4], [-3, -4, -5]]) + 1j * np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    hop_t = {
        't1': np.array([[1, 2, 3], [4, 5, 6]]) + 1j * np.array([[.1, .2, .3], [.4, .5, .6]]),
        't2': np.array([[7, 8, 9], [10, 11, 12]]) + 1j * np.array([[.7, .8, .9], [.10, .11, .12]]),
        't3': np.array([[13, 14, 15], [16, 17, 18]]) + 1j * np.array([[.13, .14, .15], [.16, .17, .18]]),
        't4': np.array([[19, 20], [21, 22]]) + 1j * np.array([[.19, .20], [.21, .22]]),
        't5': np.array([[23, 24, 25], [26, 27, 28], [29, 30, 31]]) + 1j * np.array([[.23, .24, .25], [.26, .27, .28], [.29, .30, .31]]),
    }
    model = pb.Model(hexagonal_lattice(ons_a.T, ons_b.T, hop_t), pb.translational_symmetry(),
                     pb.force_complex_numbers(), pb.force_double_precision(), pb.force_phase())
    model.set_wave_vector(k_vector)
    ham = model.hamiltonian.todense().T
    assert ham.shape == (5, 5)
    a_idx = model.system.to_hamiltonian_indices(model.lattice.sublattices["A"].unique_id)
    b_idx = model.system.to_hamiltonian_indices(model.lattice.sublattices["B"].unique_id)
    d1 = np.array(model.lattice.sublattices["B"].position-model.lattice.sublattices["A"].position)
    d2, d3 = d1 - a1, d1 - a1 - a2
    hop_term = hop_t["t1"].T * np.exp(1j * k_vector @ d1)
    hop_term += hop_t["t2"].T * np.exp(1j * k_vector @ d2)
    hop_term += hop_t["t3"].T * np.exp(1j * k_vector @ d3)
    ons_term_a = ons_a + (hop_t["t4"].T * np.exp(1j * k_vector @ a1) + np.conj(hop_t["t4"].T).T * np.exp(1j * k_vector @ -a1))
    ons_term_b = ons_b + (hop_t["t5"].T * np.exp(1j * k_vector @ (a1+a2)) + np.conj(hop_t["t5"].T).T * np.exp(1j * k_vector @ -(a1+a2)))
    expected_ham = np.zeros((5, 5), dtype=np.complex128)
    expected_ham[a_idx[0]:(a_idx[-1]+1), a_idx[0]:(a_idx[-1]+1)] = ons_term_a
    expected_ham[b_idx[0]:(b_idx[-1]+1), b_idx[0]:(b_idx[-1]+1)] = ons_term_b
    expected_ham[a_idx[0]:(a_idx[-1]+1), b_idx[0]:(b_idx[-1]+1)] = np.conj(hop_term).T
    expected_ham[b_idx[0]:(b_idx[-1]+1), a_idx[0]:(a_idx[-1]+1)] = hop_term
    assert np.sum(np.abs(ham - expected_ham)) < 1e-10


